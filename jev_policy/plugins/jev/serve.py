#!/usr/bin/env python3
"""Serve Jev-Omni over HTTP so the LIBERO harness can reach it.

Why a server at all: Jev-Omni needs torch >= 2.10 and transformers 5.17, while
LIBERO needs numpy < 2, mujoco 2.3.7 and robosuite 1.4.1. Those two sets cannot
share an interpreter, so the model runs in its own environment and the harness
talks to it -- the same split openpi uses for its policy server.

    # in the jev-omni env
    python scripts/serve_jev_omni.py --shard          # 2x24GB GPUs
    # in the jev-libero env
    python scripts/eval_libero.py --planner.backend jev-omni

POST /predict  {state, question, options[], image_png_b64?}
            -> {prediction, prediction_index, confidence, probabilities}
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import logging
import os
import tempfile
import time

logger = logging.getLogger("jev-omni-server")

MODEL_ID = "akhilaaa3/Jev-Omni"
BASE_ID = "google/gemma-4-12B-it"


def load_classifier(shard: bool, device: str, model_path=None, base_path=None):
    """Load Jev-Omni, optionally sharded across several GPUs.

    The upstream `load_jev_omni()` loads the backbone in **float32** (~48GB for
    a 12B model) onto one device before casting it down, then loads the base
    multimodal model on top. That needs a single ~80GB card. `--shard` rebuilds
    the same assembly with the backbone loaded directly in bf16 and spread over
    the visible GPUs, which is what makes it fit on 2x24GB.

    The difference from upstream is that non-Linear parameters are bf16 rather
    than fp32 -- upstream casts every Linear to bf16 immediately after loading
    anyway, so this is close but not bit-identical.
    """
    import importlib.util
    import sys
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download

    # Explicit paths let the weights live anywhere -- e.g. a NAS share, where
    # the hub's symlinked cache layout cannot be used because os.symlink fails
    # with EIO on CIFS.
    merged = model_path or snapshot_download(MODEL_ID)
    sys.path.insert(0, merged)

    if not shard:
        from jev_omni import load_jev_omni

        return load_jev_omni(device=device)

    # --- sharded assembly -------------------------------------------------
    # Order matters. The obvious sequence -- load the fine-tuned decoder, then
    # the multimodal base -- holds two full 12B towers at once (~48GB) and fills
    # both 3090s before the base is a tenth loaded. Instead: load the base,
    # throw away the text tower we are about to replace, and only then load the
    # decoder into the space that frees. Peak is one-and-a-bit towers, not two.
    import gc

    from transformers import AutoConfig, AutoProcessor

    spec = importlib.util.spec_from_file_location(
        "jev_omni_text_loader", Path(merged) / "load_model.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)

    import importlib as _importlib

    import transformers
    from jev_omni import JevOmni, _find_backbone

    base = base_path or snapshot_download(BASE_ID)
    logger.info("loading multimodal base %s", BASE_ID)
    base_config = AutoConfig.from_pretrained(base)
    model = (
        getattr(transformers, base_config.architectures[0])
        .from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
        .eval()
    )

    path, old = _find_backbone(model)
    parent, _, name = path.rpartition(".")
    holder = model.get_submodule(parent) if parent else model
    logger.info("dropping the base text tower at %s to make room", path)
    setattr(holder, name, torch.nn.Identity())
    del old
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("after dropping base text tower")

    config = json.loads((Path(merged) / "decision_config.json").read_text())
    backbone_cls = getattr(
        _importlib.import_module(config["backbone_module"]), config["backbone_class"]
    )
    logger.info("loading decision backbone in bf16, sharded")
    decoder = backbone_cls.from_pretrained(
        Path(merged) / "backbone",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()

    buffers_path = Path(merged) / "runtime_buffers.pt"
    if buffers_path.exists():
        buffers = torch.load(buffers_path, map_location="cpu", weights_only=True)
        for buffer_name, value in buffers.items():
            buffer_parent, _, attr = buffer_name.rpartition(".")
            module = decoder.get_submodule(buffer_parent) if buffer_parent else decoder
            # Each buffer must land on the device of the module that uses it.
            # Sharding puts layer N on a different GPU than layer 0, so sending
            # every buffer to the first parameter's device -- which a single-GPU
            # loader gets away with -- makes ops like
            # `hidden_states *= self.layer_scalar` fail with a device mismatch
            # partway up the stack.
            try:
                target = next(module.parameters()).device
            except StopIteration:
                target = next(decoder.parameters()).device
            setattr(module, attr, value.to(target))

    head = loader.Head256(config["hidden_size"]).to(device).eval()
    head.load_state_dict(
        torch.load(Path(merged) / "head.pt", map_location=device, weights_only=True)
    )

    setattr(holder, name, decoder)
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("assembled")
    return JevOmni(model, head, AutoProcessor.from_pretrained(base), decoder, device)


def _log_vram(label: str) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    used = [
        f"cuda:{i} {torch.cuda.memory_allocated(i) / 1e9:.1f}GB"
        for i in range(torch.cuda.device_count())
    ]
    logger.info("VRAM %s: %s", label, ", ".join(used))


def make_handler(classifier):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # quieter than the default access log
            pass

        def _send(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": "ok"})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/predict":
                self._send({"error": "not found"}, 404)
                return

            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))

            media_path = None
            try:
                if request.get("image_png_b64"):
                    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    handle.write(base64.b64decode(request["image_png_b64"]))
                    handle.close()
                    media_path = handle.name

                started = time.perf_counter()
                result = classifier.predict(
                    state=request["state"],
                    question=request["question"],
                    options=request["options"],
                    media=media_path,
                    modality="image" if media_path else "text",
                )
                result["latency_s"] = time.perf_counter() - started
                self._send(result)
            except Exception as exc:  # surface to the client rather than 500-ing blind
                logger.exception("predict failed")
                self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
            finally:
                if media_path:
                    os.unlink(media_path)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", help="Local Jev-Omni directory (skips snapshot_download)")
    parser.add_argument("--base-path", help="Local google/gemma-4-12B-it directory")
    parser.add_argument(
        "--shard",
        action="store_true",
        help="Load the backbone in bf16 across all visible GPUs. Required below ~80GB VRAM.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("loading Jev-Omni (shard=%s); this takes a few minutes", args.shard)
    classifier = load_classifier(args.shard, args.device, args.model_path, args.base_path)
    logger.info("ready on http://%s:%d", args.host, args.port)

    http.server.ThreadingHTTPServer((args.host, args.port), make_handler(classifier)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
