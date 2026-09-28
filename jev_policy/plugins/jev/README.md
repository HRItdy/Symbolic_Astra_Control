# Jev plugin (currently off)

Two decision-layer backends, isolated here so the core harness neither imports
nor depends on them:

| backend | model | needs |
|---|---|---|
| `jev` | TypeSafe's hosted Jev System One | `TYPESAFE_API_KEY`, `pip install typesafe-sdk` |
| `jev-omni` | community `akhilaaa3/Jev-Omni` (Gemma-4-12B-IT finetune) | local GPUs, ~72GB weights |

## Turning it back on

Either set the environment variable:

```bash
JEV_POLICY_PLUGINS=jev python scripts/eval_libero.py --planner.backend jev-omni
```

or make it permanent by editing `jev_policy/plugins/__init__.py`:

```python
ENABLED_BY_DEFAULT = ("jev",)
```

## Jev-Omni server

The weights are already on the NAS, so no re-download is needed:

```bash
conda activate jev-omni
python -m jev_policy.plugins.jev.serve --shard \
    --model-path ~/synology/hf-cache/hub/models--akhilaaa3--Jev-Omni/snapshots/c050d51354147985d13286cf4acf90f562f2c631 \
    --base-path  ~/synology/jev-omni/gemma-4-12B-it
```

Then, in the `jev-libero` env:

```bash
JEV_POLICY_PLUGINS=jev python scripts/eval_libero.py --planner.backend jev-omni \
    --task-suite-name libero_spatial --num-trials-per-task 5
```

Other entry points: `python -m jev_policy.plugins.jev.check` (hosted connectivity
and latency) and `python -m jev_policy.plugins.jev.fetch` (download + verify
weights). The setup traps — CUDA build, missing processor deps, shard load order,
per-layer buffers, CIFS symlinks — are documented in the main README.

## Where it got to

`libero_spatial`, 5 trials/task: **36%** for `jev-omni` against **78%** for the
scripted `heuristic`. The plan *order* was usually right; target selection was
the weak point. See the main README for the breakdown and the next fixes.
