# jev-policy

Controlling a Panda arm through the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
benchmark with **Jev**, TypeSafe AI's System One model, as the decision layer.

## Why the architecture looks like this

Jev is not a vision-language-action model, and it is important to be clear about
that before reading the code. Per [TypeSafe's docs](https://docs.typesafe.ai/sdk/python)
it is **text-only**, answers in **70–500ms**, and returns **typed decisions**
(`Choice` / `Score` / `Noul`) rather than continuous values. LIBERO wants a
7-DoF end-effector delta at 20Hz, from pixels. Jev cannot produce that, and
pretending otherwise would just be an expensive way to fail.

What Jev is good at is the middle layer — which is exactly how TypeSafe frames
it for robotics: the model decides, and IK and physics stay in code. So:

```
  ┌─────────────────────────────────────────────────────────┐
  │ perception/scene_state.py    MuJoCo state → symbolic text│   every step, code
  │   "holding: nothing                                      │
  │    bowl_1: distance=21cm offset(right=+5,forward=-10)"   │
  └────────────────────────┬─────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────┐
  │ policies/jev_planner.py      Jev: typed decision         │   ~5-15 calls/episode
  │   Choice(skill)  Choice(target)  Noul(task_complete)     │   70-500ms each
  │   → ("descend", "bowl_1", confidence=0.83)               │
  └────────────────────────┬─────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────┐
  │ skills/primitives.py         closed-loop motion          │   20Hz, code only
  │   → OSC_POSE delta [dx,dy,dz,drx,dry,drz,gripper]        │
  └─────────────────────────────────────────────────────────┘
```

The planner is called **only at skill boundaries**, so Jev's latency costs a few
seconds per episode instead of making real-time control impossible.

Two properties of the model are load-bearing here, not decoration:

- **Targets are enumerated from the live scene.** The `Choice` criteria are built
  from the objects actually present in this episode, so a hallucinated object
  name is unrepresentable — the failure mode you'd normally validate against
  simply cannot occur.
- **Confidence is calibrated.** A decision below `--planner.min-confidence`
  triggers a retreat-and-replan instead of a probably-wrong grasp. That
  threshold is the main thing worth tuning.

## Backends

| backend | what it does | needs |
|---|---|---|
| `heuristic` | scripted pick-and-place sequence | nothing |
| `random` | random legal skills; smoke test | nothing |
| `astra` | **GPT-6-Astra drives the robot directly — no skill library** | `OPENAI_API_KEY` |
| `jev` / `jev-omni` | decision layer over the skill library | **plugin, currently off** |

### The Jev backends are a plugin now

They live in `jev_policy/plugins/jev/` and are not registered unless asked for,
so a default run never imports the TypeSafe SDK, torch, or anything needing a
GPU (there is a test asserting exactly that, in a subprocess). Switch them on
with an environment variable:

```bash
JEV_POLICY_PLUGINS=jev python scripts/eval_libero.py --planner.backend jev-omni
```

or permanently, via `ENABLED_BY_DEFAULT` in `jev_policy/plugins/__init__.py`.
Asking for a disabled backend tells you how to enable it rather than just
failing. See `jev_policy/plugins/jev/README.md`; the weights are still on the
NAS, so nothing needs re-downloading.

## Direct control with GPT-6-Astra

`astra` is a `Controller`, not a `Planner`: `skills/` is not involved at all.
The model is handed both cameras plus the robot's own state and returns robot
commands, which go to the simulator after validation and clipping.

It cannot run at 20Hz — LIBERO steps 20 times a second and Astra answers in
seconds — so each call returns an **action chunk** executed open loop before the
next observation, the same trick the action-chunking VLAs use. Two modes:

- **`delta`** (default) — raw OSC_POSE deltas, exactly what `env.step` consumes.
  The literal reading of "control directly": nothing interprets the output.
- **`waypoint`** — a target end-effector position and gripper state, which the
  skill layer's P-controller servos to. Still the model choosing the motion, but
  in a space it can reason about.

**This costs real money.** At $10/M in and $50/M out with an image every call,
one episode is roughly $0.20-$0.40 and a 50-episode suite is $10-$20 and several
hours of wall clock. `stats` reports tokens and running spend after every run;
`--planner.reasoning-effort` is the main cost dial.

## Honest caveats

- **Privileged state.** For `jev` (text-only) the model is handed object poses
  read straight from MuJoCo. `jev-omni` removes that: it sees the cameras and its
  prompt carries no coordinates. But **the skill layer still resolves a chosen
  target name to a 3D pose from the simulator** in order to servo to it. So
  `jev-omni` makes the *decision* honest while the *motion* stays privileged.
  Neither backend's numbers are comparable to OpenVLA / π0 / Octo on LIBERO,
  which are RGB-only end to end. Closing the remaining gap needs a detector that
  turns pixels into 3D target poses for `skills/`.
- **The ceiling is the skill library, not the model.** Success depends mostly on
  `skills/primitives.py`. Validate those with `--planner.backend heuristic`
  before spending a single API call.
- **The skill layer is tuned for the LIBERO Panda specifically.** The constants
  in `skills/primitives.py` were measured against the real simulator (see
  *Calibration* below), not guessed, but they are gripper- and object-specific.

## Setup

LIBERO pins old, fragile dependencies, so give it its own env. This is the exact
sequence that works — the version pins are not optional:

```bash
conda create -n jev-libero python=3.10 -y && conda activate jev-libero

pip install "numpy<2" "robosuite==1.4.1" "bddl==1.0.1" "gym==0.25.2" \
            "mujoco==2.3.7" future easydict matplotlib opencv-python \
            imageio imageio-ffmpeg tyro pytest typesafe-sdk
pip install torch --index-url https://download.pytorch.org/whl/cpu

export TYPESAFE_API_KEY=sk-...        # https://typesafe.ai
export MUJOCO_GL=egl                  # or glx, if you hit EGL errors
```

Three traps worth knowing about, all of which cost time here:

- **`mujoco==2.3.7` is required.** robosuite 1.4.1 asserts on mujoco 3.x inside
  `get_joint_qpos_addr`, which surfaces as a bare `AssertionError` during env
  construction.
- **LIBERO needs `torch` only to unpickle its init states**, so the CPU wheel is
  enough. Torch ≥ 2.6 then rejects those files under `weights_only=True`;
  `envs/libero_env.py` allowlists the numpy globals they contain rather than
  disabling the check.
- **LIBERO resolves its assets through `~/.libero/config.yaml`**, which is global
  and may already point at a different checkout than the one you install. Make
  the package on `sys.path` and that config agree, or you will silently mix two
  copies.

For the LIBERO package itself, `pip install -e` does not work on the checkouts
here: `libero/__init__.py` is missing, so `find_packages()` returns nothing and
the install is a no-op. Put it on the path instead (this is what openpi's own
README does):

```bash
echo /path/to/LIBERO > $(python -c "import site;print(site.getsitepackages()[0])")/libero_src.pth
```

## Workflow

Every command below assumes this preamble. `MUJOCO_GL` is required for offscreen
rendering; use `glx` instead of `egl` if you hit EGL errors.

```bash
cd ~/jev-policy
conda activate jev-libero
export MUJOCO_GL=egl
```

### 0. Confirm the install

```bash
python -m pytest tests/ -q          # 33 tests, no simulator or API key needed
```

If these fail, nothing downstream is trustworthy. They cover the scene encoding,
every skill's success/failure/timeout path, the planner's prompt construction
and caching, and the runner's control flow.

### 1. Smoke-test the simulator

```bash
python scripts/eval_libero.py --planner.backend heuristic \
    --task-id 0 --num-trials-per-task 1
```

About 10 seconds. Expect `SUCCESS (103 steps, 6 decisions)`. A failure here is an
environment problem, not a policy problem — check the three traps in *Setup*.

### 2. Establish the model-free baseline

```bash
python scripts/eval_libero.py --planner.backend heuristic \
    --task-suite-name libero_spatial --num-trials-per-task 5
```

Roughly 10 minutes for 50 episodes; expect **78%**. This is the number any
planner has to beat, and it costs nothing to re-run. Do this after *any* change
to `skills/`, because the skill layer is what actually moves the robot.

### 3. Run GPT-6-Astra (direct control)

```bash
export OPENAI_API_KEY=sk-...

# literal direct control: the model emits raw OSC_POSE deltas
python scripts/eval_libero.py --planner.backend astra \
    --planner.control-mode delta --task-id 0 --num-trials-per-task 2

# waypoints servoed by code -- cheaper, faster, easier for the model
python scripts/eval_libero.py --planner.backend astra \
    --planner.control-mode waypoint --task-id 0 --num-trials-per-task 2

# ablation: hand it exact object coordinates, isolating control from perception
python scripts/eval_libero.py --planner.backend astra \
    --planner.control-mode waypoint --planner.include-object-poses \
    --task-id 0 --num-trials-per-task 2
```

**Start with `--task-id 0 --num-trials-per-task 2`.** One episode is roughly
$0.20-$0.40 and a couple of minutes; a full 50-episode suite is $10-$20 and
several hours. The run prints tokens and estimated spend when it finishes.

Cost and latency dials, in order of effect: `--planner.reasoning-effort`
(reasoning bills at the output rate), `--planner.chunk-steps` (more steps per
call means fewer calls), and `--planner.control-mode waypoint` (far fewer output
tokens than emitting 20x7 numbers).

### Optional: serve Jev-Omni (plugin, off by default)

Jev-Omni needs its own environment — torch >= 2.10 and transformers 5.17 cannot
coexist with LIBERO's numpy < 2 / mujoco 2.3.7:

```bash
conda create -n jev-omni python=3.11 -y && conda activate jev-omni
pip install -r https://huggingface.co/akhilaaa3/Jev-Omni/resolve/main/requirements.txt

# The card's requirements.txt is incomplete and the default torch wheel is
# built for CUDA 13. Both of these are required as well:
pip install torch==2.14.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install librosa scipy av
```

Fetch the weights (here onto the NAS; ~72GB):

```bash
HF_HOME=~/synology/hf-cache python -m jev_policy.plugins.jev.fetch \
    --local-dir-root ~/synology/jev-omni --verify
```

Then serve, pointing at the downloaded paths:

```bash
python -m jev_policy.plugins.jev.serve --shard \
    --model-path ~/synology/hf-cache/hub/models--akhilaaa3--Jev-Omni/snapshots/<hash> \
    --base-path  ~/synology/jev-omni/gemma-4-12B-it
```

Check it is up: `curl localhost:8000/health`

**Five things that bite here**, all of them cost time the first time round:

1. **CUDA version.** The default `torch` wheel is `+cu130` and needs a CUDA 13
   driver. On a 535.x driver (CUDA 12.2) `torch.cuda.is_available()` is False
   and Jev-Omni refuses to load. Install the `+cu126` build.
2. **Missing processor dependencies.** `AutoProcessor` fails with
   `Could not import module 'Gemma4UnifiedProcessor'`, which really means
   `torchvision` (and `librosa`/`scipy`/`av`) are absent — they are not in the
   card's requirements.
3. **`--shard` load order.** Upstream's `load_jev_omni()` loads the decision
   backbone in float32 (~48GB for a 12B model) onto one device before casting it
   down, then loads the multimodal base on top: ~80GB peak. `--shard` instead
   loads the base first, **drops the text tower it is about to replace**, and
   only then loads the fine-tuned decoder in bf16 into the space that frees.
   Peak is one-and-a-bit towers rather than two, and it settles at ~26GB —
   comfortable on 2x24GB. Loading the decoder first instead fills both cards
   before the base is a tenth done.
4. **Per-layer buffers.** `runtime_buffers.pt` must be placed on the device of
   the *module that uses it*. Sending them all to the first parameter's device
   (fine on one GPU) blows up mid-stack with
   `Expected all tensors to be on the same device` at
   `hidden_states *= self.layer_scalar`.
5. **Downloading to a NAS.** CIFS/SMB has no symlink support, so the hub's
   snapshot→blob layout fails with `OSError: [Errno 5]`; use `--local-dir-root`.
   Small files have also been observed landing as **0 bytes** (`config.json`
   silently empty), so `--verify` checks every file against the Hub's sizes and
   re-fetches mismatches. Run it.

Startup reads ~72GB over the network, which at ~100MB/s is about 10 minutes.
Copying the weights to a local disk makes restarts much faster.

### Optional: check the hosted Jev connection

```bash
export TYPESAFE_API_KEY=sk-...      # https://typesafe.ai
python -m jev_policy.plugins.jev.check
```

Issues five real decisions against a hand-built scene and prints the chosen
skill, target, confidence, P(done) and per-call latency. No simulator involved,
so it isolates API problems from robot problems. Decisions should be stable
across the five identical calls.

### 4. Run a Jev backend (only with the plugin enabled)

```bash
JEV_POLICY_PLUGINS=jev python scripts/eval_libero.py --planner.backend jev-omni \
    --task-suite-name libero_spatial --num-trials-per-task 5
```

Budget roughly 5-15 decisions per episode — 5-15 calls for `jev`, double that
for `jev-omni`, which asks one question at a time. Compare against step 2 on the same
suite and trial count — a single number in isolation says nothing.

### 5. Read the results

```bash
python -c "
import json; d = json.load(open('results/libero_spatial/jev/results.json'))
print(d['success_rate'], d['planner'])
for k, v in d['per_task'].items(): print(k, v['success_rate'], v['task'][:50])
"
```

`results/<suite>/<backend>/results.json` holds per-task success rates, decision
counts, mean confidence, latency percentiles, token usage and estimated cost,
plus a per-episode `skills_executed` trace. That trace is the first thing to read
when a task fails — it shows exactly which skill looped or timed out. Videos of
failed episodes land in `results/<suite>/<backend>/videos/`.

### Tuning loop

```bash
python scripts/calibrate_grasp.py --task-id 0     # sweep grasp heights in-sim
python scripts/eval_libero.py --planner.backend heuristic --task-id 0 \
    --num-trials-per-task 5 --save-video all
```

Work on one task at a time with videos on, then re-run step 2 to confirm you
have not regressed the suite.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--planner.backend` | `jev` | `jev`, `heuristic`, `random` |
| `--planner.model` | `jev-1.13.0` | Pin it; thresholds shift between versions |
| `--planner.min-confidence` | `0.45` | Below this, retreat and re-plan |
| `--planner.cache-decisions` | `True` | Skip calls when the scene is unchanged |
| `--planner.request-timeout-s` | `10.0` | Per-call timeout |
| `--planner.host` / `--planner.port` | `127.0.0.1:8000` | `jev-omni` server |
| `--planner.ask-done` | `False` | `jev-omni`: adds a third call per decision |
| `--planner.max-options` | `20` | `jev-omni`: target list is ranked and truncated |
| `--task-suite-name` | `libero_spatial` | `_spatial`, `_object`, `_goal`, `_10`, `_90` |
| `--task-id` | all | Single task, for debugging |
| `--num-trials-per-task` | `50` | Standard protocol is 50; drop it while iterating |
| `--save-video` | `failure` | `all`, `failure`, `none` |
| `--seed` | `7` | Affects object placement |

Long runs outlive a terminal; `--num-trials-per-task 5` over a full suite takes
10-15 minutes, so redirect and background it:

```bash
python scripts/eval_libero.py --planner.backend jev \
    --task-suite-name libero_goal --num-trials-per-task 5 \
    > results/jev_goal.log 2>&1 &
tail -f results/jev_goal.log
```

## Baseline

`libero_spatial`, 5 trials per task, scripted `heuristic` planner (no model, no
API calls) — this is the skill layer's own ceiling, and the number Jev has to
beat to be earning its place:

| task | rate | task |
|---|---|---|
| 0 | 60% | black bowl between the plate and the ramekin |
| 1 | 80% | black bowl next to the ramekin |
| 2 | 100% | black bowl from table center |
| 3 | 100% | black bowl on the cookie box |
| 4 | **0%** | black bowl **in the top drawer** of the wooden cabinet |
| 5 | 100% | black bowl on the ramekin |
| 6 | 80% | black bowl next to the cookie box |
| 7 | 100% | black bowl on the stove |
| 8 | 100% | black bowl next to the plate |
| 9 | 60% | black bowl on the wooden cabinet |
| | **78%** (39/50) | |

All 11 failures are `step budget exhausted` — a missed grasp that retries until
the clock runs out, never a crash.

Task 4 is the informative one. A fixed pick-and-place sequence cannot solve it:
the bowl is inside a closed drawer, so it needs `pull_open` first. That is
precisely the decision a planner is for, and it is where Jev should separate
from this baseline — as should `libero_goal` and `libero_10`, where the required
skill order varies per task. On the eight flat tabletop tasks the sequence is
already near ceiling, so expect little headroom there.

### GPT-6-Astra, direct control — 0% (0/6)

Task 0 only, 2 trials per configuration — enough to characterise the failure
before spending on a suite.

| mode | poses given | success | mean latency | $/2 episodes |
|---|---|---|---|---|
| `waypoint` | no (vision only) | 0/2 | 10.0s | $0.42 |
| `delta` | no (vision only) | 0/2 | 13.5s | $0.81 |
| `waypoint` | **yes** (privileged) | 0/2 | 3.4s | $0.34 |

**Vision-only, it cannot ground what it sees into metric coordinates.** All 11
calls of an episode go into approach, and the reasoning says why:

> *"The bowl between the plate and ramekin remains left and nearer the front
> than the gripper. Move to a safe hover over that..."*

It knows the right qualitative move every time and never converts it into a
correct world position. Estimating an object's (x, y, z) from a third-person RGB
frame with no depth and no extrinsics is not something the model can do, and no
amount of prompting fixes it.

**Handed exact coordinates, it still fails — but differently.** Latency drops
10.0s → 3.4s and output tokens drop 4x, because there is nothing to deliberate
about; the reasoning becomes clean and correct ("hover above ... lower with open
fingers to straddle its sides"). It simply never commits: eleven calls of
"align", "maintain clearance", "before descending" exhaust the 230-step budget
during approach. Part of that is my own system prompt, which tells it to hover
before descending and evidently over-weights caution.

So the honest reading is that **grounding is the first wall, not the only one**,
and both are upstream of any comparison with the 78% scripted baseline. The
cheap next experiments are a blunter prompt, larger `--planner.chunk-steps`, and
higher `reasoning_effort` on the privileged-pose variant — that last one
isolates control quality from perception entirely.

### libero_spatial, Jev-Omni — 36% (18/50)

Same suite and trial count as the heuristic above, so the numbers are directly
comparable. 1051 model calls, 357ms mean latency, 0.69 mean confidence.

| task | heuristic | jev-omni |
|---|---|---|
| 0 | 60% | 60% |
| 1 | 80% | **0%** |
| 2 | 100% | **0%** |
| 3 | 100% | 60% |
| 4 | 0% | 0% |
| 5 | 100% | 60% |
| 6 | 80% | 60% |
| 7 | 100% | **0%** |
| 8 | 100% | 80% |
| 9 | 60% | 40% |
| **total** | **78%** | **36%** |

**Jev-Omni loses to the scripted sequence here, and that is roughly what this
suite should show.** Every `libero_spatial` task is the same pick-and-place in a
different arrangement, so a fixed sequence is near-optimal and a planner can
only deviate from it. The interesting part is *how* it fails:

- **The plan structure is usually right.** Task 7's trace is
  `move_above → descend → close_gripper → lift → place_on → open_gripper` —
  exactly the correct sequence. It fails on *targets*, not ordering.
- **Target selection is the weak point**, in two specific ways:
  1. *Placement regions pollute the vocabulary.* LIBERO registers ~16 site
     regions (`main_table_plate_region`, ...) alongside 7 real objects. On task 7
     the model placed onto `main_table_plate_region` instead of `plate_1`.
     Filtering sites out of the target list, or ranking real objects above them,
     is the obvious next fix.
  2. *Identical objects are ambiguous.* `akita_black_bowl_1` vs
     `akita_black_bowl_2` cannot be told apart from the names, and "the bowl on
     the stove" has to be resolved visually. Task 7 picked the wrong bowl.
- **Low confidence can livelock.** 98 of ~500 decisions fell below
  `min_confidence`, each replaced by `retreat`. On task 2 that produced
  `move_above → retreat → move_above → retreat`, which never progresses. The
  recovery behaviour needs to escalate rather than repeat.

### libero_goal — 12% (6/50)

The same scripted planner on `libero_goal`, where the required skill *order*
varies per task:

| task | rate | blocker |
|---|---|---|
| 0 open the middle drawer | 0% | needs `pull_open`; also a front approach, not a top-down one |
| 1 put the bowl on the stove | 60% | — |
| 2 wine bottle on top of cabinet | 0% | hover point lands *inside* the bottle (see below) |
| 3 open top drawer, put bowl inside | 0% | needs `pull_open` first |
| 4 bowl on top of cabinet | 0% | place target is a high region |
| 5 push the plate to the stove | 0% | **no push-object skill exists** |
| 6 cream cheese in the bowl | 0% | reaches correctly, grasp does not hold |
| 7 turn on the stove | 0% | **no turn-on skill exists** |
| 8 put the bowl on the plate | 60% | — |
| 9 wine bottle on the rack | 0% | same as task 2 |

**This is not yet a fair test of the decision layer.** The 78% → 12% drop
conflates three different causes, and only the second is Jev's to fix:

1. **Missing skills.** Tasks 5 and 7 need primitives that do not exist
   (`push_object`, `turn_on`). No planner can solve them with this library.
2. **Wrong skill order.** Tasks 0 and 3 need `pull_open` before the pick. This
   is the genuine planning failure, and the case Jev exists for.
3. **Primitive geometry bugs.** Measured, not guessed:
   - Tasks 2/9: `APPROACH_CLEARANCE` and `GRASP_Z_OFFSET` are measured from the
     object's *body origin*. For a tall object like a wine bottle, "10cm above
     the origin" is a point inside the bottle, so `MoveAbove` pushes into it and
     times out 3cm short. These need the object's z extent, the way
     `grasp_point()` already uses its xy radius.
   - Task 0: stalls at 0.017m against a `REACH_TOL` of 0.015m — close enough
     that a lightly obstructed approach never terminates.
   - Task 6: reaches the grasp point to within 0.2mm, so the failure is the
     grasp itself, not the approach.

Fixing (1) and (3) is primitive work that has to happen whatever drives the
planner. Until it does, `libero_goal` measures the skill library, not the model.

## Layout

```
jev_policy/
  config.py                 CLI config; suite step budgets; OSC limits
  perception/scene_state.py MuJoCo → symbolic text (discretised, relative)
  policies/
    base.py                 Planner interface, Decision
    scripted.py             heuristic + random baselines
    registry.py             backend lookup (core + enabled plugins)
  control/
    base.py                 Controller interface: model emits robot commands
    astra.py                GPT-6-Astra direct control  <- the model lives here
  plugins/
    __init__.py             plugin on/off switch
    jev/                    Jev + Jev-Omni, currently OFF
    scripted.py             heuristic + random baselines
    registry.py             backend lookup
  skills/
    base.py                 Skill interface, statuses, gripper constants
    controller.py           Cartesian P-control → OSC_POSE deltas
    primitives.py           the skill vocabulary Jev chooses from
  envs/libero_env.py        suite + env construction
  runner.py                 the two-rate rollout loop
  metrics.py                success rates, JSON results
scripts/
  eval_libero.py            entrypoint
  calibrate_grasp.py        sweeps grasp heights against the real sim
tests/                      30 tests, no simulator required
```

## Tests

```bash
python -m pytest tests/ -q
```

These cover the scene encoding, every skill's success/failure/timeout paths, the
planner's prompt construction and caching (against a fake client), and the
runner's control flow (against a stub simulator). They need neither MuJoCo nor
an API key.

## Calibration

`scripts/calibrate_grasp.py` sweeps grasp heights against the real simulator and
reports which ones survive a lift. On task 0 it gives:

```
  offset   grasped   lifted  obj z rise
    0.00     False     True       0.214
    0.02     False     True       0.210
    0.04      True     True       0.206
    0.06     False    False       0.000     <- fingers land on the rim
```

`GRASP_Z_OFFSET = 0.015` sits in the middle of the working band. Note that
`grasped` reads False on two rows that lifted the bowl 21cm — that is the
load-registration lag described below, and it is why `lifted` is the column
that matters. The constants in `skills/primitives.py` came
from measurements like it, and three of them were counter-intuitive enough to be
worth recording:

- **`GRASP_AXIS` is world +y, `GRIPPER_SPAN` is 0.084m.** Measured from the
  finger pad geoms, not assumed.
- **Wide objects must be grasped at the rim.** An Akita bowl's outer diameter
  (~0.094m) exceeds the Panda's maximum opening, so a grasp aimed at its centre
  rests both fingers on the rim and stalls the descent ~4cm short — which looks
  exactly like a controller tuning problem and is not one. `grasp_point()`
  offsets by half a span so one finger drops inside the bowl.
- **A grasp does not register until it bears load.** robosuite's `_check_grasp`
  reads False for the whole moment after the fingers close and only flips true
  once the object is lifted. Any skill that asserts a grasp immediately after
  closing will abort every correct grasp; `Lift` verifies instead, by requiring
  the *object's* z to rise.
- **`PlaceOn` subtracts the grasp offset.** A rim-held bowl sits ~4cm off the
  gripper's axis; driving the eef to the target leaves the bowl beside it.

## Tuning notes

1. `skills/controller.py::DEFAULT_KP` — above ~2.0 the Panda oscillates, below
   ~0.5 skills time out. 1.2 converges to <1mm in ~30 control steps.
2. `--planner.min-confidence` — sweep it on a few tasks; the whole argument for
   a calibrated model is that this threshold does real work.
3. Skill descriptions in `SKILL_LIBRARY` are prompt surface, sent verbatim as
   `Choice` criteria. Edit them with that in mind.
4. The scene text lists all 23 objects and regions LIBERO registers, including
   placement regions. That is deliberate — regions are legitimate `place_on`
   targets — but it is the main lever on prompt size if calls get expensive.
