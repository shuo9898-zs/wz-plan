# WZ-Plan code

This directory contains the WZ-Plan research code and scenario definitions used
for roadway work-zone motion-planning experiments. It is a cleaned source bundle
derived from the local `Aug24_ppo` project. Training outputs, recorded videos,
checkpoints, caches, and the simulator executable are not included.

## Requirements

- Windows with Python 3.8.20. Python package versions are listed in
  `requirements-lock.txt`.
- The official CARLA 0.9.15 **prebuilt** WindowsNoEditor simulator, including
  `CarlaUE4.exe`, installed separately. This is the packaged executable release;
  building CARLA from source is not required. Install the matching CARLA 0.9.15
  Python API in the Python environment. The simulator and Python API are not
  distributed in this code bundle.
- SUMO 1.27.1, available through `eclipse-sumo==1.27.1` or an equivalent local
  installation.
- An NVIDIA GPU and a compatible PyTorch wheel are needed for GPU training.
  `ffmpeg` is needed only for video recording. Install Pillow when using the
  optional still-image and contact-sheet utilities in `Test/`.

If you use the optional validation-debug launcher to start CARLA, set
`CARLA_EXE` to the path of the external `CarlaUE4.exe`. Otherwise, keep the
executable on `PATH` or start CARLA manually.

## Project map

| Path | Purpose |
| --- | --- |
| `run_train.py`, `train.py`, `baseline/PPO/` | PPO training entry point and implementation |
| `models/`, `env/`, `logic/`, `sync/` | Scene encoder, CARLA-SUMO environment, rewards, termination, and synchronization |
| `config/`, `scenarios/` | Training catalog, CARLA settings, and SUMO networks/routes |
| `validation/`, `validation_debug/` | Validation cases and runner |
| `Test/` | Held-out test protocol, demo recorders, and paper-figure utilities |
| `preflight.py` | Offline dependency, import-isolation, and scenario-file check |

The original root-level import layout is preserved because the runtime imports
modules such as `config.scenario_catalog` and `baseline.PPO.runtime_v2` directly.

## Setup and checks

Create a Python 3.8 environment, install the versions in
`requirements-lock.txt`, and install the matching CARLA Python API from the
CARLA 0.9.15 package. Run commands from this directory:

```powershell
python -B preflight.py
```

`preflight.py` does not start CARLA or run an episode. It checks external
imports, verifies that project modules resolve inside this bundle, and checks
training and validation scenario paths. Continue only when it prints
`"ok": true`.

## Running experiments

Start separate CARLA 0.9.15 `CarlaUE4.exe` instances on RPC ports 2000, 2020,
and 2040. The training launcher assigns Town02, Town05, and Town10HD to those
servers. Then run:

```powershell
python run_train.py
```

Training settings and checkpoint behavior are defined in
`experiment_config.py` and `baseline/PPO/`. Training outputs are written to
`runs/`, which is intentionally excluded from this source bundle.

For the center-ego Old-OD variant with an explicit random seed, use
`python run_train_center_seeded.py --seed 27` after configuring the CARLA
servers. Its checkpoints are also generated locally and are not included here.

For the held-out test interface, provide a checkpoint and review the test-case
definitions in `Test/Scenarios/test/README.md`. The six included manifests are
currently marked `ready`; verify their geometry again if you change a case.
An example test command is:

```powershell
python -m Test.run_policy_test --scenario s2 --model C:\path\to\policy.zip --policy-name my_policy --episodes-per-origin 50
```

The scenario display labels S1 and S5 intentionally map to different internal
runtime IDs. Keep the bindings in `config/scenario_labels.py` and the
validation/test manifests when reproducing results.

## Scope of this bundle

This code bundle excludes the CARLA 0.9.15 simulator executable, trained model
files, raw rollout data, result tables, screenshots, and videos. These are
separate artifacts. The Python source, JSON scenario settings, and SUMO XML
assets needed by the project's own preflight checks are included.
