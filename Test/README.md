# Final policy test

This folder is an isolated final-test interface. It never trains or updates the
policy. Checkpoints must be selected using validation results before test data
is used.

## Protocol

- One held-out WZ/layout and three authored origins per scenario.
- Default: 50 completed episodes per origin = 150 episodes per scenario.
- Deterministic `model.predict` by default.
- Infrastructure failures are retried and reported separately.
- CSV stores every episode; JSON stores overall/per-origin success and failure
  reasons plus model, reward and test-input SHA256 hashes.
- No-rendering is the default; pass `--rendering` only for visual inspection.

The S1/S5 display/runtime mapping is intentionally identical to validation:
experiment S1 loads runtime S5/Town05, while experiment S5 loads runtime
S1/Town02.

The six included manifests currently have `status: ready`. Recheck the live
CARLA/SUMO geometry whenever a test case changes. `--allow-blocked` is reserved
for that check and should not be used for formal results.

Offline format check (does not connect to CARLA or run a policy):

```bat
cd /d X:\path\WZ-Plan_Code
python -B -m Test.Scenarios.test.check_templates
```

## Command (activated `ppo_carla` CMD)

```bat
cd /d X:\path\WZ-Plan_Code
python -m Test.run_policy_test --scenario s2 --model "D:\path\policy.zip" --policy-name "encoder1_ppo_oldod_seed7_p28" --episodes-per-origin 50
```

The evaluation environment is deliberately fixed to the Old-OD reward and
centre-ego termination contract. Only `--model` changes between policy tests.
