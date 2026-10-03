# Held-out test scenarios

This directory contains the final test split. The training catalog and
validation loader do not load these cases automatically.

Each of the six cases has one work-zone configuration (`wz1`, layout A), three
ego origins, and one destination line. S1, S2, S3, S5, and S6 include SUMO
network and route XML files. S4 uses CARLA only and includes a jaywalker
controller. The included manifests are marked `ready` following the recorded
September 2026 geometry checks; rerun live checks after changing any case.

| Scenario | Geometry and traffic contract |
| --- | --- |
| S1 | Forbidden work-zone region and multiple SUMO traffic routes |
| S2 | Curved `forbidden_polygon`, `polygon_config`, and lead/follow traffic routes |
| S3 | Left and right `safe_corridor` boundaries, open ends, and route-specific traffic |
| S4 | CARLA-only jaywalker path and three-stage trigger parameters |
| S5 | Forbidden work-zone region with lead, follow, and oncoming SUMO traffic |
| S6 | Oncoming platoon size, headway, gap, and speed distribution |

When updating a test case, change the independently authored work-zone,
props/cones, three origins, and destination in its `configs/wz1.json`. If a
SUMO route changes, update both the JSON `bg_routes` and the corresponding XML.
Mark the manifest `blocked` until new coordinates pass a live CARLA/SUMO check.

Run the offline format check from the bundle root:

```powershell
python -B -m Test.Scenarios.test.check_templates
```

The stable runtime IDs are `s1` through `s6`. The displayed S1/S5 labels
follow the same mapping as validation.
