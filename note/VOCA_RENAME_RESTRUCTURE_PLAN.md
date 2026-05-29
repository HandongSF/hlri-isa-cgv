# VOCA Rename and Repository Restructure Plan

## Context

The repository was originally named `VOCA`, then renamed to `hlri-isa-cgv`.
The target system name is now **VOCA**.

There is also an existing reference repository at:

```text
/home/gunminy/VOCA
```

For the first cleanup pass, use the naming and file structure style from
`/home/gunminy/VOCA` while preserving the newer depth PointNav work currently in
`/home/gunminy/hlri-isa-cgv`.

## Goals

- Rename old project/system references to `VOCA`.
- Align file names with the existing `/home/gunminy/VOCA` convention.
- Keep the depth PointNav additions intact.
- Make import paths explicit and easier to maintain.
- Separate project code from vendored or adapted third-party policy code.
- Avoid large behavioral changes during the first rename pass.

## Current Relevant Structure

```text
hlri-isa-cgv/
  README.md
  constants.py
  habitat_config.py
  vlm_planner.py
  objnav_benchmark.py
  metrics_summary.py
  depth_pointnav_controller.py
  cv_utils/
    yoloe_detector.py
  data_utils/
    geometry.py
    depth_pointnav_geometry.py
    reachable_waypoint.py
  llm_utils/
    gpt_request.py
    gpt_request_gemini.py
    gpt_request_ollama.py
    navigation_prompts.py
    priors_parser.py
  third_party/
    vlfm_pointnav/
  vlfm/
    obs_transformers/
      resize.py
  note/
```

## Reference Naming From `/home/gunminy/VOCA`

```text
VOCA/
  README.md
  REPO_CLEANUP.md
  settings.py
  habitat_config.py
  vlm_planner.py
  objnav_benchmark.py
  metrics_summary.py
  cv_utils/
    yoloe_detector.py
  data_utils/
    geometry.py
  llm_utils/
    gemini_request.py
    ollama_request.py
    navigation_prompts.py
    priors_parser.py
```

## Rename Map

Apply these renames first, then fix imports.

| Current file | Target file | Reason |
| --- | --- | --- |
| `constants.py` | `settings.py` | Central project settings and environment-variable based paths. |
| `habitat_config.py` | `habitat_config.py` | Habitat-specific configuration helpers. |
| `vlm_planner.py` | `vlm_planner.py` | Planner is not necessarily GPT-4V specific; VOCA uses VLM terminology. |
| `cv_utils/yoloe_detector.py` | `cv_utils/yoloe_detector.py` | More concrete detector module name, matches `/home/gunminy/VOCA`. |
| `data_utils/geometry.py` | `data_utils/geometry.py` | Shorter reusable geometry module name. |
| `llm_utils/gpt_request_gemini.py` | `llm_utils/gemini_request.py` | Backend-specific request module. |
| `llm_utils/gpt_request_ollama.py` | `llm_utils/ollama_request.py` | Backend-specific request module. |
| `llm_utils/navigation_prompts.py` | `llm_utils/navigation_prompts.py` | Clearer prompt module name. |
| `depth_pointnav_controller.py` | `pointnav_controller.py` or `controllers/depth_pointnav.py` | Keep as `depth_pointnav_controller.py` for first pass if minimizing import churn. |

Keep these names for now:

| File or directory | Decision |
| --- | --- |
| `objnav_benchmark.py` | Keep. This is the main ObjectNav evaluation entry point. |
| `evaluate_policy.py` | Removed. PixelNav evaluation is no longer part of the VOCA ObjectNav code path. |
| `metrics_summary.py` | Keep. |
| `policy_agent.py` | Removed. Legacy PixelNav executor is no longer used by ObjectNav. |
| `policy_network.py` | Removed. Legacy PixelNav network is no longer used by ObjectNav. |
| `data_utils/depth_pointnav_geometry.py` | Keep. This is new functionality not present in `/home/gunminy/VOCA`. |
| `data_utils/reachable_waypoint.py` | Keep. This is new functionality not present in `/home/gunminy/VOCA`. |

## Import Updates Required

After file renames, update imports consistently:

```python
from constants import ...
```

to:

```python
from settings import ...
```

```python
from habitat_config import hm3d_config
```

to:

```python
from habitat_config import hm3d_config
```

```python
from vlm_planner import GPT4V_Planner
```

to:

```python
from vlm_planner import VLMPlanner
```

For minimal churn, a compatibility alias can be kept temporarily:

```python
class VLMPlanner:
    ...

GPT4V_Planner = VLMPlanner
```

Then remove the alias after all call sites are migrated.

## Class and Symbol Rename Map

Do this after file renames are stable.

| Current symbol | Target symbol | Notes |
| --- | --- | --- |
| `GPT4V_Planner` | `VLMPlanner` | Main VOCA planner class. |
| `POINTNAV_CHECKPOINT` | `POINTNAV_CHECKPOINT_PATH` | More explicit path name. |

## Third-Party and VLFM Policy Code

The current PointNav policy code should not be mixed with VOCA project modules
without a clear boundary.

Current:

```text
third_party/vlfm_pointnav/
vlfm/obs_transformers/
```

Recommended first-pass decision:

- Keep `third_party/vlfm_pointnav/` in place for now.
- Keep `vlfm/obs_transformers/resize.py` in place for now if Habitat config loading depends on this module path.
- Add a VOCA-facing wrapper module later, for example:

```text
pointnav_controller.py
third_party/vlfm_pointnav/
```

or, in a later package-style refactor:

```text
voca/
  controllers/
    depth_pointnav.py
  policies/
    pointnav.py
  _vendor/
    vlfm_pointnav/
```

The important rule is: VOCA application code should import a VOCA wrapper, not
reach directly into vendored code from many places.

## Suggested First-Pass Structure

This matches `/home/gunminy/VOCA` while preserving the new PointNav files:

```text
hlri-isa-cgv/
  README.md
  REPO_CLEANUP.md
  settings.py
  habitat_config.py
  vlm_planner.py
  objnav_benchmark.py
  metrics_summary.py
  depth_pointnav_controller.py
  cv_utils/
    yoloe_detector.py
  data_utils/
    geometry.py
    depth_pointnav_geometry.py
    reachable_waypoint.py
  llm_utils/
    gemini_request.py
    ollama_request.py
    navigation_prompts.py
    priors_parser.py
  third_party/
    vlfm_pointnav/
  vlfm/
    obs_transformers/
      resize.py
  note/
    VOCA_RENAME_RESTRUCTURE_PLAN.md
```

## Later Package-Style Structure

After the simple rename pass works, move toward an importable package:

```text
VOCA/
  pyproject.toml
  README.md
  configs/
  scripts/
  voca/
    __init__.py
    settings.py
    habitat_config.py
    benchmark/
      objnav.py
      metrics.py
    controllers/
      depth_pointnav.py
    planners/
      vlm.py
    perception/
      yoloe.py
    geometry/
      camera.py
      depth_pointnav.py
      reachable_waypoint.py
    policy/
      pixelnav_agent.py
      pixelnav_network.py
      pointnav.py
    llm/
      gemini.py
      ollama.py
      prompts.py
      priors_parser.py
    integrations/
      habitat/
        obs_transformers/
          resize.py
    _vendor/
      vlfm_pointnav/
  checkpoints/
  data/
  outputs/
  note/
```

Do not jump directly to this structure until the first-pass rename is tested.

## Execution Order

1. Add this plan document.
2. Rename files using the first-pass rename map.
3. Update imports with `rg` and small patches.
4. Rename classes with temporary compatibility aliases.
5. Replace old project text:
   - `VOCA`
   - `ISA`
   - `hlri-isa-cgv`
   - `GPT4V` where it means the planner abstraction
6. Run a syntax/import smoke test.
7. Run a minimal benchmark command with a small episode count.
8. Only after that, consider the package-style `voca/` refactor.

## Smoke Tests

Use import checks first:

```bash
python -m py_compile settings.py habitat_config.py vlm_planner.py objnav_benchmark.py
python - <<'PY'
from settings import POINTNAV_CHECKPOINT
from habitat_config import hm3d_config
from vlm_planner import VLMPlanner
from depth_pointnav_controller import DepthPointNavController
print("VOCA import smoke test passed")
PY
```

Then run a tiny evaluation:

```bash
python objnav_benchmark.py --eval_episodes 1
```

## Risks

- Habitat config files may refer to module paths such as `vlfm.obs_transformers.resize`.
  Do not move `vlfm/obs_transformers/resize.py` until those config references are checked.
- `third_party/vlfm_pointnav` contains adapted external code. Keep license/copyright
  headers and avoid blending it into VOCA-owned modules.
- Renaming `GPT4V_Planner` immediately can break old scripts. Use a compatibility alias
  for one transition commit.
- Moving to a `voca/` package changes import behavior. Do it only after the simple
  file rename pass is stable.

## Open Decisions

- Should the repository directory itself become `/home/gunminy/VOCA`, replacing the
  current reference repository, or should `hlri-isa-cgv` remain the working copy until
  the cleanup is complete?
- Should `depth_pointnav_controller.py` stay at the root for compatibility, or move
  into a new `controllers/` directory now?
- Should old `gpt_request.py` remain as a provider router, or be renamed to a neutral
  `llm_request.py` module?
