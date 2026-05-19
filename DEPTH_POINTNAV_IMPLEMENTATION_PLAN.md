# Depth PointNav Controller Implementation Spec

## Goal

Replace the current PixNav local execution input path with a VLFM PointNav-compatible
input path while keeping the existing high-level waypoint selection logic.

Current PixNav input:

```text
goal_image + goal_mask + current RGB + collision flag
```

New Depth PointNav input:

```python
{
    "depth": depth_tensor,  # (1, H, W, 1), float32
    "pointgoal_with_gps_compass": rho_theta_tensor,  # (1, 2), [[rho, theta]]
}
```

The `goal_mask` is no longer policy input. It is only used to extract a waypoint pixel.

## VLFM PointNav Contract

Reference files:

```text
/home/gunminy/vlfm-main/vlfm/policy/utils/pointnav_policy.py
/home/gunminy/vlfm-main/vlfm/policy/base_objectnav_policy.py
/home/gunminy/vlfm-main/vlfm/utils/geometry_utils.py
```

Use:

```text
/home/gunminy/vlfm-main/data/pointnav_weights.pth
```

Policy observation keys:

```python
obs_pointnav = {
    "depth": depth_tensor,
    "pointgoal_with_gps_compass": rho_theta_tensor,
}
```

Required tensor shapes:

```python
depth_tensor.shape == (1, H, W, 1)
rho_theta_tensor.shape == (1, 2)
```

Default config:

```yaml
local_controller:
  name: depth_pointnav
  pointnav_policy_path: /home/gunminy/vlfm-main/data/pointnav_weights.pth
  depth_image_shape: [224, 224]
  pointnav_stop_radius: 0.9
  reset_pointnav_on_new_waypoint: true
```

VLFM PointNav uses `Discrete(4)`. Verify the Habitat action ID mapping in FENav before
running evaluation:

```text
STOP
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
```

Do not reuse PixNav-only action handling for actions `4` and `5` in the PointNav path.

## Required New Functions

### 1. Extract waypoint pixel from mask

Current planner returns `goal_image` and `goal_mask`, not a raw pixel.

Add:

```python
def extract_waypoint_pixel_from_mask(goal_mask: np.ndarray) -> tuple[int, int] | None:
    ...
```

Implementation:

- If mask has positive pixels, return centroid or center of the filled rectangle.
- Return `(u, v)` in image coordinates.
- Return `None` if the mask is empty.

### 2. Robust depth lookup

Add:

```python
def lookup_valid_depth(
    depth: np.ndarray,
    pixel: tuple[int, int],
    min_depth: float,
    max_depth: float,
    window: int = 5,
) -> float | None:
    ...
```

Implementation:

- Read depth at `(u, v)`.
- Valid depth must be finite and `min_depth < d < max_depth`.
- If invalid, search a local window and return the median valid depth.
- Return `None` if no valid depth exists.

### 3. Pixel to world waypoint

Add:

```python
def build_depth_waypoint_from_pixel(
    pixel: tuple[int, int],
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> DepthWaypoint:
    ...
```

Use metric depth here.

Important:

- Use the camera/sensor pose, not only the agent body pose.
- The `goal_mask`, `rgb`, `depth`, and camera pose must come from the same timestep.
- If the selected mask comes from a panorama/history frame, store the matching depth and
  camera pose, or recompute the waypoint after rotating to the selected view.

### 4. World waypoint to PointGoal

VLFM convention:

```python
local_goal = rotation_matrix(-heading) @ (goal_xy - robot_xy)
rho = np.linalg.norm(local_goal)
theta = np.arctan2(local_goal[1], local_goal[0])
```

Meaning:

- local x: forward/backward
- local y: left/right
- positive `theta`: goal is to the left / CCW from above

Add:

```python
def compute_relative_pointgoal(
    waypoint_world: np.ndarray,
    current_agent_xy: np.ndarray,
    current_heading: float,
) -> PointGoal:
    ...
```

Test this sign once in Habitat:

- Center pixel should produce `theta ~= 0`.
- Left-side pixel should make the policy turn left.
- Right-side pixel should make the policy turn right.

### 5. Depth preprocessing for PointNav

Geometry needs metric depth. VLFM PointNav expects policy depth shaped and normalized
for inference.

Add:

```python
def preprocess_depth_for_pointnav(
    depth: np.ndarray | torch.Tensor,
    min_depth: float,
    max_depth: float,
    output_size: tuple[int, int],
    device: str,
) -> torch.Tensor:
    ...
```

Implementation:

- Input may be `(H, W)` or `(H, W, 1)`.
- Clip to `[min_depth, max_depth]`.
- Normalize to `[0, 1]`.
- Resize to `output_size`, default `(224, 224)`.
- Return `(1, H, W, 1)` float32 tensor.

Use:

```python
depth_for_geometry = obs["depth"]  # metric
depth_for_policy = preprocess_depth_for_pointnav(obs["depth"], ...)
```

### 6. Depth PointNav controller wrapper

Add:

```python
class DepthPointNavController:
    def __init__(self, cfg):
        ...

    def reset(self):
        ...

    def on_new_waypoint(self):
        ...

    def act(self, depth_obs, rho: float, theta: float):
        ...
```

Behavior:

- Load VLFM PointNav weights.
- Maintain recurrent hidden state and previous action.
- Reset at episode start.
- Reset on new VLM waypoint for the first implementation.
- Call policy with `deterministic=True`.
- Return FENav/Habitat-compatible action ID.

Internal policy call:

```python
depth_tensor = preprocess_depth_for_pointnav(depth_obs, ...)
rho_theta_tensor = torch.tensor([[rho, theta]], device=device, dtype=torch.float32)
obs_pointnav = {
    "depth": depth_tensor,
    "pointgoal_with_gps_compass": rho_theta_tensor,
}
action = wrapped_pointnav.act(obs_pointnav, masks, deterministic=True)
```

## Main Loop Integration

Required flow:

```python
if controller_name == "pixnav":
    action = pixnav_controller.act(obs_rgb, goal_image, goal_mask, collided)

elif controller_name == "depth_pointnav":
    if current_waypoint is None or need_new_waypoint:
        goal_image, goal_mask, ... = nav_planner.make_plan(...)
        waypoint_pixel = extract_waypoint_pixel_from_mask(goal_mask)

        if waypoint_pixel is None:
            need_new_waypoint = True
            continue

        current_waypoint = build_depth_waypoint_from_pixel(
            pixel=waypoint_pixel,
            depth=obs["depth"],
            camera_intrinsics=camera_intrinsics,
            camera_position=current_camera_position,
            camera_rotation=current_camera_rotation,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        if not current_waypoint.valid:
            need_new_waypoint = True
            continue

        pointnav_controller.on_new_waypoint()

    pointgoal = compute_relative_pointgoal(
        waypoint_world=current_waypoint.world_position,
        current_agent_xy=current_agent_xy,
        current_heading=current_heading,
    )

    if pointgoal.rho < pointnav_stop_radius:
        need_new_waypoint = True
        continue

    action = pointnav_controller.act(
        depth_obs=obs["depth"],
        rho=pointgoal.rho,
        theta=pointgoal.theta,
    )
```

STOP handling:

- `rho < pointnav_stop_radius` means the local waypoint is reached.
- Replan or verify target.
- Do not treat local waypoint STOP as ObjectNav episode success unless the target object
  has been confirmed.

## Data Structures

```python
@dataclass
class DepthWaypoint:
    pixel_u: int
    pixel_v: int
    initial_depth: float | None
    world_position: np.ndarray | None
    valid: bool
    failure_reason: str | None = None
```

```python
@dataclass
class PointGoal:
    rho: float
    theta: float
```

## Logging

Log these fields for debugging:

```text
waypoint_pixel
waypoint_depth
waypoint_world_position
pointgoal_rho
pointgoal_theta
pointnav_action
conversion_failed
failure_reason
policy_stop_triggered
replan_triggered
```

## Minimum Sanity Checks

- Center pixel gives `theta ~= 0`.
- Left/right pixels produce opposite theta signs.
- Rotating in place changes recomputed theta consistently.
- During forward motion toward a valid waypoint, `rho` decreases.
- PointNav action IDs match FENav/Habitat action IDs.
- PixNav path still works when `local_controller.name == pixnav`.
