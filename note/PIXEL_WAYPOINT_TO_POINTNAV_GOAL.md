# Pixel Waypoint to PointNav Goal

이 문서는 현재 FENav `depth_pointnav` 구현에서 pixel waypoint가 PointNav policy의
relative goal `(rho, theta)`로 변환되는 실제 알고리즘을 정리한다.

## Summary

현재 구현은 segmentation mask 내부의 point를 쓰지 않는다. YOLOE bounding box의
center 근처 pixel을 waypoint anchor로 사용한다.

```text
YOLOE bbox
-> bbox center 주변 synthetic mask
-> mask 하단 anchor pixel
-> pixel depth lookup
-> camera 3D point
-> world waypoint
-> 0.10m offset toward agent
-> current pose 기준 (rho, theta)
-> PointNav policy
```

## 1. Waypoint Pixel Source

`gpt4v_planner.py`의 `apply_priors_on_image()`는 YOLOE detection을 실행한다.
현재 호출은 box만 사용한다.

```python
yoloe_detection(..., retina_masks=False)
```

따라서 실제 object segmentation mask는 사용하지 않는다. 선택된 detection의
bounding box center를 계산한다.

```python
px = int((x1 + x2) * 0.5)
py = int((y1 + y2) * 0.5)
```

그 다음 `(px, py)` 주변에 반지름 `8px`짜리 작은 사각형 mask를 만든다.

```python
r = 8
cv2.rectangle(
    debug_mask,
    (int(px - r), int(py - r)),
    (int(px + r), int(py + r)),
    (255,),
    -1,
)
```

이 `debug_mask`가 `objnav_benchmark.py`에서는 `goal_mask`로 전달된다.

`extract_anchor_pixel_from_mask(goal_mask)`는 mask의 가장 아래쪽 band에서 평균
pixel을 구한다.

```python
bottom_y = ys.max()
band_height = max(2, int(round(mask.shape[0] * 0.03)))
bottom_band = ys >= bottom_y - band_height
u = round(xs[bottom_band].mean())
v = round(ys[bottom_band].mean())
```

하지만 `goal_mask` 자체가 bbox center 주변의 작은 사각형이므로 최종 waypoint
pixel은 사실상 다음과 같다.

```text
u ~= bbox center x
v ~= bbox center y + small downward bias
```

즉 현재 구현의 pixel waypoint는 segmentation 기반이 아니라 bbox center 기반이다.

## 2. Depth Sensor Values

`depth_pointnav` 모드에서는 `config_utils.py`의 `_enable_depth_pointnav_mode()`가
Habitat depth sensor를 다음 값으로 설정한다.

```python
min_depth = 0.5
max_depth = 5.0
normalize_depth = True
```

따라서 `obs["depth"]`는 meter depth가 아니라 normalized depth이다. geometry
계산에서만 metric depth로 복원한다.

```python
depth_metric = depth_norm * (max_depth - min_depth) + min_depth
depth_metric = depth_norm * 4.5 + 0.5
```

예시:

```text
depth_norm = 0.0 -> 0.5m
depth_norm = 0.5 -> 2.75m
depth_norm = 1.0 -> 5.0m
```

PointNav policy 입력에는 normalized depth를 그대로 넣고, pixel backprojection에만
metric depth를 사용한다.

## 3. Depth Lookup at the Waypoint Pixel

waypoint pixel `(u, v)`의 depth는 `lookup_valid_depth()`에서 결정한다.

먼저 center pixel의 metric depth를 읽는다.

```python
d = depth_metric[v, u]
```

valid 조건은 정확히 다음과 같다.

```python
np.isfinite(d) and min_depth <= d < max_depth
```

현재 실제 값 기준으로는:

```python
np.isfinite(d) and 0.5 <= d < 5.0
```

따라서 아래 값들은 invalid이다.

```text
d == 5.0
d < 0.5
d >= 5.0
NaN
Inf
```

center pixel이 invalid이면 주변 `5x5` window를 본다.

```text
x: u - 2 ... u + 2
y: v - 2 ... v + 2
```

이 window 안에서 valid depth만 모은 뒤 median을 사용한다.

```python
d = np.median(valid_depths)
```

median은 정렬했을 때의 중앙값이다. 평균보다 outlier에 덜 민감해서 object depth와
background depth가 섞인 작은 patch에서 더 안정적이다.

주변 `5x5`에도 valid depth가 없고 pixel이 이미지 안에 있으면 fallback으로
`max_depth`를 사용한다.

```python
d = 5.0
target_kind = "max_depth_fallback"
```

주의할 점: depth sensor가 `min_depth=0.5`이므로 0.5m보다 가까운 표면은 보통
0.5m 근처로 clipping된다. 현재 valid 조건은 `d == 0.5`를 valid로 보기 때문에,
이 경우 waypoint depth는 최소 거리인 `0.5m`로 유지된다. 반면 `d == 5.0`은
여전히 max range clipping 또는 ray miss일 수 있으므로 invalid로 본다.

## 4. Pixel + Depth to Camera 3D Point

카메라 intrinsic은 `habitat_camera_intrinsic()`에서 계산한다.

```python
cx = (width - 1.0) / 2.0
cy = (height - 1.0) / 2.0
f = (width / 2.0) / tan(hfov / 2.0)
fx = f
fy = f
```

pixel `(u, v)`와 depth `d`를 camera 좌표계의 3D point로 변환한다.

```python
x_image = (u - cx) * d / fx
y_image = (v - cy) * d / fy

point_camera = np.array([d, -x_image, -y_image])
```

이미지 중앙 pixel이면 `x_image ~= 0`, `y_image ~= 0`이므로:

```text
point_camera ~= [d, 0, 0]
```

즉 camera 정면 `d` meter 앞의 점이 된다.

## 5. Camera 3D Point to World Point

현재 agent pose는 Habitat observation의 `gps`, `compass`에서 가져온다.

```python
gps = obs["gps"]
heading = obs["compass"][0]

agent_xy = np.array([gps[0], -gps[1]])
camera_position = np.array([agent_xy[0], agent_xy[1], camera_height])
```

`gps[1]`에 minus를 붙이는 것은 VLFM 좌표계 convention에 맞추기 위한 변환이다.

heading으로 yaw rotation matrix를 만든다.

```python
R = np.array([
    [cos(heading), -sin(heading), 0.0],
    [sin(heading),  cos(heading), 0.0],
    [0.0,           0.0,          1.0],
])
```

camera point를 world 좌표로 변환한다.

```python
raw_world = R @ point_camera + camera_position
```

이 값은 bbox center 근처 pixel ray를 depth `d`만큼 따라간 3D world anchor이다.

## 6. Offset Toward the Agent

현재 구현은 raw 3D anchor를 그대로 PointNav target으로 쓰지 않는다.
`resolve_reachable_floor_waypoint()`에서 raw point를 agent 방향으로 `0.10m` 당긴다.

```python
raw_xy = raw_world[:2]
direction = agent_xy - raw_xy
waypoint_xy = raw_xy + normalize(direction) * 0.10
```

이 offset은 object surface나 wall에 찍힌 raw depth point를 그대로 goal로 쓰는
문제를 줄이기 위한 처리이다.

최종 waypoint는 다음 의미를 가진다.

```text
pixel-projected raw anchor보다 agent 쪽으로 10cm 가까운 world point
```

## 7. World Waypoint to Relative PointNav Goal

navigation loop에서는 매 step 현재 pose를 다시 읽는다.

```python
current_agent_xy = np.array([gps[0], -gps[1]])
current_heading = compass
```

고정된 `waypoint_world[:2]`와 현재 위치의 차이를 구한다.

```python
delta_world = waypoint_world[:2] - current_agent_xy
```

이를 현재 agent heading 기준 local 좌표로 회전한다.

```python
local_goal = R(-current_heading) @ delta_world
```

마지막으로 polar coordinate로 변환한다.

```python
rho = np.linalg.norm(local_goal)
theta = np.arctan2(local_goal[1], local_goal[0])
```

의미:

```text
rho   = 현재 agent 위치에서 waypoint까지의 평면 거리, meter
theta = 현재 agent heading 기준 waypoint의 상대 방향, radian
```

방향 해석:

```text
theta = 0 -> 정면
theta > 0 -> 왼쪽
theta < 0 -> 오른쪽
```

## 8. PointNav Policy Input

`DepthPointNavController.act()`는 다음 observation을 PointNav policy에 넣는다.

```python
observations = {
    "depth": depth_tensor,
    "pointgoal_with_gps_compass": torch.tensor([[rho, theta]]),
}
```

여기서 `depth_tensor`는 현재 Habitat `obs["depth"]`를 `224x224`로 resize한
normalized depth이다.

현재 controller 설정:

```python
depth_image_shape = (224, 224)
pointnav_stop_radius = 0.9
max_pointnav_steps = 32
reset_pointnav_on_new_waypoint = True
```

따라서:

```text
rho < 0.9m -> PointNav를 부르지 않고 stop action 0
한 waypoint당 최대 32 step 실행
새 waypoint가 생기면 PointNav recurrent state reset
```

## Final Algorithm

현재 구현의 전체 알고리즘은 다음과 같다.

```text
1. YOLOE가 target 또는 prior object bounding box를 찾는다.
2. 선택된 bbox center 주변 17x17 synthetic mask를 만든다.
3. mask 하단 band 평균을 waypoint pixel (u, v)로 사용한다.
4. Habitat normalized depth를 0.5m~5.0m metric depth로 복원한다.
5. (u, v)의 depth가 0.5 <= d < 5.0이면 사용한다.
6. invalid이면 주변 5x5 valid depth의 median을 사용한다.
7. 그래도 없으면 d = 5.0m를 사용한다.
8. pixel (u, v)와 metric depth d를 camera 3D point로 back-project한다.
9. 현재 camera pose를 이용해 camera 3D point를 world anchor로 변환한다.
10. raw world anchor를 agent 방향으로 0.10m 당겨 final world waypoint를 만든다.
11. 매 step final world waypoint를 현재 gps/compass pose 기준 local coordinate로 변환한다.
12. local coordinate를 polar coordinate (rho, theta)로 바꾼다.
13. normalized depth와 (rho, theta)를 PointNav policy에 입력한다.
```

여기서 depth `d`는 geometry 계산을 위한 metric depth이고, PointNav policy에
입력되는 depth image는 Habitat에서 받은 normalized depth이다. final waypoint는
world 좌표에 고정되며, `(rho, theta)`는 매 step 현재 pose 기준으로 다시 계산된다.
