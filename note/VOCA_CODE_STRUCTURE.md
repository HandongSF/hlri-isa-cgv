# VOCA 코드 및 모듈 구성 정리

이 문서는 현재 VOCA 코드 구조를 기준으로, 이어서 개발할 때 각 파일이 어떤 책임을 가지는지 빠르게 파악하기 위한 메모다.

## 현재 큰 구조

```text
hlri-isa-cgv/
  objnav_benchmark.py
  settings.py
  voca/
    habitat/
    navigation/
      objectnav/
      pointnav/
      waypoint/
    perception/
    planning/
    llm/
  third_party/
  note/
```

핵심 실행 경로는 `objnav_benchmark.py -> ObjectNavEpisodeRunner -> VOCANavigator -> VLMPlanner / DepthPointNavController` 순서다.

## 실행 진입점

### `objnav_benchmark.py`

ObjectNav benchmark 실행 진입점이다. 지금은 여러 episode를 순차 실행하는 top-level runner 역할만 맡는다.

주요 책임:
- CLI 인자 파싱
- Habitat ObjectNav 환경 생성
- YOLOE 모델 생성
- `VOCANavigator` 생성
- `ObjectNavEpisodeRunner.run_episode()`를 episode 수만큼 반복 호출
- episode별 metric을 `objnav_hm3d.csv`에 누적 저장

여기에 넣으면 안 좋은 것:
- 한 episode 내부 navigation loop
- panorama scan 절차
- VLM/YOLOE 판단 로직
- PointNav action 계산

현재 실행 명령:

```bash
python objnav_benchmark.py --eval_episodes 1
```

## Habitat 관련 모듈

### `voca/habitat/config.py`

Habitat 설정을 만든다.

주요 책임:
- HM3D/MP3D ObjectNav config 생성
- dataset path, scene dataset path 설정
- top-down map, collisions 등 measurement 설정
- forward step size, turn angle, success distance 설정
- depth PointNav 모드에서 GPS/Compass sensor와 depth sensor 설정

중요한 점:
- `depth_pointnav=True`일 때 depth sensor를 PointNav policy가 사용할 수 있게 맞춘다.
- 현재 benchmark는 `hm3d_config(..., depth_pointnav=True)`를 사용한다.

### `voca/habitat/camera.py`

Habitat RGB/depth camera 설정에서 pinhole camera intrinsic matrix를 만든다.

주요 책임:
- RGB/depth sensor width, height, hfov 일치 확인
- depth pixel을 world point로 투영할 때 필요한 intrinsic 계산

## ObjectNav 실행 계층

### `voca/navigation/objectnav/runner.py`

하나의 Habitat episode를 실제로 실행하는 클래스다.

주요 클래스:
- `ObjectNavEpisodeRunner`
- `ObjectNavEpisodeRunnerConfig`

주요 책임:
- `env.reset()`으로 episode 시작
- `env.step(action)` 실행
- RGB frame, top-down map frame 기록
- episode video 저장
- episode distance/time/metric 계산
- `VOCANavigator`가 판단한 action과 plan을 Habitat 환경에서 실제로 실행

현재 runner가 알고 있는 VOCA sensing maneuver:
- `FULL_SCAN_TURNS = 11`: 초기/검증/deadlock 재탐색용 360도 scan
- `LOCAL_SCAN_LEFT_TURNS = 3`: local scan 시작 전 왼쪽 90도 회전
- `LOCAL_SCAN_RIGHT_TURNS = 6`: 왼쪽 90도에서 오른쪽 90도까지 훑기
- `HABITAT_TURN_DEG = 30`: Habitat turn action 1회의 각도

주의할 점:
- runner는 Habitat 의존성을 가진다. `env.step`, `env.get_metrics`, `env.episode_over`, `top_down_map`은 runner 책임이다.
- 다만 scan maneuver는 navigation 알고리즘 성격도 있다. 지금은 과도한 추상화를 피하기 위해 runner 내부 helper와 상수로 명시해 둔 상태다.
- 나중에 Habitat 외 실행환경으로 옮길 때는 이 runner가 가장 먼저 바뀔 가능성이 높다.

### `voca/navigation/objectnav/navigator.py`

VOCA ObjectNav의 판단과 상태를 관리한다.

주요 클래스:
- `VOCANavigator`
- `VOCANavigatorConfig`
- `VOCANavigatorAction`
- `ObjectNavPlan`
- `LocalScanResult`

주요 책임:
- episode별 object goal reset
- text priors 요청
- 360도 panorama로 VLM/YOLOE plan 생성
- verification plan 생성/적용
- deadlock plan 생성/적용
- local scan 결과 평가
- `goal_flag`, `pending_verify`, `prev_boxes`, `heading_offset` 같은 navigation 상태 관리
- goal mask와 depth에서 reachable PointNav waypoint 생성
- 현재 waypoint 기준으로 depth PointNav action 계산
- LLM/YOLOE runtime metric 제공

현재 내부 의존성:
- `VLMPlanner`: VLM/LLM + YOLOE 기반 goal/mask 선택
- `DepthPointNavController`: depth + pointgoal -> low-level action
- `voca.navigation.pointnav.geometry`: mask/depth/pixel/world/pointgoal 변환
- `voca.navigation.waypoint.reachable`: raw depth waypoint를 agent 쪽으로 약간 offset

주의할 점:
- `navigator.py`는 아직 ObjectNav state machine과 depth waypoint 생성 책임을 같이 가지고 있다.
- 다음 리팩토링 후보는 depth waypoint 생성 로직을 `pointnav/waypoint_builder.py` 같은 별도 모듈로 빼는 것이다.

## PointNav 관련 모듈

### `voca/navigation/pointnav/policy.py`

third-party VLFM PointNav policy를 VOCA 쪽 API로 감싼 wrapper다.

주요 책임:
- checkpoint path와 device를 받아 vendored PointNav policy 로드
- `act(observations, masks, deterministic)` 호출 전달
- recurrent state reset

직접 알고리즘을 구현하는 파일은 아니고, 외부 policy와 VOCA 사이의 adapter다.

### `voca/navigation/pointnav/controller.py`

PointNav policy를 local controller처럼 쓰게 해주는 계층이다.

주요 책임:
- Habitat depth observation을 PointNav policy 입력 크기 `(224, 224)`로 변환
- `(rho, theta)` pointgoal tensor 생성
- recurrent mask 관리
- waypoint별 step count 관리
- max step 초과 시 stop 반환
- 새 waypoint가 들어오면 policy state reset

현재 `VOCANavigator.act()`가 이 controller를 호출한다.

### `voca/navigation/pointnav/geometry.py`

depth 기반 waypoint와 pointgoal 계산 유틸이다.

주요 책임:
- goal mask에서 target pixel 추출
- Habitat normalized depth를 metric depth로 복원
- pixel + depth + camera intrinsic/extrinsic으로 world point 계산
- world waypoint와 현재 agent pose에서 `(rho, theta)` 계산

중요한 데이터 구조:
- `DepthWaypoint`
- `PointGoal`

### `voca/navigation/waypoint/reachable.py`

raw depth waypoint를 그대로 쓰지 않고 agent 방향으로 약간 offset해 reachable한 목표점처럼 만든다.

현재 구현:
- raw anchor world point에서 agent 방향으로 `OFFSET_FALLBACK_M = 0.10m` 이동
- 결과 `DepthWaypoint.target_kind = "offset_from_anchor"`

아직 navmesh projection 같은 정교한 reachable check는 아니다.

## Planning / Perception / LLM

### `voca/planning/vlm_planner.py`

VLM/LLM와 YOLOE detection을 결합해 다음 subgoal을 고르는 큰 planner다.

주요 책임:
- text priors 요청
- panorama image 기반 VLM direction 선택
- YOLOE class prompt 갱신
- priors 기반 detection scoring
- goal image, goal mask, debug visualization 생성
- bbox similarity로 deadlock 여부 판단에 필요한 비교 제공
- LLM/YOLOE duration과 call count 기록

주의할 점:
- 현재 가장 큰 파일 중 하나다.
- LLM 호출, parser, perception postprocess, planner state가 한 파일에 섞여 있다.
- 다만 동작 리스크가 크므로 현재는 runner/navigator 경계 안정화 이후에 분리하는 것이 좋다.

### `voca/perception/yoloe_detector.py`

YOLOE 모델 초기화와 detection wrapper다.

주요 책임:
- checkpoint path resolve
- YOLOE 모델 로드 및 device 설정
- text prompt class 설정
- RGB image에서 bbox/mask/confidence/class id 추출

중요한 데이터 구조:
- `Detections`

### `voca/llm/`

LLM/VLM 호출과 prompt/parser 모듈이다.

주요 파일:
- `gemini_request.py`: Gemini/Vertex AI backend 호출
- `ollama_request.py`: Ollama backend 호출
- `navigation_prompts.py`: navigation prompt와 prior class list
- `priors_parser.py`: LLM 응답 JSON parsing, priors 정리

`settings.LLM_BACKEND` 값에 따라 `VLMPlanner`가 Gemini 또는 Ollama backend를 선택한다.

## third_party

### `third_party/vlfm_pointnav/`

VLFM PointNav policy를 vendored dependency로 보관한다.

현재 사용 경로:
- `voca/navigation/pointnav/policy.py`가 `third_party.vlfm_pointnav.pointnav_policy.WrappedPointNavResNetPolicy`를 감싼다.

주의할 점:
- checkpoint 로딩 시 예전 `vlfm.*` module path가 필요할 수 있어, vendored policy 쪽에 compatibility shim이 들어가 있다.
- third-party 코드는 가능하면 직접 수정하지 않고 VOCA wrapper에서 감싸는 방향이 좋다.

## 현재 책임 분리 상태

현재는 다음 정도로 나뉘어 있다.

```text
objnav_benchmark.py
  여러 episode benchmark loop, setup, CSV 저장

ObjectNavEpisodeRunner
  단일 Habitat episode 실행, env.step, recording, episode metric 생성

VOCANavigator
  ObjectNav 판단/state, VLM plan, local scan 평가, PointNav action 생성

VLMPlanner
  LLM/VLM + YOLOE 기반 subgoal/mask 선택

DepthPointNavController
  depth + pointgoal -> low-level action
```

이 구조는 당장 실험을 돌리기에는 충분히 명확한 중간 상태다.

## 아직 완벽히 분리되지 않은 부분

### 1. Runner에 navigation maneuver가 일부 남아 있음

`runner.py`는 Habitat 실행자지만, full scan/local scan action sequence도 알고 있다.

현재는 다음 상수로 의도를 명시해 둔 상태다.

```python
FULL_SCAN_TURNS
LOCAL_SCAN_LEFT_TURNS
LOCAL_SCAN_RIGHT_TURNS
LOCAL_SCAN_START_DEG
HABITAT_TURN_DEG
```

완전히 분리하려면 `VOCANavigator`가 command/action sequence를 반환하고 runner가 실행만 하게 만들 수 있다. 다만 지금은 과설계일 수 있어 보류한다.

### 2. Navigator가 depth waypoint 생성도 맡고 있음

`VOCANavigator` 안에 pose 변환, depth mask -> world waypoint 변환, reachable offset 적용이 들어 있다.

다음 후보:

```text
voca/navigation/pointnav/waypoint_builder.py
```

여기로 `build_current_depth_waypoint`, `refresh_depth_waypoint` 일부를 옮기면 navigator는 ObjectNav 상태와 판단에 더 집중할 수 있다.

### 3. VLMPlanner가 큼

`voca/planning/vlm_planner.py`는 LLM 호출, priors, YOLOE postprocess, bbox 비교, visualization이 섞여 있다.

나중에 나눈다면 후보는 다음과 같다.

```text
voca/planning/priors.py
voca/planning/subgoal_selector.py
voca/planning/deadlock.py
voca/planning/visualization.py
```

하지만 지금 당장 나누면 동작 리스크가 크다.

## 개발할 때 기준

새 코드를 넣을 위치는 다음 기준으로 정한다.

- Habitat config/env/metric/topdown/video 실행 문제면 `voca/habitat` 또는 `ObjectNavEpisodeRunner`
- ObjectNav 상태, replan, verification, deadlock 판단이면 `VOCANavigator`
- depth PointNav action 또는 pointgoal 입력 문제면 `voca/navigation/pointnav`
- mask/depth pixel/world 변환이면 `pointnav/geometry.py` 또는 향후 `pointnav/waypoint_builder.py`
- object detection이면 `voca/perception/yoloe_detector.py`
- LLM prompt, parsing, backend 호출이면 `voca/llm`
- VLM/YOLOE 결과를 결합해 subgoal을 고르는 문제면 `voca/planning/vlm_planner.py`

## 다음으로 안전한 리팩토링 후보

우선순위는 다음 순서가 좋다.

1. `VOCANavigator`의 depth waypoint 생성 책임을 `pointnav/waypoint_builder.py`로 이동
2. `runner.py`의 recording/video 저장을 별도 `EpisodeRecorder`로 이동할지 검토
3. `VLMPlanner`에서 priors/parsing/detection scoring을 작게 분리
4. full scan/local scan을 `NavigationCommand` 형태로 추상화할지 검토

당장은 1번이 가장 안전하다. ObjectNav 알고리즘 흐름을 크게 바꾸지 않으면서 `navigator.py`의 책임을 줄일 수 있다.
