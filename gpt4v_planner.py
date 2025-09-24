import numpy as np
from llm_utils.gpt_request import gptv_response
from llm_utils.nav_prompt import GPT4V_PROMPT
from llm_utils.priors_parser import parse_llm_json, extract_priors
from cv_utils.yoloe_tools import *
import cv2
import time  # <-- 추가
# 변경/추가 import
import os   # NEW
import json # NEW


class GPT4V_Planner:
    def __init__(self,yoloe_model):
        self.gptv_trajectory = []
        self.yoloe_model = yoloe_model
        self.detect_objects = ['bed','sofa','chair','plant','tv','toilet','floor']
        # ---- LLM/플래너/모듈별 시간 계측 저장소 ----
        self.llm_call_count = 0
        self.llm_durations = []          # 각 LLM 호출 소요시간(초)
        self.planner_durations = []      # 각 make_plan 호출 전체 소요시간(초)
        # ---- Priors 저장소 ----
        self.priors_log = []       # 에피소드 동안 쌓인 priors 히스토리 (리스트)
        self.latest_priors = None  # 가장 최근 priors 스냅샷 (딕셔너리)
        self._last_prompt = None
        # 🔧 내부 프롬프트/클래스 캐시
        self._prompt_classes = None
        self._floor_aliases = ['floor', 'ground', 'flooring']
        self._negatives = []

        # 📁 로깅 폴더 (끄기: None)
        self._logdir = None  # NEW


    # -------------------------
    # (선택) 로그 저장 활성화
    # -------------------------
    def enable_run_logger(self, out_dir: str):  # NEW
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        self._logdir = out_dir

    def _dump_llm_call(self, snapshot: dict):  # NEW
        """각 LLM 호출 결과를 JSONL로 누적 저장 + 원본/파노라마/선택 이미지 파일로도 보관."""
        if not self._logdir:
            return
        # 1) JSONL
        path = os.path.join(self._logdir, "llm_calls.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        # 2) 텍스트(raw)
        if "raw" in snapshot and isinstance(snapshot["raw"], str):
            idx = snapshot.get("call_index", len(self.priors_log))
            with open(os.path.join(self._logdir, f"raw_{idx:04d}.txt"), "w", encoding="utf-8") as f:
                f.write(snapshot["raw"])

    def export_priors_log(self, path: str):  # NEW
        """에피소드 종료 시 전체 priors_log를 한 번에 저장하고 싶다면 호출."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.priors_log, f, ensure_ascii=False, indent=2)

    
    def reset(self,object_goal):
        # translation to align for the detection model
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv'
        else:
            self.object_goal = object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

        # ---- 에피소드 시작 시 계측 초기화 ----
        self.llm_call_count = 0
        self.llm_durations = []
        self.planner_durations = []
        self.priors_log = []
        self.latest_priors = None

        # 🔧 캐시/프롬프트 상태 완전 초기화
        self._last_prompt = None
        self._prompt_classes = None
        self._floor_aliases = ['floor', 'ground', 'flooring']

        self._negatives = []        # Lookalikes(네거티브) 캐시
        self._set_prompt_from_priors({})  # 빈 priors로 초기 클래스셋 구성

    def concat_panoramic(self,images,angles):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10,3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
               continue
            copy_images[i] = cv2.putText(copy_images[i],"Angle %d"%angles[i],(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i//2) % 3
            background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,10*(col+1)+col*width:10*(col+1)+col*width+width,:] = copy_images[i]
        return background_image
    
    def make_plan(self, pano_images):
        _plan_t0 = time.perf_counter()

        # 1) LLM로 방향/priors 결정
        direction, goal_flag = self.query_gpt4v(pano_images)
        direction_image = pano_images[direction]  # Habitat obs['rgb']는 RGB 배열

        # 2) YOLOE 프롬프트(클래스셋) 보장
        self._set_prompt_from_priors(self.latest_priors or {})
        prompt_classes = getattr(self, "_prompt_classes", None)
        floor_aliases  = getattr(self, "_floor_aliases", ['floor','ground','flooring'])
        if not prompt_classes:
            self._set_prompt_from_priors(self.latest_priors or {})
            prompt_classes = self._prompt_classes

        # 3) YOLOE 추론: "박스만" 사용 (세그 헤드 OFF)
        det = yoloe_detection(
            image=direction_image,       # RGB 그대로
            target_classes=prompt_classes,
            model=self.yoloe_model,
            box_threshold=0.25,
            iou_threshold=0.50,
            run_extra_nms=False,
            use_text_prompt=False,
            retina_masks=False,          # ★ 세그멘테이션 비활성화
        )

        H, W = direction_image.shape[:2]
        debug_image = np.array(direction_image)

        # 4) 클래스 인덱스 계산
        try:
            goal_idx = prompt_classes.index(self.object_goal)
        except ValueError:
            goal_idx = -1
        floor_idx_set = {prompt_classes.index(n) for n in floor_aliases if n in prompt_classes}

        # 5) 바운딩 박스만으로 중심점 선택
        #    (라이브러리별 속성명을 모두 커버)
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None:
            xyxy = getattr(det, "boxes", None)
        cls_ids = getattr(det, "class_id", np.empty((0,), dtype=int))
        has_boxes = xyxy is not None and len(xyxy) > 0 and cls_ids.size > 0

        # 기본값: 화면 중앙
        px, py = W // 2, H // 2

        def _pick_largest(indices):
            if len(indices) == 0:
                return None
            areas = []
            for i in indices:
                x1, y1, x2, y2 = map(float, xyxy[i])
                areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            return int(indices[int(np.argmax(areas))])

        if has_boxes:
            # 5-1) 목표 물체가 있으면: 가장 큰 박스 중심
            if goal_idx >= 0 and np.any(cls_ids == goal_idx):
                goal_flag = True
                idxs = np.where(cls_ids == goal_idx)[0]
                top = _pick_largest(idxs)
                if top is not None:
                    x1, y1, x2, y2 = map(float, xyxy[top])
                    px, py = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            else:
                # 5-2) 목표가 없으면: 바닥(floor) 중 가장 큰 박스 중심
                goal_flag = False
                sel = np.where(np.isin(cls_ids, list(floor_idx_set)))[0]
                top = _pick_largest(sel)
                if top is not None:
                    x1, y1, x2, y2 = map(float, xyxy[top])
                    px, py = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

        # 6) 디버그 표식 & 정책용 마스크(박스 없이도 동일 규격)
        r = 8
        x1, x2 = max(0, px - r), min(W, px + r)
        y1, y2 = max(0, py - r), min(H, py + r)

        # 디버그 사각형(시각화)
        cv2.rectangle(debug_image, (px - r, py - r), (px + r, py + r), (255, 0, 0), -1)

        # ★ 정책 입력용 마스크: 2D uint8, 0/255 (PixelNav가 내부에서 /255.0 처리)
        debug_mask = np.zeros((H, W), dtype=np.uint8)
        debug_mask[y1:y2, x1:x2] = 255

        # 로깅(선택): 내부 트래젝토리에는 0/255 저장
        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(debug_mask.copy())

        self.planner_durations.append(time.perf_counter() - _plan_t0)

        # ★ Policy_Agent.reset()이 goal_image를 BGR→RGB로 바꾸므로,
        #   여기서는 BGR로 넘겨 일관 유지
        goal_image_bgr = cv2.cvtColor(direction_image, cv2.COLOR_RGB2BGR)
        return goal_image_bgr, debug_mask, debug_image, direction, goal_flag




    def _set_prompt_from_priors(self, priors: dict):
        floor_aliases = ['floor', 'ground', 'flooring']

        supports   = set(map(str.lower, (priors or {}).get("Supports", []) or []))
        cooccurs   = set(map(str.lower, (priors or {}).get("StrongCooccurs", []) or []))
        gateways   = set(map(str.lower, (priors or {}).get("Gateways", []) or []))
        lookalikes = set(map(str.lower, (priors or {}).get("Lookalikes", []) or []))

        # Scene→Object 힌트 평탄화 (너무 커지지 않도록 전체를 합치되 중복 제거)
        hint_objs = []
        hints = (priors or {}).get("SceneToObjectHints", {}) or {}
        for objs in hints.values():
            hint_objs.extend(list(map(str.lower, objs or [])))
        hint_set = set(hint_objs)

        # YOLOE 포지티브: 기본 탐지 + floor + (supports ∪ cooccurs ∪ gateways ∪ hint_objs)
        extra_positive = sorted((supports | cooccurs | gateways | hint_set | set(floor_aliases)))
        prompt_classes = list(dict.fromkeys(self.detect_objects + extra_positive))

        # 네거티브(lookalikes)는 후처리에서 패널티/제외용으로 보관
        self._negatives = sorted(list(lookalikes))

        # 모델 클래스셋 적용 (변경 시에만)
        if self._last_prompt != tuple(prompt_classes):
            set_yoloe_classes(self.yoloe_model, prompt_classes)
            self._last_prompt = tuple(prompt_classes)

        self._prompt_classes = prompt_classes
        self._floor_aliases = floor_aliases


    def query_gpt4v(self, pano_images):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)

        cv2.imwrite("monitor-panoramic.jpg", inference_image)
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        self.gptv_trajectory.append("\nInput:\n%s \n" % text_content)
        self.panoramic_trajectory.append(inference_image)

        raw_answer = None
        parsed = None

        for _ in range(10):
            t0 = time.perf_counter()
            try:
                raw_answer = gptv_response(text_content, inference_image, GPT4V_PROMPT)
            except Exception:
                raw_answer = None
            finally:
                self.llm_call_count += 1
                self.llm_durations.append(time.perf_counter() - t0)

            if not raw_answer:
                continue

            # ✅ angle/priors만 안정적으로 파싱
            parsed = parse_llm_json(raw_answer, valid_angles=list(map(int, angles.tolist())))
            if not parsed:
                continue

            # angle 확인
            try:
                a = int(parsed["Angle"])
            except Exception:
                a = None

            if a is None or a not in set(int(x) for x in angles):
                continue

            # ✅ priors 표준 형태로 추출 & 저장
            priors = extract_priors(parsed, raw_answer)

            snapshot = {
                "target": self.object_goal,
                "angles": list(map(int, angles.tolist())) if hasattr(angles, "tolist") else list(angles),
                "selected_angle": int(a),
                "priors": priors,
                "raw": raw_answer,
            }
            self.priors_log.append(snapshot)
            self.latest_priors = priors
            print("Priors: ", priors)
            print("[LLM] angle:", a)
            print("[LLM] supports/cooccurs/gateways/lookalikes sizes:",
                len(priors.get("Supports", [])),
                len(priors.get("StrongCooccurs", [])),
                len(priors.get("Gateways", [])),
                len(priors.get("Lookalikes", [])))
            break  # 성공 조건

        self.gptv_trajectory.append("GPT-4V Answer:\n%s" % (raw_answer if raw_answer is not None else "<EMPTY>"))
        self.panoramic_trajectory.append(inference_image)

        try:
            idx = (int(parsed['Angle'] // 30)) % max(1, len(pano_images))
            return idx, bool(parsed.get('Flag', False))
        except Exception:
            return np.random.randint(0, max(1, len(pano_images))), False


    def apply_priors_on_image(self, image: np.ndarray,
                            conf_threshold: float = 0.15,
                            iou_threshold: float = 0.50):
        # 프롬프트/클래스셋 확보
        prompt_classes = getattr(self, "_prompt_classes", None)
        floor_aliases  = getattr(self, "_floor_aliases", ['floor','ground','flooring'])
        if not prompt_classes:
            self._set_prompt_from_priors(self.latest_priors or {})
            prompt_classes = self._prompt_classes

        # 입력은 Habitat의 RGB
        rgb = image
        H, W = rgb.shape[:2]

        # ★ 정책용 반환 마스크: 2D uint8 (0/255)
        debug_mask = np.zeros((H, W), dtype=np.uint8)

        # YOLOE: 박스만 사용 (세그 OFF)
        det = yoloe_detection(
            image=rgb,
            target_classes=prompt_classes,
            model=self.yoloe_model,
            box_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            run_extra_nms=False,
            use_text_prompt=False,
            retina_masks=False,   # ★ 세그멘테이션 비활성화
        )



        # 결과 접근 (라이브러리 래퍼가 xyxy 또는 boxes를 쓸 수 있어 대비)
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None:
            xyxy = getattr(det, "boxes", None)
        cls_ids = getattr(det, "class_id", None)

        # 유틸
        def _safe_boxes_ok():
            return (xyxy is not None) and (len(xyxy) > 0) and (cls_ids is not None) and (np.size(cls_ids) > 0)

        def _area(i):
            x1, y1, x2, y2 = map(float, xyxy[i])
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        def _center(i):
            x1, y1, x2, y2 = map(float, xyxy[i])
            return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

        def _pick_largest(indices):
            if len(indices) == 0:
                return None
            areas = [_area(i) for i in indices]
            return int(indices[int(np.argmax(areas))])

        # 목표 클래스 인덱스
        goal_name = getattr(self, "object_goal", "") or ""
        try:
            goal_idx = prompt_classes.index(goal_name)
        except ValueError:
            goal_idx = -1

        vis_bgr = draw_detections_bgr(rgb, det, goal_idx=goal_idx)

        # 0) 기본값: 화면 중앙
        px, py = W // 2, H // 2
        pri_flag = False  # 목표 직접 발견 시 True

        if _safe_boxes_ok():
            cls_ids = np.asarray(cls_ids).astype(int)

            # 1) 타깃 우선: 같은 클래스 중 가장 큰 박스
            if goal_idx >= 0 and np.any(cls_ids == goal_idx):
                inds = np.where(cls_ids == goal_idx)[0]
                top = _pick_largest(inds)
                if top is not None:
                    px, py = _center(top)
                    pri_flag = True
                else:
                    # 2) priors 기반 스코어링 (네거티브는 강한 패널티)
                    pri = self.latest_priors or {
                        "Supports": [], "StrongCooccurs": [], "Gateways": [],
                        "Lookalikes": [], "SceneToObjectHints": {}
                    }
                    supports = set(map(str.lower, pri.get("Supports", [])))
                    cooccurs = set(map(str.lower, pri.get("StrongCooccurs", [])))
                    gateways = set(map(str.lower, pri.get("Gateways", [])))
                    NEG      = set(map(str.lower, getattr(self, "_negatives", []) or []))

                    WEIGHTS = {"supports": 0.6, "cooccurs": 0.4, "gateways": 0.2, "negative": -1.0}

                    def _name(ci):
                        return str(prompt_classes[int(ci)]).lower() if 0 <= int(ci) < len(prompt_classes) else ""

                    best_i, best_s, best_a = None, -1e9, -1.0
                    for i, ci in enumerate(cls_ids):
                        name = _name(ci)
                        if name in NEG:
                            score = WEIGHTS["negative"]
                        elif name in supports:
                            score = WEIGHTS["supports"]
                        elif name in cooccurs:
                            score = WEIGHTS["cooccurs"]
                        elif name in gateways:
                            score = WEIGHTS["gateways"]
                        else:
                            score = 0.0
                        area = _area(i)
                        if (score > best_s) or (np.isclose(score, best_s) and area > best_a):
                            best_i, best_s, best_a = i, score, area

                    if best_i is not None and best_a > 0 and best_s > WEIGHTS["negative"]:
                        px, py = _center(best_i)
                    else:
                        # 3) floor 최대 박스 폴백
                        floor_set = set(map(str.lower, floor_aliases))
                        floor_inds = [i for i, ci in enumerate(cls_ids) if _name(ci) in floor_set]
                        top = _pick_largest(floor_inds)
                        if top is not None:
                            px, py = _center(top)
                        # else 중앙 유지

        # 4) 정책용 마스크(정사각형 점)
        r = 8
        x1, x2 = max(0, px - r), min(W, px + r)
        y1, y2 = max(0, py - r), min(H, py + r)
        debug_mask[y1:y2, x1:x2] = 255
        # 선택된 포인트도 같이 그리려면:
        cv2.drawMarker(
            vis_bgr, (px, py),
            (0, 255, 255) if not pri_flag else (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2
        )
        # 반환: (이미지 BGR, 마스크 uint8, 목표발견여부), 디버깅용 이미지
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), debug_mask, pri_flag, vis_bgr
