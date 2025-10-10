import numpy as np
from llm_utils.gpt_request import gptv_response, gpt_response
from llm_utils.nav_prompt import GPT4V_PROMPT, PRIORS_PROMPT, PRIOR_CLASS_LIST
from llm_utils.priors_parser import parse_llm_json, extract_priors, parse_decision_json
from cv_utils.yoloe_tools import *
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import time  # <-- 추가
# 변경/추가 import
import os   # NEW
import json # NEW
import math


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
        """
        Returns:
            goal_image_bgr (np.ndarray, BGR): 정책 입력용 목표 이미지(BGR)
            debug_mask     (np.ndarray, uint8 HxW): 정책 입력용 0/255 마스크
            debug_image    (np.ndarray, RGB): 선택 포인트가 네모로 찍힌 원본 방향 프레임(디버그)
            vis_bgr        (np.ndarray, BGR): YOLOE 검출(박스/라벨) + 선택 포인트가 표시된 오버레이
            direction      (int): 선택된 파노라마 인덱스
            goal_flag      (bool): 타깃이 직접 검출되었는지 여부
        """
        _plan_t0 = time.perf_counter()

        # 1) LLM로 진행 방향/priors 결정
        direction, _ = self.query_gpt4v(pano_images)
        direction_image = pano_images[direction]  # RGB

        # 2) YOLOE 클래스 프롬프트 보장 (priors 반영)
        self._set_prompt_from_priors(self.latest_priors or {})

        # 3) priors-기반 웨이포인트 선택을 apply_priors_on_image로 통일
        #    - 이 함수가 YOLOE 박스 → priors 스코어링 → 바닥 폴백까지 수행
        goal_image_bgr, debug_mask, pri_flag, vis_bgr = self.apply_priors_on_image(
            image=direction_image,
        )

        # 4) 디버그용: 원본 RGB 프레임에 선택 포인트(마스크 중심) 표시
        debug_image = np.array(direction_image)  # RGB 복사
        ys, xs = np.where(debug_mask > 0)
        if len(xs) > 0:
            px = int(np.mean(xs))
            py = int(np.mean(ys))
        else:
            H, W = debug_image.shape[:2]
            px, py = W // 2, H // 2
        r = 8
        cv2.rectangle(debug_image, (px - r, py - r), (px + r, py + r), (255, 0, 0), -1)  # RGB에 표식

        # 5) 내부 로그/트래젝토리 업데이트
        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(debug_mask.copy())
        self.planner_durations.append(time.perf_counter() - _plan_t0)

        # 6) 리턴 (기존 시그니처 유지)
        return goal_image_bgr, debug_mask, debug_image, vis_bgr, direction, pri_flag


    def _set_prompt_from_priors(self, priors: dict):
        floor_aliases = ['floor', 'ground', 'flooring']

        supports   = set(map(str.lower, (priors or {}).get("Supports", []) or []))
        cooccurs   = set(map(str.lower, (priors or {}).get("StrongCooccurs", []) or []))
        gateways   = set(map(str.lower, (priors or {}).get("Gateways", []) or []))
        lookalikes = set(map(str.lower, (priors or {}).get("Lookalikes", []) or []))

        # 모든 priors (lookalikes 포함)을 YOLOE 클래스셋에 포함
        extra_positive = sorted(
            supports | cooccurs | gateways | lookalikes | set(floor_aliases)
        )
        prompt_classes = list(dict.fromkeys(self.detect_objects + extra_positive))

        # lookalikes도 YOLOE로 감지되도록 추가하되,
        # 후처리 scoring 단계에서만 패널티를 줌
        self._negatives = sorted(list(lookalikes))

        # 모델 클래스셋 적용 (변경 시에만)
        if self._last_prompt != tuple(prompt_classes):
            set_yoloe_classes(self.yoloe_model, prompt_classes)
            self._last_prompt = tuple(prompt_classes)

        self._prompt_classes = prompt_classes
        self._floor_aliases = floor_aliases

    def query_priors_text(self):
        """
        이미지 없이 텍스트만으로 PRIORS를 받아온다.
        - 시스템 프롬프트: PRIORS_PROMPT
        - 유저 프롬프트: "<Target Object>:<name>"
        - 쿼리 시도: 10회
        반환: 표준 priors(dict)
            {
            "Supports": [...],
            "StrongCooccurs": [...],
            "Gateways": [...],
            "Lookalikes": [...]
            }
        """
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        # (선택) 트래젝토리 로그에 입력 기록
        self.gptv_trajectory.append("\nInput(priors):\n%s \n" % text_content)

        raw_answer = None
        priors = None

        for try_idx in range(10):
            t0 = time.perf_counter()
            try:
                # 이미지 없이 텍스트 전용 호출
                raw_answer = gpt_response(text_content, system_prompt=PRIORS_PROMPT + PRIOR_CLASS_LIST)
            except Exception:
                raw_answer = None
            finally:
                self.llm_call_count += 1
                self.llm_durations.append(time.perf_counter() - t0)

            if not raw_answer:
                continue

            # priors 전용 파서 (각도/플래그/리즌 미사용)
            parsed = None
            try:
                parsed = parse_llm_json(raw_answer)  # alias: priors만 반환
            except Exception:
                parsed = None
            if not parsed:
                continue

            # 표준 PRIORS 딕셔너리만 추출
            priors = extract_priors(parsed, raw_answer) or {}

            # 키 보장
            for k in ("Supports", "StrongCooccurs", "Gateways", "Lookalikes"):
                priors.setdefault(k, [])

            # 내부 로그(원하면 파일 저장 훅도 사용 가능)
            snapshot = {
                "target": self.object_goal,
                "priors": priors,
                "raw": raw_answer,
                "call_index": try_idx,
                "call_type": "text_priors",
            }
            self.priors_log.append(snapshot)
            # self._dump_llm_call(snapshot)  # 주석 해제 시 파일로도 저장

            print("[LLM/PRIORS] sizes:",
                len(priors.get("Supports", [])),
                len(priors.get("StrongCooccurs", [])),
                len(priors.get("Gateways", [])),
                len(priors.get("Lookalikes", [])))
            break  # 성공 시도 종료

        # 실패 시 안전 디폴트
        if not isinstance(priors, dict):
            priors = {
                "Supports": [],
                "StrongCooccurs": [],
                "Gateways": [],
                "Lookalikes": [],
            }

        # 상태 업데이트 + YOLOE 클래스셋 적용
        self.latest_priors = priors
        self._set_prompt_from_priors(priors)

        # (선택) 응답 원문도 트래젝토리에 남김
        self.gptv_trajectory.append("PRIORS Answer:\n%s" % (raw_answer if raw_answer is not None else "<EMPTY>"))

        return priors


    def query_gpt4v(self, pano_images):
        """
        LLM에서 Reason/Angle/Flag만 받아와 진행 방향을 정한다.
        priors는 사용/저장하지 않는다.
        """
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)

        # (옵션) 기록용 저장
        cv2.imwrite("monitor-panoramic.jpg", inference_image)

        text_content = "<Target Object>:{}\n".format(self.object_goal)
        self.gptv_trajectory.append("\nInput:\n%s \n" % text_content)
        self.panoramic_trajectory.append(inference_image)

        raw_answer = None
        parsed = None

        def _to_bool(x):
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return x != 0
            if isinstance(x, str):
                return x.strip().lower() in ("true", "1", "yes", "y", "t")
            return False

        valid_angles = set(int(x) for x in angles.tolist())

        for try_idx in range(10):
            t0 = time.perf_counter()
            try:
                # ⚠️ GPT4V_PROMPT는 시스템프롬프트로 들어간다고 가정
                raw_answer = gptv_response(text_content, inference_image, GPT4V_PROMPT)
            except Exception:
                raw_answer = None
            finally:
                self.llm_call_count += 1
                self.llm_durations.append(time.perf_counter() - t0)

            if not raw_answer:
                continue

            # Reason/Angle/Flag만 파싱
            parsed = parse_decision_json(raw_answer, valid_angles=list(valid_angles))
            if not parsed:
                continue

            # Angle 검증
            try:
                a = int(parsed.get("Angle"))
            except Exception:
                a = None
            if a is None or a not in valid_angles:
                continue

            # Flag 안정 변환
            flag = _to_bool(parsed.get("Flag", False))
            reason = parsed.get("Reason", "")

            # (선택) 호출 로그 저장 — priors는 비움
            snapshot = {
                "target": self.object_goal,
                "angles": list(map(int, angles.tolist())) if hasattr(angles, "tolist") else list(angles),
                "selected_angle": int(a),
                "flag": bool(flag),
                "reason": reason,
                "raw": raw_answer,
                "call_index": try_idx,
            }

            # 파일로 남기고 싶으면 주석 해제
            # self._dump_llm_call(snapshot)

            print("[LLM] angle:", a, "| flag:", flag)
            if reason:
                print("[LLM] reason:", reason)
            break  # 성공

        self.gptv_trajectory.append("GPT-4V Answer:\n%s" % (raw_answer if raw_answer is not None else "<EMPTY>"))
        self.panoramic_trajectory.append(inference_image)

        try:
            idx = (int(parsed['Angle']) // 30) % max(1, len(pano_images))
            return idx, _to_bool(parsed.get('Flag', False))
        except Exception:
            return np.random.randint(0, max(1, len(pano_images))), False


    def apply_priors_on_image(self, image: np.ndarray,
                            conf_threshold: float = 0.10,
                            iou_threshold: float = 0.50,
                            return_boxes: bool = False):  # ★ 추가
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

        # 결과 접근
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None:
            xyxy = getattr(det, "boxes", None)
        cls_ids = getattr(det, "class_id", None)
        # ★ 박스별 confidence(1D) 확보
        box_conf = np.asarray(det.confidence).astype(np.float32) if getattr(det, "confidence", None) is not None else None

        # ---- 여기서부터: 표준화된 바운딩박스 리스트 생성 ----
        boxes_std = []  # list[dict]
        def _push_box(i):
            x1, y1, x2, y2 = map(float, xyxy[i])
            # clamp
            x1 = max(0.0, min(x1, W - 1)); y1 = max(0.0, min(y1, H - 1))
            x2 = max(x1+1.0, min(x2, W));  y2 = max(y1+1.0, min(y2, H))
            w = x2 - x1; h = y2 - y1
            cx = x1 + 0.5 * w; cy = y1 + 0.5 * h
            ci = int(cls_ids[i]) if cls_ids is not None and np.size(cls_ids) > i else None
            cn = str(prompt_classes[ci]).lower() if (ci is not None and 0 <= ci < len(prompt_classes)) else None
            sc = float(box_conf[i]) if box_conf is not None and len(box_conf) > i else None
            boxes_std.append({
                "cls": cn,
                "cls_id": ci,
                "score": sc,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_xywh": [int(x1), int(y1), int(w), int(h)],
                "center": [float(cx), float(cy)],
                "area": float(w*h),
                "norm": {
                    "cx": float(cx / W), "cy": float(cy / H),
                    "w": float(w / W),  "h": float(h / H),
                    "area": float((w*h) / (W*H + 1e-12)),
                }
            })

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

        # IoU
        def _iou(i, j):
            x1a, y1a, x2a, y2a = map(float, xyxy[i])
            x1b, y1b, x2b, y2b = map(float, xyxy[j])
            inter_x1 = max(x1a, x1b); inter_y1 = max(y1a, y1b)
            inter_x2 = min(x2a, x2b); inter_y2 = min(y2a, y2b)
            inter_w  = max(0.0, inter_x2 - inter_x1)
            inter_h  = max(0.0, inter_y2 - inter_y1)
            inter    = inter_w * inter_h
            union    = _area(i) + _area(j) - inter + 1e-9
            return inter / union

        # 목표 클래스 인덱스
        goal_name = getattr(self, "object_goal", "") or ""
        try:
            goal_idx = prompt_classes.index(goal_name)
        except ValueError:
            goal_idx = -1

        vis_bgr = draw_detections_bgr(rgb, det, goal_idx=goal_idx)

        # 기본 좌표/플래그
        px, py = W // 2, H // 2
        pri_flag = False

        if _safe_boxes_ok():
            # 표준화 리스트 채우기
            for i in range(len(xyxy)):
                _push_box(i)

            cls_ids = np.asarray(cls_ids).astype(int)

            def _name(ci):
                return str(prompt_classes[int(ci)]).lower() if 0 <= int(ci) < len(prompt_classes) else ""

            def _norm_area(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                return max(0.0, (x2 - x1) * (y2 - y1)) / float(W * H + 1e-6)

            def _bottom_bias(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                h_bottom = max(y1, y2)
                return (h_bottom / H)

            # priors 집합
            pri = self.latest_priors or {"Supports": [], "StrongCooccurs": [], "Gateways": [], "Lookalikes": []}
            supports = set(map(str.lower, pri.get("Supports", [])))
            cooccurs = set(map(str.lower, pri.get("StrongCooccurs", [])))
            gateways = set(map(str.lower, pri.get("Gateways", [])))
            NEG      = set(map(str.lower, getattr(self, "_negatives", []) or []))

            WEIGHTS = {"supports": 0.4, "cooccurs": 0.2, "gateways": 0.4, "negative": 0.0}
            ALPHA_CONF, BETA_AREA, GAMMA_BOTTOM = 0.01, 0.05, 0.03

            def _fallback_via_priors():
                nonlocal px, py
                best_i, best_score = None, -1e9
                best_area = -1.0
                for i, ci in enumerate(cls_ids):
                    name = _name(ci)
                    if name in NEG:
                        base = WEIGHTS["negative"]
                    elif name in supports:
                        base = WEIGHTS["supports"]
                    elif name in cooccurs:
                        base = WEIGHTS["cooccurs"]
                    elif name in gateways:
                        base = WEIGHTS["gateways"]
                    else:
                        base = 0.0

                    conf = float(box_conf[i]) if box_conf is not None else 1.0
                    score = base + ALPHA_CONF * conf + BETA_AREA * _norm_area(i) + GAMMA_BOTTOM * _bottom_bias(i)

                    a = _area(i)
                    if (score > best_score) or (np.isclose(score, best_score) and a > best_area):
                        best_i, best_score, best_area = i, score, a

                if best_i is not None and best_score > WEIGHTS["negative"]:
                    px, py = _center(best_i); return

                floor_set = set(map(str.lower, floor_aliases))
                floor_inds = [i for i, ci in enumerate(cls_ids) if _name(ci) in floor_set]
                if len(floor_inds) > 0:
                    areas = [_area(i) for i in floor_inds]
                    top = floor_inds[int(np.argmax(areas))]
                    px, py = _center(top); return

            # ★ 타깃 확정: conf 컷오프 + lookalikes IoU 검사
            MIN_TARGET_CONF = 0.3
            LA_IOU_THRES    = 0.99
            used_target = False

            if goal_idx >= 0 and np.any(cls_ids == goal_idx):
                target_inds = np.where(cls_ids == goal_idx)[0]

                if box_conf is None:
                    target_inds = np.array([], dtype=int)
                else:
                    ok_mask = box_conf[target_inds] >= MIN_TARGET_CONF
                    target_inds = target_inds[ok_mask]

                if len(target_inds) > 0:
                    top = _pick_largest(target_inds)
                    look_inds = [k for k, ci in enumerate(cls_ids) if _name(ci) in NEG]

                    if top is not None and any(_iou(top, lk) >= LA_IOU_THRES for lk in look_inds):
                        _fallback_via_priors()
                    else:
                        if top is not None:
                            px, py = _center(top)
                            pri_flag = True
                            used_target = True

            if not used_target and not pri_flag:
                _fallback_via_priors()

        # 정책용 마스크
        r = 8
        x1, x2 = max(0, px - r), min(W, px + r)
        y1, y2 = max(0, py - r), min(H, py + r)
        debug_mask[y1:y2, x1:x2] = 255

        cv2.drawMarker(
            vis_bgr, (px, py),
            (0, 255, 255) if not pri_flag else (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2
        )

        # ★ 최근 박스 저장(원하면 비교용으로 활용)
        self._last_bboxes = boxes_std

        # ★ 호환성 유지: 기본은 4-튜플, 필요 시 5번째로 boxes_std 반환
        if return_boxes:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), debug_mask, pri_flag, vis_bgr, boxes_std
        else:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), debug_mask, pri_flag, vis_bgr

    
    def obs_goal_object(self,
                        image: np.ndarray,
                        conf_threshold: float = 0.30,
                        iou_threshold: float = 0.50) -> bool:
        """
        현재 object_goal의 존재 여부만 간단히 판정.
        - YOLOE box_threshold=conf_threshold로 필터링된 결과에서
          목표 클래스가 1개 이상 있으면 True, 없으면 False.
        - IoU 기반 추가 판정은 하지 않음(간단/빠른 체크).
        """
        # 클래스셋 보장(priors 반영)
        prompt_classes = getattr(self, "_prompt_classes", None)
        if not prompt_classes:
            self._set_prompt_from_priors(self.latest_priors or {})
            prompt_classes = self._prompt_classes

        goal_name = getattr(self, "object_goal", "") or ""
        try:
            goal_idx = prompt_classes.index(goal_name)
        except ValueError:
            # 현재 프롬프트 클래스에 목표가 없다면 탐지 불가
            return False

        # YOLOE 탐지 (세그 OFF, NMS는 내부에서 iou_threshold 사용)
        det = yoloe_detection(
            image=image,
            target_classes=prompt_classes,
            model=self.yoloe_model,
            box_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            run_extra_nms=False,
            use_text_prompt=False,
            retina_masks=False,
        )

        cls_ids = getattr(det, "class_id", None)
        if cls_ids is None or len(cls_ids) == 0:
            return False

        cls_ids = np.asarray(cls_ids).astype(int)
        # box_threshold에서 이미 컷이 되었으므로 신뢰도 배열이 없어도 OK
        return bool(np.any(cls_ids == goal_idx))

    def are_bboxes_similar(
        self,
        prev_boxes: List[Dict[str, Any]],
        curr_boxes: List[Dict[str, Any]],
        *,
        iou_thresh: float = 0.85,
        center_tol: float = 0.01,
        area_tol: float = 0.10,
        min_match_ratio: float = 0.95,
        class_sensitive: bool = True,
        ignore_classes: Optional[List[str]] = None, 
        max_count_delta: int = 0,
        return_detail: bool = False,
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]: 
        

        ignore_set = set([c.lower() for c in (ignore_classes or [])])
        MIN_CONF = 0.0  # 필요 시 0.2~0.3으로 올리세요.

        def _ok(b):
            # ★ 버그 수정: cx만 두 번 보던 문제 → cx, cy, area 모두 확인
            if not (b and ("bbox_xyxy" in b) and ("norm" in b)):
                return False
            n = b["norm"]
            return all(k in n for k in ("cx", "cy", "area"))

        def _cls_name(b):
            # ★ 클래스 이름 없으면 cls_id로 폴백
            c = b.get("cls")
            if c is None:
                cid = b.get("cls_id")
                c = str(cid) if cid is not None else None
            return str(c).lower() if c is not None else None

        def _iou_xyxy(a, b) -> float:
            ax1, ay1, ax2, ay2 = map(float, a["bbox_xyxy"])
            bx1, by1, bx2, by2 = map(float, b["bbox_xyxy"])
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            union = area_a + area_b - inter + 1e-9
            return inter / union

        def _center_dist(a, b) -> float:
            dx = float(a["norm"]["cx"]) - float(b["norm"]["cx"])
            dy = float(a["norm"]["cy"]) - float(b["norm"]["cy"])
            return math.hypot(dx, dy)

        def _area_change(a, b) -> float:
            aa = max(1e-9, float(a["norm"]["area"]))
            bb = max(1e-9, float(b["norm"]["area"]))
            return abs(aa - bb) / max(aa, bb)

        # --- 유효 / 필터링 ---
        def _filter(v):
            out = []
            for b in (v or []):
                if not _ok(b):
                    continue
                if _cls_name(b) in ignore_set:
                    continue
                sc = b.get("score", None)
                if (sc is not None) and (sc < MIN_CONF):
                    continue
                out.append(b)
            return out

        prev = _filter(prev_boxes)
        curr = _filter(curr_boxes)

        n_prev, n_curr = len(prev), len(curr)
        detail = {"n_prev": n_prev, "n_curr": n_curr, "matched": 0, "pairs": [], "reason": ""}

        if n_prev == 0 and n_curr == 0:
            return (True, {**detail, "reason": "both empty"}) if return_detail else True
        if n_prev == 0 or n_curr == 0:
            return (False, {**detail, "reason": "one side empty"}) if return_detail else False
        if abs(n_prev - n_curr) > max_count_delta:
            return (False, {**detail, "reason": "count delta too large"}) if return_detail else False

        # --- 후보 페어 만들기: 싼 검사(center, area) 먼저 → 통과 시 IoU 계산 ---
        candidates = []
        for i, pb in enumerate(prev):
            pn = _cls_name(pb)
            for j, cb in enumerate(curr):
                if class_sensitive:
                    cn = _cls_name(cb)
                    if (pn is None) or (cn is None) or (pn != cn):
                        continue
                cdist = _center_dist(pb, cb)
                if cdist > center_tol:
                    continue
                adiff = _area_change(pb, cb)
                if adiff > area_tol:
                    continue
                iou = _iou_xyxy(pb, cb)
                if iou >= iou_thresh:
                    candidates.append((iou, i, j, cdist, adiff))

        # --- 그리디 매칭 (IoU 내림차순) ---
        candidates.sort(key=lambda x: -x[0])
        used_prev, used_curr = set(), set()
        for iou, i, j, cdist, adiff in candidates:
            if i in used_prev or j in used_curr:
                continue
            used_prev.add(i); used_curr.add(j)
            detail["pairs"].append((i, j, float(iou), float(cdist), float(adiff)))

        matched = len(detail["pairs"])
        detail["matched"] = matched
        denom = max(n_prev, n_curr)
        ratio = matched / max(1, denom)
        similar = ratio >= min_match_ratio
        detail["match_ratio"] = float(ratio)
        detail["reason"] = "ok" if similar else "low match ratio"

        return (similar, detail) if return_detail else similar
