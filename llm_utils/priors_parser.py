# llm_utils/priors_parser.py
import json
import ast
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

ANGLE_SET_12 = {i * 30 for i in range(12)}  # {0,30,...,330}

_SPLIT_RE = re.compile(r"[,\;/]+")

def _lower_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        parts = [p.strip().lower() for p in _SPLIT_RE.split(x) if p.strip()]
        return parts
    if isinstance(x, (list, tuple)):
        return [str(p).strip().lower() for p in x if str(p).strip()]
    # dict나 기타는 무시
    return []

def _dedup(seq: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def _norm_angle(v: Any, valid_angles: Optional[Iterable[int]] = None) -> Optional[int]:
    try:
        a = int(v)
    except Exception:
        return None
    s = set(valid_angles) if valid_angles is not None else ANGLE_SET_12
    return a if a in s else None

def _find_braced(text: str) -> Optional[str]:
    """가장 그럴싸한 { .. } 블록 1개를 뽑아온다 (JSON/파이썬 dict 모두 커버)."""
    if not text:
        return None
    # 가장 긴 중괄호 블록을 고른다
    cand = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not cand:
        return None
    # 보통 마지막/가장 긴 게 유효
    cand.sort(key=len, reverse=True)
    return cand[0]

def _to_dict(s: str) -> Optional[dict]:
    # 1) JSON 시도
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) 파이썬 literal dict 시도
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
        # 일부 모델은 {"Answer": {...}} 같은 래핑을 돌려주기도 함
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass
    return None

def _pick_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
        # 대소문자 다양성 처리
        for kk in d.keys():
            if kk.lower() == k.lower():
                return d[kk]
    return default

def parse_llm_json(raw_text: str, valid_angles: Optional[Iterable[int]] = None) -> Optional[Dict[str, Any]]:
    """
    LLM 원시 텍스트에서 유효한 딕셔너리를 복구하고 표준 필드로 정리한다.
    표준 필드:
      - Angle (int in valid_angles or {0,30,...,330})
      - Flag (bool, optional)
      - Reason (str, optional)
      - Supports, StrongCooccurs, Gateways, Lookalikes: List[str]
      - SceneToObjectHints: Dict[str, List[str]]
    """
    blob = _find_braced(raw_text)
    if not blob:
        return None

    d = _to_dict(blob)
    if not isinstance(d, dict):
        return None

    # {"Answer": {...}} 형태 처리
    if "Answer" in d and isinstance(d["Answer"], dict):
        d = d["Answer"]

    # 각 필드 추출
    angle = _norm_angle(_pick_first(d, ["Angle", "angle"]), valid_angles)
    if angle is None:
        return None

    flag = _pick_first(d, ["Flag", "flag", "Found", "found", "Goal", "goal"], False)
    flag = bool(flag)

    reason = _pick_first(d, ["Reason", "reason", "Rationale", "rationale"], "")

    # --- PRIORS 동의어 처리 ---
    # 메인 스키마
    supports = _lower_list(_pick_first(d, ["Supports", "SupportObjects", "support", "support_objects"]))
    cooccurs = _lower_list(_pick_first(d, ["StrongCooccurs", "Cooccurs", "RelatedObjects", "related_objects"]))
    gateways = _lower_list(_pick_first(d, ["Gateways", "Waypoints", "SearchTargets", "navigation_cues"]))
    lookalikes = _lower_list(_pick_first(d, ["Lookalikes", "Negatives", "Distractors", "Confusables"]))

    # scene→object 힌트는 dict 또는 list 가능
    hints = _pick_first(d, ["SceneToObjectHints", "scene_hints", "RoomHints", "sceneToObject"])
    scene_hints: Dict[str, List[str]] = {}
    if isinstance(hints, dict):
        for k, v in hints.items():
            scene_hints[str(k).strip().lower()] = _dedup(_lower_list(v))
    elif isinstance(hints, (list, tuple)):
        # 리스트만 온 경우: generic 키로 묶어 둔다
        scene_hints["generic"] = _dedup(_lower_list(hints))
    elif isinstance(hints, str):
        scene_hints["generic"] = _dedup(_lower_list(hints))

    # 구(舊) 프롬프트 호환 (있으면 보조로 합치기)
    cooccurs += _lower_list(_pick_first(d, ["RelatedObjects", "related"]))
    gateways += _lower_list(_pick_first(d, ["SearchTargets", "search_targets"]))
    # AvoidZones 는 탐지 프롬프트로 쓰기엔 곤란하니 저장만 해두고 사용처에서 결정
    avoid_zones = _lower_list(_pick_first(d, ["AvoidZones", "avoid", "avoid_zones"]))
    if avoid_zones:
        scene_hints["avoid"] = _dedup(avoid_zones)

    return {
        "Angle": int(angle),
        "Flag": flag,
        "Reason": str(reason),
        "Supports": _dedup(supports),
        "StrongCooccurs": _dedup(cooccurs),
        "Gateways": _dedup(gateways),
        "Lookalikes": _dedup(lookalikes),
        "SceneToObjectHints": scene_hints,
        "raw_blob": blob,   # 필요 시 추적
    }

def extract_priors(answer: Dict[str, Any], raw_text: Optional[str] = None) -> Dict[str, Any]:
    """
    parse_llm_json 결과(dict)에서 표준 PRIORS 딕셔너리만 뽑아 반환.
    """
    if not isinstance(answer, dict):
        return {
            "Supports": [],
            "StrongCooccurs": [],
            "Gateways": [],
            "Lookalikes": [],
            "SceneToObjectHints": {},
        }
    return {
        "Supports": list(map(str, answer.get("Supports", []))),
        "StrongCooccurs": list(map(str, answer.get("StrongCooccurs", []))),
        "Gateways": list(map(str, answer.get("Gateways", []))),
        "Lookalikes": list(map(str, answer.get("Lookalikes", []))),
        "SceneToObjectHints": dict(answer.get("SceneToObjectHints", {})),
    }
