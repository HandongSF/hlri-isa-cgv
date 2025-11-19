# llm_utils/priors_parser.py
import json
import ast
import re
from typing import Any, Dict, Iterable, List, Optional

# 정규식
_SPLIT_RE   = re.compile(r"[,\;/]+")
_WS_RE      = re.compile(r"\s+")
_BRACED_RE  = re.compile(r"\{.*\}", flags=re.DOTALL)

# 형식 규칙(개수 제한 없음; 형식만 체크)
MAX_ITEM_WORDS   = 3
REQ_GATEWAYS     = ["gateway", "entrance", "opened door"]  # 항상 포함 보장

# ----------------- 유틸 -----------------
def _lower_list(x: Any) -> List[str]:
    """문자열/리스트/튜플을 소문자 리스트로 정규화. 기타 타입은 빈 리스트."""
    if x is None:
        return []
    if isinstance(x, str):
        parts = [p.strip().lower() for p in _SPLIT_RE.split(x) if p.strip()]
        return parts
    if isinstance(x, (list, tuple)):
        return [str(p).strip().lower() for p in x if str(p).strip()]
    return []

def _dedup(seq: Iterable[str]) -> List[str]:
    """순서를 유지하는 중복 제거."""
    return list(dict.fromkeys(seq))

def _find_braced(text: str) -> Optional[str]:
    """원시 텍스트에서 가장 그럴싸한 {..} 블록 하나를 추출."""
    if not text:
        return None
    cand = _BRACED_RE.findall(text)
    if not cand:
        return None
    cand.sort(key=len, reverse=True)  # 가장 긴 것을 우선
    return cand[0]

def _to_dict(s: str) -> Optional[dict]:
    """JSON → 실패 시 Python literal(dict) 파싱."""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass
    return None

def _pick_first(d: dict, keys: List[str], default=None):
    """여러 후보 키 중 먼저 발견되는 값을 반환(대소문자 관대)."""
    for k in keys:
        if k in d:
            return d[k]
        for kk in d.keys():
            if kk.lower() == k.lower():
                return d[kk]
    return default

def _word_count(s: str) -> int:
    return len([w for w in _WS_RE.split(s.strip()) if w]) if isinstance(s, str) else 0

def _filter_item_len(items: List[str], violations: List[str], field_name: str) -> List[str]:
    """
    리스트 아이템 형식 제한 적용:
      - 단어 수 <= 3
      - 숫자 포함 금지
    개수 제한은 적용하지 않음.
    """
    out = []
    for it in items:
        if not it:
            continue
        if _word_count(it) > MAX_ITEM_WORDS:
            violations.append(f"{field_name}:drop_long_item('{it}')")
            continue
        if any(ch.isdigit() for ch in it):
            violations.append(f"{field_name}:drop_numeric_item('{it}')")
            continue
        out.append(it)
    return out

def _ensure_gateways_required(gws: List[str]) -> List[str]:
    """gateways에 gateway가 항상 포함되도록 보강(중복 제거, 개수 제한 없음)."""
    s = set(gws)
    add = [g for g in REQ_GATEWAYS if g not in s]
    return _dedup(add + gws)

# ----------------- PRIORS 전용 파서 -----------------
def parse_priors_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    LLM 원시 텍스트에서 PRIORS만 복구한다.
    반환 필드:
      - Supports, StrongCooccurs, Gateways, Lookalikes: List[str]
      - violations: List[str]  # 규칙 위반/수정 사항 기록
      - raw_blob: str          # 추출된 원시 블록(디버깅용)
    Angle/Flag/Reason 등은 전혀 다루지 않는다.
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

    violations: List[str] = []

    # --- PRIORS 동의어 처리 ---
    supports   = _lower_list(_pick_first(d, ["Supports", "SupportObjects", "support", "support_objects"]))
    cooccurs   = _lower_list(_pick_first(d, ["StrongCooccurs", "Cooccurs", "RelatedObjects", "related_objects"]))
    gateways   = _lower_list(_pick_first(d, ["Gateways", "Waypoints", "SearchTargets", "navigation_cues"]))
    lookalikes = _lower_list(_pick_first(d, ["Lookalikes", "Negatives", "Distractors", "Confusables"]))

    # 구 프롬프트 호환 합류
    cooccurs += _lower_list(_pick_first(d, ["RelatedObjects", "related"]))
    gateways += _lower_list(_pick_first(d, ["SearchTargets", "search_targets"]))

    # 형식 필터 + dedup (개수 제한 없음)
    supports   = _dedup(_filter_item_len(supports, violations, "supports"))
    cooccurs   = _dedup(_filter_item_len(cooccurs, violations, "cooccurs"))
    gateways   = _dedup(_filter_item_len(gateways, violations, "gateways"))
    lookalikes = _dedup(_filter_item_len(lookalikes, violations, "lookalikes"))

    # 필수 게이트웨이 보장
    gateways = _ensure_gateways_required(gateways)

    return {
        "Supports": supports,
        "StrongCooccurs": cooccurs,
        "Gateways": gateways,
        "Lookalikes": lookalikes,
        "violations": violations,
        "raw_blob": blob,
    }

# 하위 호환: 기존 이름으로도 사용할 수 있도록 alias
parse_llm_json = parse_priors_json

def extract_priors(answer: Dict[str, Any], raw_text: Optional[str] = None) -> Dict[str, Any]:
    """
    PRIORS dict만 깔끔히 반환.
    (SceneToObjectHints는 현재 스키마에 없으므로 포함하지 않음)
    """
    if not isinstance(answer, dict):
        return {
            "Supports": [],
            "StrongCooccurs": [],
            "Gateways": [],
            "Lookalikes": [],
        }
    return {
        "Supports": list(map(str, answer.get("Supports", []))),
        "StrongCooccurs": list(map(str, answer.get("StrongCooccurs", []))),
        "Gateways": list(map(str, answer.get("Gateways", []))),
        "Lookalikes": list(map(str, answer.get("Lookalikes", []))),
    }

# ======== Decision(Reason/Angle/Flag) 전용 파서 ========
# 허용 각도 기본셋(12방위)
ANGLE_SET_12 = {i * 30 for i in range(12)}  # {0,30,...,330}
DECISION_MAX_REASON_WORDS = 30

def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true", "yes", "y", "1"}:
            return True
        if t in {"false", "no", "n", "0"}:
            return False
    return bool(x)

def _norm_angle(v: Any, valid_angles: Optional[Iterable[int]] = None) -> Optional[int]:
    """정수 변환 + 허용 집합 검사."""
    try:
        a = int(v)
    except Exception:
        return None
    s = set(valid_angles) if valid_angles is not None else ANGLE_SET_12
    return a if a in s else None

def _trim_reason(reason: str) -> str:
    """Reason을 30 단어로 트림."""
    if not isinstance(reason, str):
        return ""
    words = [w for w in _WS_RE.split(reason.strip()) if w]
    if len(words) <= DECISION_MAX_REASON_WORDS:
        return " ".join(words)
    return " ".join(words[:DECISION_MAX_REASON_WORDS]) + " …"

def parse_decision_json(raw_text: str, valid_angles: Optional[Iterable[int]] = None) -> Optional[Dict[str, Any]]:
    """
    LLM 원문에서 Reason/Angle/Flag만 복구한다.
    반환:
      {
        "Angle": int (valid_angles 또는 {0,30,...,330} 중 하나),
        "Flag": bool,
        "Reason": str (최대 30단어로 트림),
        "violations": List[str],
        "raw_blob": str
      }
    실패 시 None
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

    violations: List[str] = []

    # --- Angle ---
    angle_raw = _pick_first(d, ["Angle", "angle"])
    angle = _norm_angle(angle_raw, valid_angles)
    if angle is None:
        return None

    # --- Flag ---
    flag_raw = _pick_first(d, ["Flag", "flag", "Found", "found", "Goal", "goal"], False)
    flag = _to_bool(flag_raw)

    # --- Reason ---
    reason_raw = _pick_first(d, ["Reason", "reason", "Rationale", "rationale"], "")
    reason = _trim_reason(str(reason_raw))
    if _word_count(str(reason_raw)) > DECISION_MAX_REASON_WORDS:
        violations.append(f"reason_trimmed_to_{DECISION_MAX_REASON_WORDS}_words")

    return {
        "Angle": int(angle),
        "Flag": flag,
        "Reason": reason,
        "violations": violations,
        "raw_blob": blob,
    }

def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """
    리스트에서 중복 항목 제거(순서 보존).
    비교 키는 str.strip().lower()를 사용하되, 반환은 원본 표기를 보존한다.
    """
    seen = set()
    out: List[str] = []
    for x in items or []:
        k = x.strip().lower() if isinstance(x, str) else x
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def parse_prior_class_block(block: str) -> List[str]:
    """
    PRIOR_CLASS_LIST 블록에서 'VALID_CLASSES:' 아래 라인들을 파싱해
    소문자 문자열 리스트로 반환(빈 줄 제외, 순서 보존·중복 제거).
    """
    if not isinstance(block, str):
        return []
    lines = block.splitlines()
    started = False
    raw: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        if s.lower().startswith("valid_classes"):
            started = True
            continue
        if started:
            raw.append(s.lower())
    return dedupe_preserve_order(raw)
