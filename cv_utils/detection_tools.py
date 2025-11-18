from groundingdino.util.inference import Model
from constants import *
import torchvision
import torch
import cv2
import numpy as np
from typing import Optional, List

def initialize_dino_model(dino_config=GROUNDING_DINO_CONFIG_PATH,
                          dino_checkpoint=GROUNDING_DINO_CHECKPOINT_PATH,
                          device="cuda:0"):
    model = Model(model_config_path=dino_config,
                  model_checkpoint_path=dino_checkpoint,
                  device=device)
    return model

def openset_detection(image,
                      target_classes,
                      dino_model,
                      box_threshold: float = 0.2,
                      text_threshold: float = 0.4,
                      nms_threshold: float = 0.5):
    """
    GroundingDINO openset detection.
    - image: RGB np.ndarray (H,W,3)
    - target_classes: List[str]
    - dino_model: groundingdino.util.inference.Model
    """
    detections = dino_model.predict_with_classes(
        image=image,
        classes=target_classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # class_id == None 필터링
    valid_mask = (detections.class_id != None)
    detections.xyxy       = detections.xyxy[valid_mask]
    detections.confidence = detections.confidence[valid_mask]
    detections.class_id   = detections.class_id[valid_mask]

    # NMS
    if len(detections.xyxy) > 0:
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()

        detections.xyxy       = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id   = detections.class_id[nms_idx]

    # YOLOE용 Detections처럼 class_names 달아주기
    detections.class_names = list(target_classes)
    return detections


def draw_detections_bgr(
    image_rgb: np.ndarray,
    det,
    goal_idx: int = -1,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """
    RGB 입력에 GroundingDINO detections를 그려서 BGR로 반환.
    - det.xyxy: (N,4)
    - det.class_id: (N,)
    - det.confidence: (N,)
    - det.class_names: List[str]  ← openset_detection에서 붙여둠
    goal_idx(=prompt_classes 인덱스)가 있으면 그 클래스는 초록색, 나머지는 파란색.
    """
    vis = image_rgb.copy()
    H, W = (vis.shape[0], vis.shape[1]) if vis.ndim >= 2 else (480, 640)
    if thickness is None:
        thickness = max(1, int(round(0.002 * (H + W))))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if getattr(det, "xyxy", None) is None or len(det.xyxy) == 0:
        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    class_names: Optional[List[str]] = getattr(det, "class_names", None)
    class_ids = getattr(det, "class_id", None)
    confidences = getattr(det, "confidence", None)

    for i, (x1, y1, x2, y2) in enumerate(det.xyxy):
        ci = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else -1

        if class_names is not None and 0 <= ci < len(class_names):
            name = str(class_names[ci])
        else:
            name = f"id:{ci}"

        conf = float(confidences[i]) if confidences is not None and i < len(confidences) else None

        is_goal = (goal_idx >= 0 and ci == goal_idx)
        color_box = (0, 255, 0) if is_goal else (255, 0, 0)  # BGR

        x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color_box, thickness)

        label = f"{name}" + (f" {conf:.2f}" if conf is not None else "")
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, thickness)
        cv2.rectangle(vis, (x1i, y1i - th - 6), (x1i + tw + 4, y1i), color_box, -1)
        cv2.putText(
            vis, label,
            (x1i + 2, y1i - 4),
            font, 0.5,
            (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

__all__ = [
    "initialize_dino_model",
    "openset_detection",
    "draw_detections_bgr",
]
