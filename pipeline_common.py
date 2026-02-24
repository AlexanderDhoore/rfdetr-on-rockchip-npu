from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import supervision as sv
from PIL import Image


def load_coco_classes() -> dict[int, str]:
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES  # type: ignore

        return dict(COCO_CLASSES)
    except Exception:
        return {}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x_c = boxes[..., 0]
    y_c = boxes[..., 1]
    w = np.clip(boxes[..., 2], a_min=0.0, a_max=None)
    h = np.clip(boxes[..., 3], a_min=0.0, a_max=None)
    return np.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), axis=-1)


def preprocess_image(image_path: str, input_h: int, input_w: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized = image.resize((input_w, input_h))
    image_array = np.array(resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std

    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    original_scene = np.array(image).copy()
    return image_array, original_scene, orig_h, orig_w


def make_ort_session(model_path: str, intra_threads: int = 4, inter_threads: int = 1) -> ort.InferenceSession:
    if intra_threads <= 0:
        raise ValueError("intra_threads must be > 0")
    if inter_threads <= 0:
        raise ValueError("inter_threads must be > 0")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def run_model_single_input(session: ort.InferenceSession, input_tensor: np.ndarray) -> list[np.ndarray]:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})


def run_model_multi_input(session: ort.InferenceSession, input_tensors: list[np.ndarray]) -> list[np.ndarray]:
    inputs = session.get_inputs()
    if len(input_tensors) != len(inputs):
        raise ValueError(
            f"Input count mismatch: model expects {len(inputs)} tensors but got {len(input_tensors)}."
        )
    feed = {inputs[i].name: input_tensors[i] for i in range(len(inputs))}
    return session.run(None, feed)


def run_full_onnx(
    session: ort.InferenceSession, input_tensor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    outputs = run_model_single_input(session, input_tensor)
    if len(outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from full ONNX model, got {len(outputs)}.")
    return outputs[0], outputs[1], outputs[2] if len(outputs) >= 3 else None


def run_split_onnx(
    backbone_session: ort.InferenceSession,
    detector_session: ort.InferenceSession,
    input_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    backbone_outputs = run_model_single_input(backbone_session, input_tensor)
    detector_input_count = len(detector_session.get_inputs())
    if detector_input_count > len(backbone_outputs):
        raise ValueError(
            f"Detector expects {detector_input_count} inputs but backbone produced {len(backbone_outputs)} outputs."
        )

    detector_outputs = run_model_multi_input(detector_session, backbone_outputs[:detector_input_count])
    if len(detector_outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from detector ONNX model, got {len(detector_outputs)}.")
    return detector_outputs[0], detector_outputs[1], detector_outputs[2] if len(detector_outputs) >= 3 else None


def postprocess_onnx(
    dets: np.ndarray,
    logits: np.ndarray,
    orig_h: int,
    orig_w: int,
    threshold: float,
    num_select: int,
    masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    prob = sigmoid(logits)
    batch_size, _, num_classes = prob.shape
    flat_prob = prob.reshape(batch_size, -1)

    k = min(num_select, flat_prob.shape[1])
    topk_idx_unsorted = np.argpartition(flat_prob, -k, axis=1)[:, -k:]
    topk_scores_unsorted = np.take_along_axis(flat_prob, topk_idx_unsorted, axis=1)
    topk_order = np.argsort(-topk_scores_unsorted, axis=1)

    topk_indexes = np.take_along_axis(topk_idx_unsorted, topk_order, axis=1)
    scores = np.take_along_axis(topk_scores_unsorted, topk_order, axis=1)
    topk_boxes = topk_indexes // num_classes
    labels = topk_indexes % num_classes

    boxes_xyxy = box_cxcywh_to_xyxy(dets)
    batch_indices = np.arange(batch_size)[:, None]
    boxes = boxes_xyxy[batch_indices, topk_boxes]
    boxes *= np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)[None, None, :]

    keep = scores[0] > threshold
    selected_boxes = boxes[0][keep]
    selected_scores = scores[0][keep]
    selected_labels = labels[0][keep].astype(np.int32)

    selected_masks: np.ndarray | None = None
    if masks is not None:
        if masks.ndim != 4:
            raise ValueError(f"Expected masks shape [B, Q, Hm, Wm], got {masks.shape}")

        selected_query_masks = masks[np.arange(batch_size)[:, None], topk_boxes][0][keep]
        resized_masks = []
        bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        for mask in selected_query_masks:
            mask_resized = np.array(
                Image.fromarray(mask.astype(np.float32), mode="F").resize((orig_w, orig_h), resample=bilinear),
                dtype=np.float32,
            )
            resized_masks.append(mask_resized > 0.0)
        if resized_masks:
            selected_masks = np.stack(resized_masks, axis=0)
        else:
            selected_masks = np.zeros((0, orig_h, orig_w), dtype=bool)

    return selected_boxes, selected_scores, selected_labels, selected_masks


def print_raw_diff(
    tag: str,
    dets_a: np.ndarray,
    logits_a: np.ndarray,
    dets_b: np.ndarray,
    logits_b: np.ndarray,
    masks_a: np.ndarray | None = None,
    masks_b: np.ndarray | None = None,
) -> tuple[float, float, float | None]:
    dets_abs_diff = np.abs(dets_a - dets_b)
    logits_abs_diff = np.abs(logits_a - logits_b)
    dets_max = float(dets_abs_diff.max())
    logits_max = float(logits_abs_diff.max())
    print(tag)
    print(f"dets max_abs_diff:   {dets_max:.8f}")
    print(f"dets mean_abs_diff:  {dets_abs_diff.mean():.8f}")
    print(f"labels max_abs_diff: {logits_max:.8f}")
    print(f"labels mean_abs_diff:{logits_abs_diff.mean():.8f}")

    masks_max: float | None = None
    if masks_a is not None and masks_b is not None:
        masks_abs_diff = np.abs(masks_a - masks_b)
        masks_max = float(masks_abs_diff.max())
        print(f"masks max_abs_diff:  {masks_max:.8f}")
        print(f"masks mean_abs_diff: {masks_abs_diff.mean():.8f}")
    elif masks_a is not None or masks_b is not None:
        raise ValueError("One pipeline has masks while the other does not.")

    return dets_max, logits_max, masks_max


def print_post_diff(
    tag: str,
    boxes_a: np.ndarray,
    conf_a: np.ndarray,
    cls_a: np.ndarray,
    boxes_b: np.ndarray,
    conf_b: np.ndarray,
    cls_b: np.ndarray,
    masks_a: np.ndarray | None = None,
    masks_b: np.ndarray | None = None,
) -> None:
    print(tag)
    print(f"a detections: {len(cls_a)}")
    print(f"b detections: {len(cls_b)}")
    if len(cls_a) == len(cls_b) and len(cls_a) > 0:
        boxes_diff = np.abs(boxes_a - boxes_b)
        conf_diff = np.abs(conf_a - conf_b)
        same_classes = np.array_equal(cls_a, cls_b)
        print(f"post boxes max_abs_diff:   {boxes_diff.max():.8f}")
        print(f"post scores max_abs_diff:  {conf_diff.max():.8f}")
        print(f"post classes identical:    {same_classes}")
        if masks_a is not None and masks_b is not None:
            same_masks = np.array_equal(masks_a, masks_b)
            mask_diff_ratio = float(np.mean(masks_a != masks_b))
            print(f"post masks identical:      {same_masks}")
            print(f"post masks diff_ratio:     {mask_diff_ratio:.8f}")
        elif masks_a is not None or masks_b is not None:
            raise ValueError("One pipeline has masks while the other does not.")
    elif len(cls_a) == 0 and len(cls_b) == 0:
        print("Both pipelines produced zero detections above threshold.")
    else:
        print("Detection counts differ after thresholding.")


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return inter / (area_a + area_b - inter + 1e-9)


def print_post_diff_matched(
    tag: str,
    boxes_a: np.ndarray,
    conf_a: np.ndarray,
    cls_a: np.ndarray,
    boxes_b: np.ndarray,
    conf_b: np.ndarray,
    cls_b: np.ndarray,
    masks_a: np.ndarray | None = None,
    masks_b: np.ndarray | None = None,
) -> None:
    print(tag)
    print(f"a detections: {len(cls_a)}")
    print(f"b detections: {len(cls_b)}")

    if len(cls_a) == 0 and len(cls_b) == 0:
        print("Both pipelines produced zero detections above threshold.")
        return

    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    ious: list[float] = []

    for i in range(len(cls_a)):
        best_j = -1
        best_iou = -1.0
        for j in range(len(cls_b)):
            if j in used_b:
                continue
            if int(cls_b[j]) != int(cls_a[i]):
                continue
            cur_iou = _box_iou_xyxy(boxes_a[i], boxes_b[j])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
            pairs.append((i, best_j))
            ious.append(best_iou)

    print(f"matched pairs (class-aware): {len(pairs)}")
    print(f"unmatched a: {len(cls_a) - len(pairs)}")
    print(f"unmatched b: {len(cls_b) - len(pairs)}")

    if len(pairs) == 0:
        print("No class-consistent matches found.")
        return

    idx_a = np.array([p[0] for p in pairs], dtype=np.int64)
    idx_b = np.array([p[1] for p in pairs], dtype=np.int64)

    matched_boxes_a = boxes_a[idx_a]
    matched_boxes_b = boxes_b[idx_b]
    matched_conf_a = conf_a[idx_a]
    matched_conf_b = conf_b[idx_b]
    matched_cls_a = cls_a[idx_a]
    matched_cls_b = cls_b[idx_b]

    boxes_diff = np.abs(matched_boxes_a - matched_boxes_b)
    conf_diff = np.abs(matched_conf_a - matched_conf_b)
    print(f"matched boxes max_abs_diff:  {boxes_diff.max():.8f}")
    print(f"matched scores max_abs_diff: {conf_diff.max():.8f}")
    print(f"matched classes identical:   {bool(np.array_equal(matched_cls_a, matched_cls_b))}")
    print(f"matched IoU mean:            {float(np.mean(ious)):.8f}")
    print(f"matched IoU min:             {float(np.min(ious)):.8f}")

    if masks_a is not None and masks_b is not None:
        matched_masks_a = masks_a[idx_a]
        matched_masks_b = masks_b[idx_b]
        mask_diff_ratio = float(np.mean(matched_masks_a != matched_masks_b))
        print(f"matched masks diff_ratio:    {mask_diff_ratio:.8f}")
    elif masks_a is not None or masks_b is not None:
        raise ValueError("One pipeline has masks while the other does not.")


def annotate_and_save(
    scene: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    output_path: str,
    coco_classes: dict[int, str] | None = None,
    masks: np.ndarray | None = None,
) -> None:
    if masks is not None:
        dets = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=confidences,
            class_id=class_ids.astype(np.int32),
            mask=masks.astype(bool),
        )
    else:
        dets = sv.Detections(xyxy=boxes_xyxy, confidence=confidences, class_id=class_ids.astype(np.int32))
    coco_classes = coco_classes or {}
    labels = []
    for class_id, conf in zip(dets.class_id, dets.confidence):
        name = coco_classes.get(int(class_id), f"class_{int(class_id)}")
        labels.append(f"{name} {float(conf):.2f}")

    if masks is not None:
        annotated = sv.MaskAnnotator().annotate(scene=scene.copy(), detections=dets)
        annotated = sv.BoxAnnotator().annotate(scene=annotated, detections=dets)
    else:
        annotated = sv.BoxAnnotator().annotate(scene=scene.copy(), detections=dets)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=dets, labels=labels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(output_path)


def ensure_shape_match(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for {name}: {a.shape} vs {b.shape}")


def ensure_files_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
