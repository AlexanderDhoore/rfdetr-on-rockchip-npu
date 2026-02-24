import argparse
from pathlib import Path

import numpy as np

from pipeline_common import (
    annotate_and_save,
    ensure_files_exist,
    ensure_shape_match,
    load_coco_classes,
    make_ort_session,
    postprocess_onnx,
    preprocess_image,
    print_post_diff,
    print_raw_diff,
    run_full_onnx,
    run_split_onnx,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full ONNX vs split ONNX outputs (expected exact match).")
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument("--output-dir", default="output", help="Directory with ONNX files.")
    parser.add_argument(
        "--detector-onnx",
        default="detection_fp32.onnx",
        help="Detector ONNX filename inside output-dir (default keeps exact full-vs-split check).",
    )
    parser.add_argument("--output-image", default="bus_onnx.jpg", help="Annotated output image path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--num-select", type=int, default=300, help="Top-K predictions before thresholding.")
    parser.add_argument(
        "--allow-nonzero-diff",
        action="store_true",
        help="Do not fail when raw full-vs-split diffs are non-zero.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    full_onnx = output_dir / "inference_model.onnx"
    backbone_onnx = output_dir / "backbone.onnx"
    detector_onnx = output_dir / args.detector_onnx

    ensure_files_exist([Path(args.image), full_onnx, backbone_onnx, detector_onnx])
    Path(args.output_image).unlink(missing_ok=True)

    full_session = make_ort_session(str(full_onnx))
    backbone_session = make_ort_session(str(backbone_onnx))
    detector_session = make_ort_session(str(detector_onnx))

    full_input = full_session.get_inputs()[0]
    backbone_input = backbone_session.get_inputs()[0]

    if full_input.shape[2:] != backbone_input.shape[2:]:
        raise ValueError(
            f"Input size mismatch: full={full_input.shape} backbone={backbone_input.shape}. "
            "Export both models at the same resolution."
        )

    input_h = int(full_input.shape[2])
    input_w = int(full_input.shape[3])
    input_tensor, original_scene, orig_h, orig_w = preprocess_image(args.image, input_h, input_w)

    dets_full, logits_full, masks_full = run_full_onnx(full_session, input_tensor)
    dets_split, logits_split, masks_split = run_split_onnx(backbone_session, detector_session, input_tensor)

    ensure_shape_match("dets", dets_full, dets_split)
    ensure_shape_match("labels", logits_full, logits_split)
    if masks_full is not None and masks_split is not None:
        ensure_shape_match("masks", masks_full, masks_split)
    elif masks_full is not None or masks_split is not None:
        raise ValueError("One ONNX pipeline returned masks while the other did not.")

    dets_max, logits_max, masks_max = print_raw_diff(
        "Raw tensor sanity check (full ONNX vs split ONNX)",
        dets_full,
        logits_full,
        dets_split,
        logits_split,
        masks_full,
        masks_split,
    )

    full_xyxy, full_conf, full_cls, full_masks = postprocess_onnx(
        dets=dets_full,
        logits=logits_full,
        orig_h=orig_h,
        orig_w=orig_w,
        threshold=args.threshold,
        num_select=args.num_select,
        masks=masks_full,
    )
    split_xyxy, split_conf, split_cls, split_masks = postprocess_onnx(
        dets=dets_split,
        logits=logits_split,
        orig_h=orig_h,
        orig_w=orig_w,
        threshold=args.threshold,
        num_select=args.num_select,
        masks=masks_split,
    )

    print_post_diff(
        "Postprocess sanity check (full ONNX vs split ONNX)",
        full_xyxy,
        full_conf,
        full_cls,
        split_xyxy,
        split_conf,
        split_cls,
        full_masks,
        split_masks,
    )

    annotate_and_save(
        scene=original_scene,
        boxes_xyxy=split_xyxy,
        confidences=split_conf,
        class_ids=split_cls,
        output_path=args.output_image,
        coco_classes=load_coco_classes(),
        masks=split_masks,
    )
    print(f"Saved {args.output_image} with {len(split_cls)} detections above threshold {args.threshold}.")

    raw_mismatch = dets_max != 0.0 or logits_max != 0.0 or (masks_max is not None and masks_max != 0.0)
    if not args.allow_nonzero_diff and raw_mismatch:
        raise RuntimeError(
            "Full ONNX and split ONNX outputs are not exactly equal. "
            f"dets_max={dets_max:.8f}, logits_max={logits_max:.8f}, "
            f"masks_max={masks_max if masks_max is not None else 'n/a'}"
        )


if __name__ == "__main__":
    main()
