import argparse
from pathlib import Path
from rknn.api import RKNN

from pipeline_common import (
    annotate_and_save,
    ensure_files_exist,
    ensure_shape_match,
    load_coco_classes,
    make_ort_session,
    postprocess_onnx,
    preprocess_image,
    print_post_diff_matched,
    print_raw_diff,
    run_full_onnx,
    run_model_multi_input,
)


def run_backbone_rknn_sim(backbone_onnx: Path, input_tensor, target_platform: str):
    rknn = RKNN()
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        ret = rknn.load_onnx(model=str(backbone_onnx))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"rknn.init_runtime (simulator) failed with code {ret}")

        outputs = rknn.inference(inputs=[input_tensor], data_format=["nchw"])
        if outputs is None:
            raise RuntimeError("rknn.inference (simulator) returned None")
        return list(outputs)
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full ONNX vs RKNN(simulated backbone)+ONNX(detector).")
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument("--output-dir", default="output", help="Directory with ONNX/RKNN files.")
    parser.add_argument(
        "--detector-onnx",
        default="detection.onnx",
        help="Detector ONNX filename inside output-dir (default: dynamic-quantized detector).",
    )
    parser.add_argument("--output-image", default="bus_rknn.jpg", help="Annotated output image path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--num-select", type=int, default=300, help="Top-K predictions before thresholding.")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target platform.")
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

    rknn_outputs = run_backbone_rknn_sim(backbone_onnx, input_tensor, args.target_platform)
    detector_input_count = len(detector_session.get_inputs())
    if detector_input_count > len(rknn_outputs):
        raise ValueError(
            f"Detector expects {detector_input_count} inputs but RKNN backbone produced {len(rknn_outputs)} outputs."
        )

    detector_outputs = run_model_multi_input(detector_session, rknn_outputs[:detector_input_count])
    if len(detector_outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from detector ONNX model, got {len(detector_outputs)}.")
    dets_rknn = detector_outputs[0]
    logits_rknn = detector_outputs[1]
    masks_rknn = detector_outputs[2] if len(detector_outputs) >= 3 else None

    ensure_shape_match("dets", dets_full, dets_rknn)
    ensure_shape_match("labels", logits_full, logits_rknn)
    if masks_full is not None and masks_rknn is not None:
        ensure_shape_match("masks", masks_full, masks_rknn)
    elif masks_full is not None or masks_rknn is not None:
        raise ValueError("One pipeline returned masks while the other did not.")

    print_raw_diff(
        "Raw tensor sanity check (full ONNX vs RKNN+ONNX simulated)",
        dets_full,
        logits_full,
        dets_rknn,
        logits_rknn,
        masks_full,
        masks_rknn,
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
    rknn_xyxy, rknn_conf, rknn_cls, rknn_masks = postprocess_onnx(
        dets=dets_rknn,
        logits=logits_rknn,
        orig_h=orig_h,
        orig_w=orig_w,
        threshold=args.threshold,
        num_select=args.num_select,
        masks=masks_rknn,
    )

    print_post_diff_matched(
        "Postprocess sanity check (full ONNX vs RKNN+ONNX simulated)",
        full_xyxy,
        full_conf,
        full_cls,
        rknn_xyxy,
        rknn_conf,
        rknn_cls,
        full_masks,
        rknn_masks,
    )

    annotate_and_save(
        scene=original_scene,
        boxes_xyxy=rknn_xyxy,
        confidences=rknn_conf,
        class_ids=rknn_cls,
        output_path=args.output_image,
        coco_classes=load_coco_classes(),
        masks=rknn_masks,
    )
    print(f"Saved {args.output_image} with {len(rknn_cls)} detections above threshold {args.threshold}.")


if __name__ == "__main__":
    main()
