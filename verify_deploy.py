import argparse
from pathlib import Path

from rknnlite.api import RKNNLite

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


def run_backbone_rknn_lite(backbone_rknn: Path, input_tensor):
    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(str(backbone_rknn))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}")

        outputs = rknn.inference(inputs=[input_tensor], data_format=["nchw"])
        if outputs is None:
            raise RuntimeError("RKNNLite.inference returned None")
        return list(outputs)
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full ONNX vs RKNNLite(backbone)+ONNX(detector) on real NPU.")
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument("--output-dir", default="output", help="Directory with ONNX/RKNN files.")
    parser.add_argument(
        "--detector-onnx",
        default="detection.onnx",
        help="Detector ONNX filename inside output-dir (default: dynamic-quantized detector).",
    )
    parser.add_argument("--output-image", default="bus_deploy.jpg", help="Annotated output image path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--num-select", type=int, default=300, help="Top-K predictions before thresholding.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    full_onnx = output_dir / "inference_model.onnx"
    backbone_onnx = output_dir / "backbone.onnx"
    detector_onnx = output_dir / args.detector_onnx
    backbone_rknn = output_dir / "backbone.rknn"

    ensure_files_exist([Path(args.image), full_onnx, backbone_onnx, detector_onnx, backbone_rknn])
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

    rknn_outputs = run_backbone_rknn_lite(backbone_rknn, input_tensor)
    detector_input_count = len(detector_session.get_inputs())
    if detector_input_count > len(rknn_outputs):
        raise ValueError(
            f"Detector expects {detector_input_count} inputs but RKNNLite backbone produced {len(rknn_outputs)} outputs."
        )

    detector_outputs = run_model_multi_input(detector_session, rknn_outputs[:detector_input_count])
    if len(detector_outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from detector ONNX model, got {len(detector_outputs)}.")
    dets_deploy = detector_outputs[0]
    logits_deploy = detector_outputs[1]
    masks_deploy = detector_outputs[2] if len(detector_outputs) >= 3 else None

    ensure_shape_match("dets", dets_full, dets_deploy)
    ensure_shape_match("labels", logits_full, logits_deploy)
    if masks_full is not None and masks_deploy is not None:
        ensure_shape_match("masks", masks_full, masks_deploy)
    elif masks_full is not None or masks_deploy is not None:
        raise ValueError("One pipeline returned masks while the other did not.")

    print_raw_diff(
        "Raw tensor sanity check (full ONNX vs RKNNLite+ONNX)",
        dets_full,
        logits_full,
        dets_deploy,
        logits_deploy,
        masks_full,
        masks_deploy,
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
    deploy_xyxy, deploy_conf, deploy_cls, deploy_masks = postprocess_onnx(
        dets=dets_deploy,
        logits=logits_deploy,
        orig_h=orig_h,
        orig_w=orig_w,
        threshold=args.threshold,
        num_select=args.num_select,
        masks=masks_deploy,
    )

    print_post_diff_matched(
        "Postprocess sanity check (full ONNX vs RKNNLite+ONNX)",
        full_xyxy,
        full_conf,
        full_cls,
        deploy_xyxy,
        deploy_conf,
        deploy_cls,
        full_masks,
        deploy_masks,
    )

    annotate_and_save(
        scene=original_scene,
        boxes_xyxy=deploy_xyxy,
        confidences=deploy_conf,
        class_ids=deploy_cls,
        output_path=args.output_image,
        coco_classes=load_coco_classes(),
        masks=deploy_masks,
    )
    print(f"Saved {args.output_image} with {len(deploy_cls)} detections above threshold {args.threshold}.")


if __name__ == "__main__":
    main()
