import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite

from pipeline_common import ensure_files_exist, make_ort_session, preprocess_image, run_model_multi_input, run_model_single_input


def summarize(name: str, values_ms: list[float]) -> None:
    arr = np.array(values_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    std_ms = float(arr.std())
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    print(f"{name}: mean={mean_ms:.3f} ms  min={min_ms:.3f}  max={max_ms:.3f}  std={std_ms:.3f}  fps={fps:.2f}")


def stats_dict(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    std_ms = float(arr.std())
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "fps": fps,
    }


def run_full_onnx_once(session, input_tensor: np.ndarray) -> None:
    outputs = run_model_single_input(session, input_tensor)
    if len(outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from full ONNX model, got {len(outputs)}.")


def run_split_onnx_once(
    backbone_session,
    detector_session,
    input_tensor: np.ndarray,
) -> tuple[float, float]:
    detector_input_count = len(detector_session.get_inputs())

    t0 = time.perf_counter()
    backbone_outputs = run_model_single_input(backbone_session, input_tensor)
    t1 = time.perf_counter()

    if detector_input_count > len(backbone_outputs):
        raise ValueError(
            f"Detector expects {detector_input_count} inputs but split backbone produced {len(backbone_outputs)} outputs."
        )

    detector_outputs = run_model_multi_input(detector_session, backbone_outputs[:detector_input_count])
    t2 = time.perf_counter()

    if len(detector_outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from detector ONNX model, got {len(detector_outputs)}.")

    return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def run_rknn_onnx_once(
    rknn: RKNNLite,
    detector_session,
    input_tensor: np.ndarray,
) -> tuple[float, float]:
    detector_input_count = len(detector_session.get_inputs())

    t0 = time.perf_counter()
    rknn_outputs = rknn.inference(inputs=[input_tensor], data_format=["nchw"])
    t1 = time.perf_counter()

    if rknn_outputs is None:
        raise RuntimeError("RKNNLite.inference returned None")

    rknn_outputs_list = list(rknn_outputs)
    if detector_input_count > len(rknn_outputs_list):
        raise ValueError(
            f"Detector expects {detector_input_count} inputs but RKNN backbone produced {len(rknn_outputs_list)} outputs."
        )

    detector_outputs = run_model_multi_input(detector_session, rknn_outputs_list[:detector_input_count])
    t2 = time.perf_counter()

    if len(detector_outputs) < 2:
        raise ValueError(f"Expected at least 2 outputs from detector ONNX model, got {len(detector_outputs)}.")

    return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RF-DETR inference on Rock 5B (full ONNX, split ONNX, RKNN+ONNX)."
    )
    parser.add_argument("--image", default="bus.jpg", help="Input image path (used for one-time preprocessing).")
    parser.add_argument("--output-dir", default="output", help="Directory containing inference_model/backbone/detection ONNX and backbone.rknn.")
    parser.add_argument(
        "--detector-onnx",
        default="detection.onnx",
        help="Detector ONNX filename inside output-dir (default: dynamic-quantized detector).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations per pipeline.")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed iterations per pipeline.")
    parser.add_argument("--ort-intra-threads", type=int, default=4, help="ONNX Runtime intra-op threads for CPU sessions.")
    parser.add_argument("--ort-inter-threads", type=int, default=1, help="ONNX Runtime inter-op threads for CPU sessions.")
    parser.add_argument("--model-name", default="", help="Optional model name label for saved benchmark JSON.")
    parser.add_argument("--json-output", default="", help="Optional JSON output path for benchmark summary.")
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.ort_intra_threads <= 0:
        raise ValueError("--ort-intra-threads must be > 0")
    if args.ort_inter_threads <= 0:
        raise ValueError("--ort-inter-threads must be > 0")

    output_dir = Path(args.output_dir)
    full_onnx = output_dir / "inference_model.onnx"
    backbone_onnx = output_dir / "backbone.onnx"
    detector_onnx = output_dir / args.detector_onnx
    backbone_rknn = output_dir / "backbone.rknn"

    ensure_files_exist([Path(args.image), full_onnx, backbone_onnx, detector_onnx, backbone_rknn])

    full_session = make_ort_session(str(full_onnx), args.ort_intra_threads, args.ort_inter_threads)
    split_backbone_session = make_ort_session(str(backbone_onnx), args.ort_intra_threads, args.ort_inter_threads)
    split_detector_session = make_ort_session(str(detector_onnx), args.ort_intra_threads, args.ort_inter_threads)
    deploy_detector_session = make_ort_session(str(detector_onnx), args.ort_intra_threads, args.ort_inter_threads)

    full_input = full_session.get_inputs()[0]
    split_input = split_backbone_session.get_inputs()[0]
    if full_input.shape[2:] != split_input.shape[2:]:
        raise ValueError(
            f"Input size mismatch: full={full_input.shape} split_backbone={split_input.shape}. "
            "Export models at the same resolution."
        )

    input_h = int(full_input.shape[2])
    input_w = int(full_input.shape[3])
    input_tensor, _, _, _ = preprocess_image(args.image, input_h, input_w)
    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(str(backbone_rknn))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}")

        full_total_ms: list[float] = []
        split_backbone_ms: list[float] = []
        split_detector_ms: list[float] = []
        split_total_ms: list[float] = []
        rknn_backbone_ms: list[float] = []
        rknn_detector_ms: list[float] = []
        rknn_total_ms: list[float] = []

        for _ in range(args.warmup):
            run_full_onnx_once(full_session, input_tensor)

        for _ in range(args.runs):
            t0 = time.perf_counter()
            run_full_onnx_once(full_session, input_tensor)
            t1 = time.perf_counter()
            full_total_ms.append((t1 - t0) * 1000.0)

        for _ in range(args.warmup):
            run_split_onnx_once(split_backbone_session, split_detector_session, input_tensor)

        for _ in range(args.runs):
            split_bb, split_det = run_split_onnx_once(split_backbone_session, split_detector_session, input_tensor)
            split_backbone_ms.append(split_bb)
            split_detector_ms.append(split_det)
            split_total_ms.append(split_bb + split_det)

        for _ in range(args.warmup):
            run_rknn_onnx_once(rknn, deploy_detector_session, input_tensor)

        for _ in range(args.runs):
            rknn_bb, rknn_det = run_rknn_onnx_once(rknn, deploy_detector_session, input_tensor)
            rknn_backbone_ms.append(rknn_bb)
            rknn_detector_ms.append(rknn_det)
            rknn_total_ms.append(rknn_bb + rknn_det)

    finally:
        rknn.release()

    print(
        f"Benchmark config: warmup={args.warmup}, runs={args.runs}, image={args.image}, "
        f"resolution={input_h}x{input_w}, ort_intra={args.ort_intra_threads}, ort_inter={args.ort_inter_threads}"
    )
    print("")
    print("Full ONNX")
    summarize("  total", full_total_ms)
    print("")
    print("Split ONNX")
    summarize("  backbone", split_backbone_ms)
    summarize("  detector", split_detector_ms)
    summarize("  total", split_total_ms)
    print("")
    print("RKNN + ONNX")
    summarize("  backbone", rknn_backbone_ms)
    summarize("  detector", rknn_detector_ms)
    summarize("  total", rknn_total_ms)

    if args.json_output:
        payload = {
            "model_name": args.model_name,
            "image": args.image,
            "resolution": {"h": input_h, "w": input_w},
            "config": {
                "warmup": args.warmup,
                "runs": args.runs,
                "ort_intra_threads": args.ort_intra_threads,
                "ort_inter_threads": args.ort_inter_threads,
                "detector_onnx": args.detector_onnx,
            },
            "full_onnx": {
                "total": stats_dict(full_total_ms),
            },
            "split_onnx": {
                "backbone": stats_dict(split_backbone_ms),
                "detector": stats_dict(split_detector_ms),
                "total": stats_dict(split_total_ms),
            },
            "rknn_onnx": {
                "backbone": stats_dict(rknn_backbone_ms),
                "detector": stats_dict(rknn_detector_ms),
                "total": stats_dict(rknn_total_ms),
            },
        }
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print("")
        print(f"Saved benchmark JSON: {out_path}")


if __name__ == "__main__":
    main()
