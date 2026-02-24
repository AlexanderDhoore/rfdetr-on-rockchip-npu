import argparse
from pathlib import Path
from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert backbone ONNX to RKNN format.")
    parser.add_argument("--input", default="output/backbone.onnx", help="Path to input backbone ONNX model.")
    parser.add_argument("--output", default=None, help="Optional output RKNN path. Defaults to input path with .rknn suffix.")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target platform, e.g. rk3588.")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization during build.")
    args = parser.parse_args()

    inpath = Path(args.input).resolve()
    if not inpath.exists():
        raise FileNotFoundError(f"Input ONNX file not found: {inpath}")

    outpath = Path(args.output).resolve() if args.output else inpath.with_suffix(".rknn")

    rknn = RKNN()
    try:
        print(f"Config target={args.target_platform}")
        ret = rknn.config(target_platform=args.target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        print(f"Load {inpath}")
        ret = rknn.load_onnx(model=str(inpath))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        print(f"Build (quantize={args.quantize})")
        ret = rknn.build(do_quantization=args.quantize)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        outpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Export {outpath}")
        ret = rknn.export_rknn(str(outpath))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    finally:
        rknn.release()

    print("Done.")


if __name__ == "__main__":
    main()
