import argparse
from pathlib import Path
import shutil

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from rfdetr import (
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegPreview,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)


MODEL_REGISTRY = {
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge": RFDETRLarge,
    "RFDETRSegPreview": RFDETRSegPreview,
    "RFDETRSegNano": RFDETRSegNano,
    "RFDETRSegSmall": RFDETRSegSmall,
    "RFDETRSegMedium": RFDETRSegMedium,
    "RFDETRSegLarge": RFDETRSegLarge,
    "RFDETRSegXLarge": RFDETRSegXLarge,
    "RFDETRSeg2XLarge": RFDETRSeg2XLarge,
}


class BackboneWrapper(torch.nn.Module):
    def __init__(self, lwdetr: torch.nn.Module):
        super().__init__()
        self.backbone = lwdetr.backbone

    def forward(self, tensors: torch.Tensor):
        srcs, _, poss = self.backbone(tensors)
        del poss
        return tuple(srcs)


class DetectionWrapper(torch.nn.Module):
    def __init__(self, lwdetr: torch.nn.Module, poss: list[torch.Tensor], image_hw: tuple[int, int]):
        super().__init__()
        self.transformer = lwdetr.transformer
        self.refpoint_embed = lwdetr.refpoint_embed
        self.query_feat = lwdetr.query_feat
        self.class_embed = lwdetr.class_embed
        self.bbox_embed = lwdetr.bbox_embed
        self.segmentation_head = lwdetr.segmentation_head
        self.bbox_reparam = lwdetr.bbox_reparam
        self.two_stage = lwdetr.two_stage
        self.num_queries = lwdetr.num_queries
        self.num_pos = len(poss)
        self.image_h = int(image_hw[0])
        self.image_w = int(image_hw[1])

        for i, p in enumerate(poss):
            self.register_buffer(f"pos_{i}", p.detach().clone(), persistent=True)

    def forward(self, *srcs: torch.Tensor):
        if len(srcs) != self.num_pos:
            raise ValueError(f"Expected {self.num_pos} feature maps, got {len(srcs)}")

        poss = [getattr(self, f"pos_{i}") for i in range(self.num_pos)]
        srcs_list = list(srcs)

        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs_list,
            None,
            poss,
            refpoint_embed_weight,
            query_feat_weight,
        )

        outputs_masks = None

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs_list[0], [hs], (self.image_h, self.image_w))[0]
        else:
            if not self.two_stage:
                raise ValueError("Expected two_stage=True when hs is None.")

            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs_list[0],
                    [hs_enc],
                    (self.image_h, self.image_w),
                    skip_blocks=True,
                )[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks

        return outputs_coord, outputs_class


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RF-DETR ONNX models: full official ONNX + split backbone/detection ONNX."
    )
    parser.add_argument(
        "--model",
        default="RFDETRMedium",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="RF-DETR model class name.",
    )
    parser.add_argument("--weights", default=None, help="Optional local checkpoint path.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated ONNX files.")
    parser.add_argument("--resolution", type=int, default=None, help="Optional inference resolution override.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--verbose", action="store_true", help="Verbose ONNX export output.")
    parser.add_argument(
        "--no-dynamic-quant",
        action="store_true",
        help="Disable dynamic quantization for detector export.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ["inference_model.onnx", "backbone.onnx", "detection.onnx", "detection_fp32.onnx"]:
        (output_dir / file_name).unlink(missing_ok=True)

    model_kwargs = {}
    if args.weights:
        model_kwargs["pretrain_weights"] = args.weights
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution

    model_ctor = MODEL_REGISTRY[args.model]
    model = model_ctor(**model_kwargs)

    print(f"Exporting full ONNX via official RF-DETR path into: {output_dir}")
    model.export(output_dir=str(output_dir))

    full_onnx = output_dir / "inference_model.onnx"
    if not full_onnx.exists():
        raise FileNotFoundError(f"Official ONNX export did not produce expected file: {full_onnx}")

    lwdetr = model.model.model.to("cpu").eval()
    lwdetr.export()

    resolution = int(model.model.resolution)
    dummy_input = torch.randn(1, 3, resolution, resolution, dtype=torch.float32)

    with torch.no_grad():
        srcs, _, poss = lwdetr.backbone(dummy_input)

    if len(poss) != len(srcs):
        raise ValueError(f"Unexpected backbone outputs: len(srcs)={len(srcs)} len(poss)={len(poss)}")

    backbone_wrapper = BackboneWrapper(lwdetr).eval()
    detector_wrapper = DetectionWrapper(lwdetr, poss, (resolution, resolution)).eval()

    backbone_onnx = output_dir / "backbone.onnx"
    detector_fp32_onnx = output_dir / "detection_fp32.onnx"
    detector_onnx = output_dir / "detection.onnx"

    torch.onnx.export(
        backbone_wrapper,
        dummy_input,
        str(backbone_onnx),
        opset_version=args.opset,
        verbose=args.verbose,
        dynamo=False,
    )

    torch.onnx.export(
        detector_wrapper,
        tuple(srcs),
        str(detector_fp32_onnx),
        opset_version=args.opset,
        verbose=args.verbose,
        dynamo=False,
    )

    if args.no_dynamic_quant:
        shutil.copyfile(detector_fp32_onnx, detector_onnx)
    else:
        detector_prep = output_dir / "detection_prep.onnx"
        detector_prep.unlink(missing_ok=True)
        quant_pre_process(str(detector_fp32_onnx), str(detector_prep))
        quantize_dynamic(
            str(detector_prep),
            str(detector_onnx),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        detector_prep.unlink(missing_ok=True)

    print(f"Model: {args.model}")
    print(f"Exported: {full_onnx}")
    print(f"Exported: {backbone_onnx}")
    print(f"Exported: {detector_fp32_onnx}")
    print(f"Exported: {detector_onnx}")
    print(f"Resolution: {resolution}x{resolution}")
    print("Detector positional encodings are embedded as ONNX constants (not runtime inputs).")
    if args.no_dynamic_quant:
        print("Dynamic quantization disabled: detection.onnx is copied from detection_fp32.onnx.")
    else:
        print("Dynamic quantization enabled: detection.onnx is INT8-weight dynamic quantized.")


if __name__ == "__main__":
    main()
