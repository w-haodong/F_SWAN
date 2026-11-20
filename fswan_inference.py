"""Inference for the ViT-B and wavelet configuration of F-SWAN."""

from collections.abc import Mapping
from functools import lru_cache
from hashlib import file_digest
import warnings

import cv2
import numpy as np
import pywt
import torch
from torch import nn

from models.DANN import DomainClassifier
from models.dec_net import DecNet
from operation.cobb_evaluate_base import cobb_angle_calc
from operation.decoder import DecDecoder
from operation.robust_quadrilateral import fit_quadrilateral


INPUT_HEIGHT = 1280
INPUT_WIDTH = 512
OUTPUT_STRIDE = 4
VERTEBRAE = 17
LAYER_INDICES = (0, 2, 5, 11)
CONTOUR_POINTS = 32
WAVELET = "db4"
WAVELET_LEVEL = 4
WAVELET_MODE = "symmetric"
SLIM_CHECKPOINT_FORMAT = "fswan-lora-wavelet-v1"


@lru_cache(maxsize=1)
def coefficient_lengths():
    # This decomposition intentionally uses boundary extension at every level.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Level value of .* is too high")
        coefficients = pywt.wavedec(
            np.zeros(CONTOUR_POINTS), WAVELET, mode=WAVELET_MODE, level=WAVELET_LEVEL
        )
    return tuple(map(len, coefficients))


class FSwan(nn.Module):
    """Model with the parameter names used by the F-SWAN checkpoint."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dec_net = DecNet(
            {"hm": 1, "reg": 2, "wavelet": 2 * sum(coefficient_lengths())},
            vit_input_layer_indices=LAYER_INDICES,
        )
        # Retain the classifier so full checkpoints can be loaded strictly.
        self.domain_classifier = DomainClassifier(self.dec_net.final_feature_channels)

    def forward(self, image):
        features = self.encoder.get_base_model().get_intermediate_layers(
            image, n=LAYER_INDICES, reshape=True, norm=True
        )
        outputs, _, _ = self.dec_net(list(features), compute_style=False)
        return outputs


def load_model(checkpoint, dino_checkpoint, device):
    from peft import LoraConfig, get_peft_model

    try:
        from dinov3.hub.backbones import dinov3_vitb16
    except ModuleNotFoundError as exc:
        if exc.name != "dinov3":
            raise
        raise ImportError("Install the official DINOv3 package to use this CLI.") from exc

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, Mapping):
        raise ValueError("The F-SWAN checkpoint must contain a model state dictionary.")
    state = saved.get("state_dict", saved)
    if not isinstance(state, Mapping):
        raise ValueError("The checkpoint's state_dict is not a model state dictionary.")
    slim = saved.get("format") == SLIM_CHECKPOINT_FORMAT

    with open(dino_checkpoint, "rb") as backbone_file:
        if slim:
            expected_hash = saved.get("backbone_sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError("The F-SWAN checkpoint is missing a valid DINOv3 SHA-256.")
            backbone_hash = file_digest(backbone_file, "sha256").hexdigest()
            if backbone_hash != expected_hash.lower():
                raise ValueError("The DINOv3 weights do not match the SHA-256 in the F-SWAN checkpoint.")
            backbone_file.seek(0)
        backbone_state = torch.load(backbone_file, map_location="cpu", weights_only=True)

    backbone = dinov3_vitb16(pretrained=False)
    backbone.load_state_dict(backbone_state, strict=True)
    del backbone_state
    encoder = get_peft_model(
        backbone,
        LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.1, bias="none"),
    )
    model = FSwan(encoder)
    if slim:
        full_state = model.state_dict()
        expected = {name for name in full_state if not name.startswith("encoder.")}
        expected.update(
            name
            for name, parameter in model.named_parameters()
            if name.startswith("encoder.") and parameter.requires_grad
        )
        missing = sorted(expected - state.keys())
        unexpected = sorted(state.keys() - expected, key=str)
        if missing or unexpected:
            raise ValueError(
                "The F-SWAN inference checkpoint has incompatible keys "
                f"(missing: {missing[:5]}, unexpected: {unexpected[:5]})."
            )
        full_state.update(state)
        state = full_state
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def prepare_image(image_path):
    data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    if not data.size:
        raise ValueError("The image file is empty.")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode the supplied radiograph.")

    height, width = image.shape[:2]
    scale = min(INPUT_HEIGHT / height, INPUT_WIDTH / width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (INPUT_WIDTH - resized_width) // 2
    top = (INPUT_HEIGHT - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    normalized = canvas.astype(np.float32) / 255.0 - 0.5
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, scale, np.array([left, top], dtype=np.float32)


def reconstruct_corners(detection):
    center = detection[1:3]
    coefficients = detection[3:] * (INPUT_WIDTH // OUTPUT_STRIDE)
    boundaries = np.cumsum(coefficient_lengths())
    x_coefficients, y_coefficients = np.split(coefficients, [boundaries[-1]])
    x = pywt.waverec(np.split(x_coefficients, boundaries[:-1]), WAVELET, mode=WAVELET_MODE)
    y = pywt.waverec(np.split(y_coefficients, boundaries[:-1]), WAVELET, mode=WAVELET_MODE)
    contour = np.stack((x, y), axis=1)[:CONTOUR_POINTS] + center
    # The reference decoder fits on the output grid and preserves the contour center.
    grid_points = contour.astype(np.float64)
    projected, _ = fit_quadrilateral(grid_points)
    if projected is None:
        raise ValueError("The predicted contour gives a degenerate or nonconvex quadrilateral.")
    perimeter_corners = (
        (projected[:: CONTOUR_POINTS // 4] + grid_points.mean(axis=0)) * OUTPUT_STRIDE
    ).astype(np.float32)

    by_x = sorted(perimeter_corners, key=lambda point: point[0])
    top_left, bottom_left = sorted(by_x[:2], key=lambda point: point[1])
    top_right, bottom_right = sorted(by_x[2:], key=lambda point: point[1])
    return np.stack((top_left, top_right, bottom_left, bottom_right))


@torch.inference_mode()
def predict(image_path, checkpoint, dino_checkpoint, requested_device="auto"):
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Use --device auto, cpu, cuda, or cuda:N.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    image, scale, padding = prepare_image(image_path)
    model = load_model(checkpoint, dino_checkpoint, device)
    outputs = model(image.to(device))
    detections = DecDecoder().decode(
        outputs["hm"], outputs["reg"], outputs["wavelet"], K=VERTEBRAE
    )[0].cpu().numpy()
    corners = np.stack([reconstruct_corners(detection) for detection in detections])
    ordered = corners[np.argsort(corners.mean(axis=1)[:, 1])]
    pt, mt, tl, _ = cobb_angle_calc(ordered.reshape(-1, 2), None, is_train=False)
    angles = {"PT": round(float(pt), 2), "MT": round(float(mt), 2), "TL_L": round(float(tl), 2)}
    if not np.isfinite(list(angles.values())).all():
        raise ValueError("Cobb angle calculation returned a nonfinite value.")

    return {
        "angles_deg": angles,
        "corner_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
        "corners_xy_px": ((ordered - padding) / scale).tolist(),
    }
