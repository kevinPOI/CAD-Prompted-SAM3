#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Evaluate image exemplars using the official SAM3 image model.

This is a port of the local muggled_sam image-exemplar eval path to the
official SAM3 modules in this repo. It keeps the dataset/metric plumbing from
the custom script, but replaces the split custom detector modules with the
official low-level sequence:

    backbone.forward_image -> _run_encoder -> _run_decoder -> _run_segmentation_heads

Reference images are encoded once per object into prompt tokens. Those tokens
are then reused as cross-image exemplar prompts for target images.
"""

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FindStage = None
concat_padded_sequences = None
build_sam3_image_model = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


IGNORED_LABELS = {"BACKGROUND", "UNLABELLED"}


def load_sam3_symbols() -> None:
    global FindStage, concat_padded_sequences, build_sam3_image_model

    if FindStage is not None:
        return
    try:
        from sam3.model.data_misc import FindStage as _FindStage
        from sam3.model.geometry_encoders import concat_padded_sequences as _concat_padded_sequences
        from sam3.model_builder import build_sam3_image_model as _build_sam3_image_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc}. Install the official SAM3 package dependencies from pyproject.toml before running evaluation."
        ) from exc

    FindStage = _FindStage
    concat_padded_sequences = _concat_padded_sequences
    build_sam3_image_model = _build_sam3_image_model


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim < 3:
        return image_bgr
    return image_bgr[..., ::-1]


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim < 3:
        return image_rgb
    return image_rgb[..., ::-1]


def resize_array(
    array: np.ndarray,
    size_wh: Tuple[int, int],
    nearest: bool = False,
) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(array, size_wh, interpolation=interpolation)
    pil_mode = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(array).resize(size_wh, pil_mode))


def connected_component_bboxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Pure numpy fallback for connected-component bounding boxes."""

    mask = mask.astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    bboxes: List[Tuple[int, int, int, int]] = []
    starts = np.argwhere(mask)
    for start_y, start_x in starts:
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        x0 = x1 = int(start_x)
        y0 = y1 = int(start_y)
        while queue:
            y, x = queue.popleft()
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        bboxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
    return bboxes


def parse_color_key(key: str) -> Tuple[int, ...]:
    stripped = key.strip().strip("()")
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def parse_image_list(raw: str) -> Optional[set]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        try:
            value = ast.literal_eval(raw)
            items = list(value) if isinstance(value, (list, tuple, set)) else [value]
        except (ValueError, SyntaxError):
            items = [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    else:
        items = [part.strip() for part in raw.split(",") if part.strip()]
    items = [os.path.abspath(os.path.expanduser(str(item))) for item in items if item]
    return set(items) if items else None


def load_color_mapping(json_path: Path) -> Dict[str, List[Tuple[int, ...]]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    object_colors: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
    for color_key, label in raw.items():
        clean_label = label.strip()
        if not clean_label or clean_label.upper() in IGNORED_LABELS:
            continue
        object_colors[clean_label].append(parse_color_key(color_key))
    return dict(object_colors)


def collect_multi_object_samples(
    dataset_root: str,
    sub_sample: int = 5,
) -> Tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_root)
    if sub_sample < 1:
        raise ValueError("sub_sample must be >= 1")

    pattern = re.compile(r"instance_segmentation_(\d{4})\.png$")
    object_map: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    all_entries: List[Dict[str, object]] = []
    for idx, inst_path in enumerate(sorted(dataset_path.glob("instance_segmentation_*.png"))):
        if sub_sample > 1 and (idx % sub_sample) != 0:
            continue
        match = pattern.match(inst_path.name)
        if not match:
            continue
        frame_id = match.group(1)
        rgb_path = dataset_path / f"rgb_{frame_id}.png"
        if not rgb_path.is_file():
            jpg_fallback = dataset_path / f"rgb_{frame_id}.jpg"
            if jpg_fallback.is_file():
                rgb_path = jpg_fallback
        mapping_path = dataset_path / f"instance_segmentation_mapping_{frame_id}.json"
        if not (rgb_path.is_file() and mapping_path.is_file()):
            continue

        color_map = load_color_mapping(mapping_path)
        for object_id, colors in color_map.items():
            for color in colors:
                entry = {
                    "object_id": object_id,
                    "frame_id": frame_id,
                    "rgb_path": str(rgb_path),
                    "inst_path": str(inst_path),
                    "color": color,
                }
                object_map[object_id].append(entry)
                all_entries.append(entry)

    if not all_entries:
        raise RuntimeError(f"No multi-object samples found under {dataset_root}")
    return object_map, all_entries


def load_instance_segmentation(inst_path: str) -> np.ndarray:
    if cv2 is not None:
        seg = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
        if seg is None:
            raise FileNotFoundError(inst_path)
        if seg.ndim == 2:
            seg = seg[..., None]
        elif seg.shape[2] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
        elif seg.shape[2] == 3:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        return seg.astype(np.uint8)

    try:
        seg = np.asarray(Image.open(inst_path))
    except FileNotFoundError:
        raise
    if seg.ndim == 2:
        seg = seg[..., None]
    return seg.astype(np.uint8)


def mask_for_color(seg: np.ndarray, color: Tuple[int, ...]) -> np.ndarray:
    target = np.array(color, dtype=np.uint8)
    channels = seg.shape[2]
    if target.shape[0] > channels:
        target = target[:channels]
    elif target.shape[0] < channels:
        target = np.concatenate([target, np.zeros(channels - target.shape[0], dtype=np.uint8)])
    return np.all(seg == target.reshape(1, 1, -1), axis=-1).astype(np.float32)


def load_instance_masks_for_object(
    inst_path: str,
    mapping_path: str,
    object_id: str,
    seg_cache: Optional[Dict[str, np.ndarray]] = None,
    mapping_cache: Optional[Dict[str, Dict[str, List[Tuple[int, ...]]]]] = None,
) -> List[np.ndarray]:
    if mapping_cache is not None and mapping_path in mapping_cache:
        color_map = mapping_cache[mapping_path]
    else:
        color_map = load_color_mapping(Path(mapping_path))
        if mapping_cache is not None:
            mapping_cache[mapping_path] = color_map

    colors = color_map.get(object_id, [])
    if not colors:
        return []

    if seg_cache is not None and inst_path in seg_cache:
        seg = seg_cache[inst_path]
    else:
        seg = load_instance_segmentation(inst_path)
        if seg_cache is not None:
            seg_cache[inst_path] = seg

    masks: List[np.ndarray] = []
    for color in colors:
        mask = mask_for_color(seg, color)
        if mask.sum() > 0:
            masks.append(mask)
    return masks


def load_bgr(path: str) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image
    try:
        image_rgb = np.asarray(Image.open(path).convert("RGB"))
    except FileNotFoundError:
        raise
    return rgb_to_bgr(image_rgb)


def apply_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        gray = image_bgr if image_bgr.ndim == 2 else image_bgr[:, :, 0]
    elif cv2 is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.asarray(Image.fromarray(bgr_to_rgb(image_bgr)).convert("L"))
    if cv2 is not None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.repeat(gray[:, :, None], 3, axis=2)[:, :, ::-1]


def load_mask_gray(path: str) -> np.ndarray:
    if cv2 is not None:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
    else:
        try:
            mask = np.asarray(Image.open(path).convert("L"))
        except FileNotFoundError:
            raise
    return (mask > 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if cv2 is not None:
        return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        in_mask = (mask > 0).astype(np.uint8)
    else:
        in_mask = mask
    return resize_array(in_mask, (w, h), nearest=True).astype(np.float32)


def sample_points_from_mask(mask_image: np.ndarray, num_points_approx: int = 25) -> List[Tuple[float, float]]:
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    num_fib_pts = golden_ratio * num_points_approx
    pt_idx = np.arange(0, num_fib_pts, dtype=np.float32)
    radius = np.sqrt(pt_idx / num_fib_pts) / np.sqrt(2, dtype=np.float32)
    theta = 2.0 * np.pi * (pt_idx / golden_ratio)

    sample_x_norm = 0.5 + radius * np.cos(theta)
    sample_y_norm = 0.5 + radius * np.sin(theta)
    ok = (sample_x_norm > 0.0) & (sample_x_norm < 1.0) & (sample_y_norm > 0.0) & (sample_y_norm < 1.0)
    sample_x_norm, sample_y_norm = sample_x_norm[ok], sample_y_norm[ok]

    if mask_image.ndim > 2:
        if cv2 is not None and mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        else:
            mask_image = mask_image[:, :, 0]
    mask_bin = mask_image > 0
    ref_h, ref_w = mask_bin.shape[:2]

    final_samples = []
    if cv2 is not None:
        contours, _ = cv2.findContours(np.uint8(mask_bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(contour) for contour in contours if len(contour) >= 3]
    else:
        bboxes = connected_component_bboxes(mask_bin)

    for x1, y1, w, h in bboxes:
        if w < 1 or h < 1:
            continue
        sample_x_px = np.round(x1 + sample_x_norm * (w - 1)).astype(np.int32)
        sample_y_px = np.round(y1 + sample_y_norm * (h - 1)).astype(np.int32)
        in_mask = mask_bin[sample_y_px, sample_x_px]
        final_samples.append(np.column_stack((sample_x_px[in_mask], sample_y_px[in_mask])))

    if not final_samples:
        return []
    out_xy_norm = np.concatenate(final_samples) / np.float32((ref_w - 1, ref_h - 1))
    return out_xy_norm.tolist()


def build_gt_down_list(
    gt_masks: List[np.ndarray],
    preencode_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> List[torch.Tensor]:
    gt_down_list: List[torch.Tensor] = []
    for gt_mask in gt_masks:
        gt_preenc = resize_mask(gt_mask, preencode_hw)
        gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
        gt_down = F.interpolate(gt_tensor, size=target_hw, mode="nearest").squeeze(0).squeeze(0)
        gt_down_list.append(gt_down > 0.5)
    return gt_down_list


def parse_ref_view_ids(value: str) -> List[str]:
    if not value:
        return []
    ids = []
    for part in [part.strip() for part in value.split(",") if part.strip()]:
        try:
            ids.append(f"{int(part):02d}")
        except ValueError:
            ids.append(part)
    return ids


class OfficialSam3ImageExemplarAdapter:
    """Thin adapter around official SAM3 internals for cross-image exemplars."""

    def __init__(
        self,
        model,
        device: torch.device,
        resolution: int = 1008,
        dtype: torch.dtype = torch.float32,
        include_coordinate_encodings: bool = False,
    ):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.dtype = dtype
        self.include_coordinate_encodings = include_coordinate_encodings
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def autocast_context(self):
        if self.device.type == "cuda" and self.dtype == torch.bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def preprocess_bgr(self, image_bgr: np.ndarray) -> torch.Tensor:
        if cv2 is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = bgr_to_rgb(image_bgr)
        image = v2.functional.to_image(image_rgb).to(self.device)
        return self.transform(image).unsqueeze(0)

    def make_find_input(self, batch_size: int) -> object:
        return FindStage(
            img_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
            text_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def encode_reference_points(
        self,
        image_bgr: np.ndarray,
        point_xy_norm_list: List[Tuple[float, float]],
        text: str = "visual",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not point_xy_norm_list:
            raise ValueError("Reference point list is empty.")

        img_batch = self.preprocess_bgr(image_bgr)
        find_input = self.make_find_input(batch_size=1)
        with self.autocast_context():
            backbone_out = self.model.backbone.forward_image(img_batch)
            text_out = self.model.backbone.forward_text([text], device=self.device)
            backbone_out.update(text_out)
            feat_tuple = self.model._get_img_feats(backbone_out, find_input.img_ids)
            backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

            point_tensor = torch.tensor(
                point_xy_norm_list,
                device=self.device,
                dtype=img_feats[-1].dtype,
            ).view(-1, 1, 2)
            point_mask = torch.zeros(1, point_tensor.shape[0], device=self.device, dtype=torch.bool)
            point_labels = torch.ones(point_tensor.shape[0], 1, device=self.device, dtype=torch.long)

            geo_feats, geo_mask = self._encode_points_with_optional_coords(
                points=point_tensor,
                points_mask=point_mask,
                points_labels=point_labels,
                img_feats=img_feats,
                img_sizes=vis_feat_sizes,
                img_pos_embeds=img_pos_embeds,
                include_coordinate_encodings=self.include_coordinate_encodings,
            )

            txt_feats = backbone_out["language_features"][:, find_input.text_ids].to(geo_feats.dtype)
            txt_mask = backbone_out["language_mask"][find_input.text_ids]
            prompt = torch.cat([txt_feats, geo_feats], dim=0)
            prompt_mask = torch.cat([txt_mask, geo_mask], dim=1)

        return prompt.detach().cpu(), prompt_mask.detach().cpu()

    def _encode_points_with_optional_coords(
        self,
        points: torch.Tensor,
        points_mask: torch.Tensor,
        points_labels: torch.Tensor,
        img_feats: List[torch.Tensor],
        img_sizes: List[Tuple[int, int]],
        img_pos_embeds: List[torch.Tensor],
        include_coordinate_encodings: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Official geometry encoder point path, with coordinate terms optional."""

        geo = self.model.geometry_encoder
        seq_first_img_feats = img_feats[-1]
        seq_first_img_pos_embeds = img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(seq_first_img_feats)

        cur_img_feat = img_feats[-1]
        cur_img_feat = geo.img_pre_norm(cur_img_feat)
        h, w = img_sizes[-1]
        batch_size, channels = cur_img_feat.shape[-2:]
        cur_img_feat = cur_img_feat.permute(1, 2, 0).view(batch_size, channels, h, w)

        points_embed = None
        n_points = points.shape[0]
        if include_coordinate_encodings and geo.points_direct_project is not None:
            points_embed = geo.points_direct_project(points)

        if geo.points_pool_project is None:
            raise RuntimeError("Official SAM3 geometry encoder was built without point pooling.")
        grid = (points.transpose(0, 1).unsqueeze(2) * 2.0) - 1.0
        sampled = torch.nn.functional.grid_sample(cur_img_feat, grid, align_corners=False)
        sampled = sampled.squeeze(-1).permute(2, 0, 1)
        pooled = geo.points_pool_project(sampled)
        points_embed = pooled if points_embed is None else points_embed + pooled

        if include_coordinate_encodings and geo.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = geo.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(n_points, batch_size, enc_x.shape[-1])
            enc_y = enc_y.view(n_points, batch_size, enc_y.shape[-1])
            pos = geo.points_pos_enc_project(torch.cat([enc_x, enc_y], dim=-1))
            points_embed = points_embed + pos

        final_embeds = geo.label_embed(points_labels.long()) + points_embed
        final_mask = points_mask

        if geo.cls_embed is not None:
            cls = geo.cls_embed.weight.view(1, 1, geo.d_model).repeat(1, batch_size, 1)
            cls_mask = torch.zeros(batch_size, 1, dtype=final_mask.dtype, device=final_mask.device)
            final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, cls, cls_mask)

        if geo.final_proj is not None:
            final_embeds = geo.norm(geo.final_proj(final_embeds))

        if geo.encode is not None:
            for layer in geo.encode:
                final_embeds = layer(
                    tgt=final_embeds,
                    memory=seq_first_img_feats,
                    tgt_key_padding_mask=final_mask,
                    pos=seq_first_img_pos_embeds,
                )
            final_embeds = geo.encode_norm(final_embeds)

        return final_embeds, final_mask

    @torch.inference_mode()
    def generate_detections(
        self,
        image_batch_bchw: torch.Tensor,
        prompt_bundles: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = image_batch_bchw.shape[0]
        find_input = self.make_find_input(batch_size=batch_size)
        with self.autocast_context():
            backbone_out = self.model.backbone.forward_image(image_batch_bchw)
            feature_dtype = backbone_out["backbone_fpn"][-1].dtype
            prompt, prompt_mask = self._pad_prompt_bundles(prompt_bundles, feature_dtype)

            backbone_out, encoder_out, _ = self.model._run_encoder(
                backbone_out=backbone_out,
                find_input=find_input,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }
            out, hs = self.model._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            seg_img_ids = find_input.img_ids
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                seg_img_ids = backbone_out["id_mapping"][seg_img_ids]
            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

            masks = out["pred_masks"]
            boxes_xyxy = out["pred_boxes_xyxy"]
            scores = out["pred_logits"].sigmoid()
            if "presence_logit_dec" in out and out["presence_logit_dec"] is not None:
                scores = scores * out["presence_logit_dec"].sigmoid().unsqueeze(1)
                presence = out["presence_logit_dec"].sigmoid().squeeze(-1)
            else:
                presence = torch.ones(batch_size, device=scores.device, dtype=scores.dtype)
            scores = scores.squeeze(-1)

        return masks, boxes_xyxy, scores, presence

    def _pad_prompt_bundles(
        self,
        prompt_bundles: List[Tuple[torch.Tensor, torch.Tensor]],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(prompt.shape[0] for prompt, _ in prompt_bundles)
        channels = prompt_bundles[0][0].shape[-1]
        batch = len(prompt_bundles)
        prompt = torch.zeros(max_len, batch, channels, device=self.device, dtype=dtype)
        prompt_mask = torch.ones(batch, max_len, device=self.device, dtype=torch.bool)
        for idx, (cur_prompt, cur_mask) in enumerate(prompt_bundles):
            cur_prompt = cur_prompt.to(device=self.device, dtype=dtype)
            cur_mask = cur_mask.to(device=self.device, dtype=torch.bool)
            seq_len = cur_prompt.shape[0]
            prompt[:seq_len, idx] = cur_prompt[:, 0]
            prompt_mask[idx, :seq_len] = cur_mask[0]
        return prompt, prompt_mask


def _mask_bbox(mask_hw: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_hw > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def load_reference_images(
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_views: int = 4,
) -> List[np.ndarray]:
    ref_images: List[np.ndarray] = []
    lookup_ids = [object_id, object_id.upper(), object_id.lower()]
    for ref_id in ref_view_ids:
        if len(ref_images) >= max_views:
            break
        for lookup_id in lookup_ids:
            candidate = reference_dir / f"{lookup_id}_stl_base_{ref_id}.png"
            if candidate.is_file():
                ref_images.append(load_bgr(str(candidate)))
                break
    return ref_images


def build_reference_grid(ref_images: List[np.ndarray], target_height: int) -> np.ndarray:
    tile_h = max(1, target_height // 2)
    tile_w = tile_h
    tiles = []
    for idx in range(4):
        if idx < len(ref_images):
            tile = resize_array(ref_images[idx], (tile_w, tile_h), nearest=False)
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles.append(tile)
    grid = np.concatenate(
        [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)],
        axis=0,
    )
    if grid.shape[0] < target_height:
        pad = np.zeros((target_height - grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        grid = np.concatenate([grid, pad], axis=0)
    return grid[:target_height]


def save_mask_triptych(
    image_bgr: np.ndarray,
    mask_preds_nhw: torch.Tensor,
    detection_scores_n: torch.Tensor,
    gt_masks: Optional[List[np.ndarray]],
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    image_name: str,
    output_path: str,
) -> None:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else bgr_to_rgb(image_bgr)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f"image={image_name} | object_id={object_id}", fontsize=12)

    ref_grid = build_reference_grid(load_reference_images(object_id, reference_dir, ref_view_ids), h)
    axes[0].imshow(cv2.cvtColor(ref_grid, cv2.COLOR_BGR2RGB) if cv2 is not None else bgr_to_rgb(ref_grid))
    axes[1].imshow(image_rgb)
    axes[2].imshow(image_rgb)

    palette = plt.get_cmap("tab10").colors
    if gt_masks:
        for idx, gt_mask in enumerate(gt_masks):
            color = palette[(idx + 1) % len(palette)]
            mask_resized = resize_mask(gt_mask.astype(np.float32), (h, w))
            if mask_resized.max() <= 0:
                continue
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., :3] = color
            rgba[..., 3] = (mask_resized > 0) * 0.5
            axes[1].imshow(rgba)
            bbox = _mask_bbox(mask_resized)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                axes[1].add_patch(Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1, edgecolor=color, facecolor="none"))
                axes[1].text(x0, max(0, y0 - 4), f"(gt={idx})", color=color, fontsize=10)

    scores_cpu = detection_scores_n.detach().float().cpu().numpy() if detection_scores_n.numel() else np.array([])
    top_idx = np.argsort(scores_cpu)[-min(2, scores_cpu.size) :][::-1] if scores_cpu.size else []
    for rank, idx in enumerate(top_idx):
        color = palette[(rank + 1) % len(palette)]
        mask_bin = (mask_preds_nhw[int(idx)] > 0).detach().float().cpu().numpy()
        if mask_bin.max() <= 0:
            continue
        mask_resized = resize_mask(mask_bin, (h, w))
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., :3] = color
        rgba[..., 3] = mask_resized * 0.5
        axes[2].imshow(rgba)
        bbox = _mask_bbox(mask_resized)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            axes[2].add_patch(Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1, edgecolor=color, facecolor="none"))
            axes[2].text(x0, max(0, y0 - 4), f"(id={int(idx)}, prob={scores_cpu[int(idx)]:.2f})", color=color, fontsize=10)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=0.1, rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_mask_iou(pred_mask_hw: torch.Tensor, gt_mask_hw: torch.Tensor) -> torch.Tensor:
    pred_bin = pred_mask_hw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum().clamp_min(1)
    return intersection.float() / union.float()


def match_masks_to_gts(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    if not pred_masks or not gt_masks:
        return []
    pairs = []
    for p_idx, pred in enumerate(pred_masks):
        for g_idx, gt in enumerate(gt_masks):
            iou_val = float(compute_mask_iou(pred, gt).item())
            if iou_val >= iou_threshold:
                pairs.append((iou_val, p_idx, g_idx))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = [False] * len(pred_masks)
    matched_gt = [False] * len(gt_masks)
    matches = []
    for iou_val, p_idx, g_idx in pairs:
        if matched_pred[p_idx] or matched_gt[g_idx]:
            continue
        matched_pred[p_idx] = True
        matched_gt[g_idx] = True
        matches.append((p_idx, g_idx, iou_val))
    return matches


def compute_pq_stats(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> Tuple[float, int, int, int]:
    if pred_scores is not None and pred_masks:
        pred_masks = [mask for idx, mask in enumerate(pred_masks) if float(pred_scores[idx].item()) >= score_threshold]
    if not pred_masks and not gt_masks:
        return 0.0, 0, 0, 0
    if not pred_masks:
        return 0.0, 0, 0, len(gt_masks)
    if not gt_masks:
        return 0.0, 0, len(pred_masks), 0
    matches = match_masks_to_gts(pred_masks, gt_masks, iou_threshold=iou_threshold)
    return sum(match[2] for match in matches), len(matches), len(pred_masks) - len(matches), len(gt_masks) - len(matches)


def update_pq_accumulators(
    pq_stats: Dict[float, Dict[str, float]],
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
    iou_threshold: float,
) -> None:
    for score_threshold, stats in pq_stats.items():
        sum_iou, tp, fp, fn = compute_pq_stats(
            pred_masks,
            gt_masks,
            pred_scores=pred_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        stats["sum_iou"] += sum_iou
        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn


def build_detection_record(
    object_id: str,
    frame_id: str,
    pred_scores: Optional[torch.Tensor],
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    iou_threshold: float,
    top_k: int = 5,
) -> Dict[str, object]:
    matches = match_masks_to_gts(pred_masks, gt_masks, iou_threshold=iou_threshold)
    match_by_pred = {p_idx: (g_idx, iou_val) for p_idx, g_idx, iou_val in matches}
    scores_cpu = pred_scores.detach().float().cpu() if pred_scores is not None else None
    record: Dict[str, object] = {
        "object_id": object_id,
        "frame_id": frame_id,
        "num_gt": len(gt_masks),
        "num_pred": len(pred_masks),
    }
    for idx in range(top_k):
        record[f"mask{idx + 1}_score"] = float(scores_cpu[idx].item()) if scores_cpu is not None and idx < scores_cpu.numel() else None
        record[f"mask{idx + 1}_matched_gt"] = int(match_by_pred[idx][0]) if idx in match_by_pred else None
    return record


def apply_mask_nms(
    box_preds_n4: torch.Tensor,
    mask_preds_nhw: torch.Tensor,
    det_scores_n: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if det_scores_n.numel() == 0:
        return box_preds_n4, mask_preds_nhw, det_scores_n
    score_order = torch.argsort(det_scores_n, descending=True)
    if iou_threshold <= 0:
        return box_preds_n4[score_order], mask_preds_nhw[score_order], det_scores_n[score_order]

    masks_bin = mask_preds_nhw > 0
    keep: List[int] = []
    for idx in score_order.tolist():
        if not keep:
            keep.append(int(idx))
            continue
        cand = masks_bin[idx]
        cand_area = cand.sum().clamp_min(1)
        suppress = False
        for kept_idx in keep:
            kept = masks_bin[kept_idx]
            inter = (cand & kept).sum()
            union = (cand | kept).sum().clamp_min(1)
            kept_area = kept.sum().clamp_min(1)
            iou = inter.float() / union.float()
            overlap_cand = inter.float() / cand_area.float()
            overlap_kept = inter.float() / kept_area.float()
            if float(iou.item()) > iou_threshold or float(overlap_cand.item()) >= 0.95 or float(overlap_kept.item()) >= 0.95:
                suppress = True
                break
        if not suppress:
            keep.append(int(idx))

    if not keep:
        return box_preds_n4[:0], mask_preds_nhw[:0], det_scores_n[:0]
    keep_tensor = torch.as_tensor(keep, device=det_scores_n.device, dtype=torch.long)
    return box_preds_n4[keep_tensor], mask_preds_nhw[keep_tensor], det_scores_n[keep_tensor]


def _replace_by_lut(key: str, replacements: List[Tuple[str, str]]) -> str:
    for source, target in replacements:
        if source in key:
            return key.replace(source, target)
    return key


def _map_muggled_image_exemplar_fusion_key(key: str) -> Optional[str]:
    if not key.startswith("fusion_layers."):
        return None
    key = key.replace("fusion_layers.", "layers.", 1)
    key = _replace_by_lut(
        key,
        [
            ("img_selfattn.norm", "norm1"),
            ("img_selfattn.attn", "self_attn"),
            ("img_crossattn.norm", "norm2"),
            ("img_crossattn.attn", "cross_attn_image"),
            ("img_mlp.mlp.0", "norm3"),
            ("img_mlp.mlp.1", "linear1"),
            ("img_mlp.mlp.3", "linear2"),
        ],
    )
    return f"transformer.encoder.{key}"


def _map_muggled_exemplar_detector_key(key: str, tensor: torch.Tensor) -> Tuple[Optional[str], torch.Tensor]:
    decoder_prefix = "transformer.decoder."
    scoring_prefix = "dot_prod_scoring."

    if key.startswith("fusion_layers."):
        key = key.replace("fusion_layers.", "layers.", 1)
        key = _replace_by_lut(
            key,
            [
                ("query_selfattn.attn", "self_attn"),
                ("query_selfattn.norm", "norm2"),
                ("exemplar_crossattn.attn", "ca_text"),
                ("exemplar_crossattn.norm", "catext_norm"),
                ("image_crossattn.attn", "cross_attn"),
                ("image_crossattn.norm", "norm1"),
                ("query_mlp.mlp.0", "linear1"),
                ("query_mlp.mlp.2", "linear2"),
                ("query_mlp.norm", "norm3"),
            ],
        )
        return f"{decoder_prefix}{key}", tensor

    if key.startswith("out_norm_detections."):
        return f"{decoder_prefix}{key.replace('out_norm_detections.', 'norm.', 1)}", tensor

    if key.startswith("mlp_presence_score.layers.0."):
        return f"{decoder_prefix}{key.replace('mlp_presence_score.layers.0', 'presence_token_out_norm', 1)}", tensor
    if key.startswith("mlp_presence_score.layers."):
        key = _replace_by_lut(
            key,
            [
                ("mlp_presence_score.layers.1", "presence_token_head.layers.0"),
                ("mlp_presence_score.layers.3", "presence_token_head.layers.1"),
                ("mlp_presence_score.layers.5", "presence_token_head.layers.2"),
            ],
        )
        return f"{decoder_prefix}{key}", tensor

    key_after_mlp = _replace_by_lut(
        key,
        [
            ("mlp_detection_to_box.0", "bbox_embed.layers.0"),
            ("mlp_detection_to_box.2", "bbox_embed.layers.1"),
            ("mlp_detection_to_box.4", "bbox_embed.layers.2"),
            ("mlp_box_relpos_dx.0", "boxRPB_embed_x.layers.0"),
            ("mlp_box_relpos_dx.2", "boxRPB_embed_x.layers.1"),
            ("mlp_box_relpos_dy.0", "boxRPB_embed_y.layers.0"),
            ("mlp_box_relpos_dy.2", "boxRPB_embed_y.layers.1"),
            ("mlp_detection_posenc.0", "ref_point_head.layers.0"),
            ("mlp_detection_posenc.2", "ref_point_head.layers.1"),
        ],
    )
    if key_after_mlp != key:
        return f"{decoder_prefix}{key_after_mlp}", tensor

    if key == "detection_tokens":
        return f"{decoder_prefix}query_embed.weight", tensor.squeeze(0)
    if key == "anchor_boxes_cxcywh":
        return f"{decoder_prefix}reference_points.weight", tensor.squeeze(0)
    if key == "presence_token":
        return f"{decoder_prefix}presence_token.weight", tensor.squeeze(0)

    if key.startswith("detection_scoring."):
        key = key.replace("detection_scoring.", "", 1)
        key = _replace_by_lut(
            key,
            [
                ("exemplar_mlp.mlp.0", "prompt_mlp.layers.0"),
                ("exemplar_mlp.mlp.2", "prompt_mlp.layers.1"),
                ("exemplar_mlp.norm", "prompt_mlp.out_norm"),
                ("exemplar_proj", "prompt_proj"),
                ("detection_token_proj", "hs_proj"),
            ],
        )
        return f"{scoring_prefix}{key}", tensor

    return None, tensor


def _map_muggled_exemplar_segmentation_key(key: str) -> Optional[str]:
    key_after_mlp = _replace_by_lut(
        key,
        [
            ("query_mlp.0", "mask_predictor.mask_embed.layers.0"),
            ("query_mlp.2", "mask_predictor.mask_embed.layers.1"),
            ("query_mlp.4", "mask_predictor.mask_embed.layers.2"),
        ],
    )
    if key_after_mlp != key:
        return f"segmentation_head.{key_after_mlp}"

    key_after_cross_attn = _replace_by_lut(
        key,
        [
            ("image_cross_attn.attn", "cross_attend_prompt"),
            ("image_cross_attn.norm", "cross_attn_norm"),
        ],
    )
    if key_after_cross_attn != key:
        return f"segmentation_head.{key_after_cross_attn}"

    key_after_pixel_decoder = _replace_by_lut(
        key,
        [
            ("upscale_x2.postprocess.0", "pixel_decoder.conv_layers.0"),
            ("upscale_x2.postprocess.1", "pixel_decoder.norms.0"),
            ("upscale_x4.postprocess.0", "pixel_decoder.conv_layers.1"),
            ("upscale_x4.postprocess.1", "pixel_decoder.norms.1"),
            ("img_token_proj", "instance_seg_head"),
            ("semantic_proj", "semantic_seg_head"),
        ],
    )
    if key_after_pixel_decoder != key:
        return f"segmentation_head.{key_after_pixel_decoder}"

    return None


def remap_muggled_finetune_state_dict(ckpt: Dict[str, object]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Map the custom eval's three fine-tuned detector modules back to official SAM3 keys."""

    remapped: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []

    for key, tensor in ckpt.get("image_exemplar_fusion", {}).items():
        mapped_key = _map_muggled_image_exemplar_fusion_key(key)
        if mapped_key is None:
            skipped.append(f"image_exemplar_fusion.{key}")
            continue
        remapped[mapped_key] = tensor

    for key, tensor in ckpt.get("exemplar_detector", {}).items():
        mapped_key, mapped_tensor = _map_muggled_exemplar_detector_key(key, tensor)
        if mapped_key is None:
            skipped.append(f"exemplar_detector.{key}")
            continue
        remapped[mapped_key] = mapped_tensor

    for key, tensor in ckpt.get("exemplar_segmentation", {}).items():
        mapped_key = _map_muggled_exemplar_segmentation_key(key)
        if mapped_key is None:
            skipped.append(f"exemplar_segmentation.{key}")
            continue
        remapped[mapped_key] = tensor

    return remapped, skipped


def filter_state_dict_for_model(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    missing_keys: List[str] = []
    shape_mismatches: List[str] = []
    for key, tensor in state_dict.items():
        if key not in model_state:
            missing_keys.append(key)
            continue
        if model_state[key].shape != tensor.shape:
            shape_mismatches.append(f"{key}: ckpt={tuple(tensor.shape)} model={tuple(model_state[key].shape)}")
            continue
        compatible[key] = tensor
    return compatible, missing_keys, shape_mismatches


def find_reference_pair(
    object_id: str,
    reference_dir: Path,
    ref_id: str,
) -> Optional[Tuple[Path, Path]]:
    for lookup_id in [object_id, object_id.upper(), object_id.lower()]:
        stub = f"{lookup_id}_stl_base_{ref_id}"
        image_path = reference_dir / f"{stub}.png"
        mask_path = reference_dir / f"{stub}_mask.png"
        if image_path.is_file() and mask_path.is_file():
            return image_path, mask_path
    return None


def build_exemplar_tokens_for_object(
    adapter: OfficialSam3ImageExemplarAdapter,
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    num_points_approx: int,
    grayscale: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    prompts: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for ref_id in ref_view_ids:
        pair = find_reference_pair(object_id, reference_dir, ref_id)
        if pair is None:
            continue
        ref_img_path, ref_mask_path = pair
        try:
            ref_image = load_bgr(str(ref_img_path))
            ref_mask = load_mask_gray(str(ref_mask_path))
        except FileNotFoundError:
            continue
        if grayscale:
            ref_image = apply_grayscale(ref_image)
        if ref_mask.shape[:2] != ref_image.shape[:2]:
            ref_mask = resize_mask(ref_mask, ref_image.shape[:2])
        points = sample_points_from_mask(ref_mask, num_points_approx=num_points_approx)
        if not points:
            continue
        prompt, prompt_mask = adapter.encode_reference_points(ref_image, points, text="visual")
        prompts.append(prompt)
        masks.append(prompt_mask)

    if not prompts:
        return None
    return torch.cat(prompts, dim=0), torch.cat(masks, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official SAM3 with cross-image exemplar prompts.")
    parser.add_argument("--model_path", type=str, default="/home/zhenrant/rendering_prompted_muggled_sam/sam3.pt")
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="/sata1/data/kevin/realworld_datasets/3d_printing_meshes/renders_2442_0316",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        nargs="+",
        default=["/sata1/data/kevin/realworld_datasets/3d_printing_dataset"],
    )
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--num_points_approx", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sub_sample", type=int, default=1)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="outputs_eval_exemplar_official")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument("--image_list", type=str, default="")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--multi_gt_only", action="store_true")
    parser.add_argument(
        "--include_coord_enc",
        action="store_true",
        help="Include official coordinate encodings in reference point tokens. Default matches the custom script.",
    )
    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default="/home/zhenrant/rendering_prompted_muggled_sam/finetune_exemplar/run_20260322_172059/finetune_epoch_034.pth",
        help="Optional official-format state dict or muggled image-exemplar finetune checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_sam3_symbols()
    dataset_roots: List[str] = []
    for item in args.dataset_root:
        dataset_roots.extend([part.strip() for part in item.split(",") if part.strip()])
    if not dataset_roots:
        raise ValueError("No dataset roots provided.")

    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if (args.dtype == "bf16" or (not args.dtype and device.type == "cuda")) else torch.float32

    model_path = args.model_path or None
    model = build_sam3_image_model(
        checkpoint_path=model_path,
        load_from_HF=model_path is None,
        device="cpu",
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    model.to(device=device)
    model.eval()

    if args.finetune_ckpt:
        ckpt = torch.load(args.finetune_ckpt, map_location="cpu", weights_only=True)
        if any(key in ckpt for key in ("image_exemplar_fusion", "exemplar_detector", "exemplar_segmentation")):
            state_dict, skipped = remap_muggled_finetune_state_dict(ckpt)
            state_dict, unknown_keys, shape_mismatches = filter_state_dict_for_model(model, state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded muggled finetune checkpoint remap from {args.finetune_ckpt}: "
                f"loaded={len(state_dict)} skipped={len(skipped)} unknown={len(unknown_keys)} "
                f"shape_mismatch={len(shape_mismatches)} missing_after_load={len(missing)} unexpected={len(unexpected)}"
            )
            if skipped:
                print("Skipped unmapped muggled keys:", skipped[:10])
            if unknown_keys:
                print("Skipped keys missing from official model:", unknown_keys[:10])
            if shape_mismatches:
                print("Skipped shape mismatches:", shape_mismatches[:10])
        else:
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded official finetune checkpoint with missing={len(missing)} unexpected={len(unexpected)}")

    adapter = OfficialSam3ImageExemplarAdapter(
        model=model,
        device=device,
        resolution=args.resolution,
        dtype=dtype,
        include_coordinate_encodings=args.include_coord_enc,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "step_outputs")
    os.makedirs(vis_dir, exist_ok=True)

    all_entries: List[Dict[str, object]] = []
    for root in dataset_roots:
        _, cur_entries = collect_multi_object_samples(root, sub_sample=args.sub_sample)
        all_entries.extend(cur_entries)
    unique_entries: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for entry in all_entries:
        key = (str(entry["frame_id"]), str(entry["object_id"]), str(entry["rgb_path"]), str(entry["inst_path"]))
        unique_entries.setdefault(key, entry)
    all_entries = list(unique_entries.values())

    image_list = parse_image_list(args.image_list)
    if image_list:
        all_entries = [entry for entry in all_entries if os.path.abspath(str(entry["rgb_path"])) in image_list]
        if not all_entries:
            raise RuntimeError("No dataset entries matched --image_list.")
    if args.shuffle:
        random.shuffle(all_entries)

    total_batches_est = max(1, math.ceil(len(all_entries) / args.batch_size))
    if args.max_batches > 0:
        total_batches_est = min(total_batches_est, args.max_batches)
    print("Estimated batches:", total_batches_est, f"(entries={len(all_entries)}, batch_size={args.batch_size})")

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    ref_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    seg_cache: Dict[str, np.ndarray] = {}
    mapping_cache: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}

    total_iou_sum = 0.0
    total_iou_count = 0
    total_correct_count = 0
    object_iou_sum: Dict[str, float] = defaultdict(float)
    object_iou_count: Dict[str, int] = defaultdict(int)
    pq_iou_threshold = 0.5
    pq_score_thresholds = [round(0.10 + 0.01 * idx, 2) for idx in range(89)]
    pq_stats: Dict[float, Dict[str, float]] = {
        thresh: {"sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0} for thresh in pq_score_thresholds
    }

    detection_log_path = os.path.join(args.output_dir, "detection_log_official.json")
    batch_step = 0
    with open(detection_log_path, "w", encoding="utf-8") as detection_log:
        for start in range(0, len(all_entries), args.batch_size):
            subset = all_entries[start : start + args.batch_size]
            prepared: List[Dict[str, object]] = []
            for entry in subset:
                obj_id = str(entry["object_id"])
                try:
                    image_bgr = load_bgr(str(entry["rgb_path"]))
                except FileNotFoundError:
                    continue
                if args.grayscale:
                    image_bgr = apply_grayscale(image_bgr)

                mapping_path = Path(str(entry["inst_path"])).with_name(
                    f"instance_segmentation_mapping_{entry['frame_id']}.json"
                )
                try:
                    gt_masks = load_instance_masks_for_object(
                        str(entry["inst_path"]),
                        str(mapping_path),
                        obj_id,
                        seg_cache=seg_cache,
                        mapping_cache=mapping_cache,
                    )
                except FileNotFoundError:
                    continue
                if not gt_masks or (args.multi_gt_only and len(gt_masks) < 2):
                    continue

                if obj_id not in ref_cache:
                    exemplar = build_exemplar_tokens_for_object(
                        adapter=adapter,
                        object_id=obj_id,
                        reference_dir=reference_dir,
                        ref_view_ids=ref_view_ids,
                        num_points_approx=args.num_points_approx,
                        grayscale=args.grayscale,
                    )
                    if exemplar is None:
                        continue
                    ref_cache[obj_id] = exemplar

                prepared.append(
                    {
                        "object_id": obj_id,
                        "frame_id": str(entry["frame_id"]),
                        "rgb_path": str(entry["rgb_path"]),
                        "image_bgr": image_bgr,
                        "gt_masks": gt_masks,
                        "exemplar_ref": ref_cache[obj_id],
                    }
                )

            if not prepared:
                continue

            vis_target_idx = None
            if args.vis_every > 0 and (batch_step % args.vis_every) == 0:
                vis_target_idx = random.randrange(len(prepared))

            img_batch = torch.cat([adapter.preprocess_bgr(item["image_bgr"]) for item in prepared], dim=0)
            prompt_bundles = [item["exemplar_ref"] for item in prepared]

            t0 = time.time()
            masks, boxes, scores, pres = adapter.generate_detections(img_batch, prompt_bundles)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.time()
            display_step = batch_step + 1
            print(f"step {display_step}/{total_batches_est} detection time: {t1 - t0:.3f}s")

            batch_ious: List[float] = []
            batch_correct = 0
            for local_idx, entry in enumerate(prepared):
                local_scores = scores[local_idx]
                local_masks = masks[local_idx]
                local_boxes = boxes[local_idx]
                if args.det_filter > 1e-3:
                    keep = local_scores > args.det_filter
                    local_scores = local_scores[keep]
                    local_masks = local_masks[keep]
                    local_boxes = local_boxes[keep]

                gt_down_list = build_gt_down_list(
                    entry["gt_masks"],
                    (args.resolution, args.resolution),
                    masks.shape[-2:],
                    device,
                )
                if local_scores.numel() == 0:
                    update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                    continue

                boxes_nms, masks_nms, scores_nms = apply_mask_nms(
                    local_boxes,
                    local_masks,
                    local_scores,
                    iou_threshold=args.nms_iou,
                )
                if scores_nms.numel() == 0:
                    update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                    continue

                pred_masks_list = [(masks_nms[k] > 0) for k in range(masks_nms.shape[0])]
                update_pq_accumulators(
                    pq_stats,
                    pred_masks_list,
                    gt_down_list,
                    pred_scores=scores_nms,
                    iou_threshold=pq_iou_threshold,
                )

                detection_log.write(
                    json.dumps(
                        build_detection_record(
                            str(entry["object_id"]),
                            str(entry["frame_id"]),
                            scores_nms,
                            pred_masks_list,
                            gt_down_list,
                            iou_threshold=pq_iou_threshold,
                            top_k=5,
                        )
                    )
                    + "\n"
                )

                best_iou = 0.0
                for gt_down in gt_down_list:
                    best_iou = max(best_iou, float(compute_mask_iou(masks_nms[0], gt_down).item()))
                batch_ious.append(best_iou)
                total_iou_sum += best_iou
                total_iou_count += 1
                if best_iou > 0.5:
                    batch_correct += 1
                    total_correct_count += 1
                obj_id = str(entry["object_id"])
                object_iou_sum[obj_id] += best_iou
                object_iou_count[obj_id] += 1

                if vis_target_idx is not None and local_idx == vis_target_idx:
                    out_path = os.path.join(vis_dir, f"step_{batch_step:06d}.png")
                    save_mask_triptych(
                        entry["image_bgr"],
                        masks_nms,
                        scores_nms,
                        entry["gt_masks"],
                        object_id=str(entry["object_id"]),
                        reference_dir=reference_dir,
                        ref_view_ids=ref_view_ids,
                        image_name=str(entry["rgb_path"]),
                        output_path=out_path,
                    )

            if batch_ious:
                avg_iou = sum(batch_ious) / len(batch_ious)
                correct_rate = batch_correct / len(batch_ious)
                print(
                    f"step {display_step}/{total_batches_est} avg_iou={avg_iou:.4f} "
                    f"correct_rate={correct_rate:.3f} samples={len(batch_ious)}"
                )
            batch_step += 1
            if args.max_batches > 0 and batch_step >= args.max_batches:
                break

    if total_iou_count > 0:
        print(
            f"overall_avg_iou={total_iou_sum / total_iou_count:.4f} "
            f"correct_rate={total_correct_count / total_iou_count:.3f} samples={total_iou_count}"
        )
    if object_iou_count:
        print("per_object_iou:")
        for obj_id in sorted(object_iou_count):
            print(f"  {obj_id}: avg_iou={object_iou_sum[obj_id] / object_iou_count[obj_id]:.4f} samples={object_iou_count[obj_id]}")
    for score_threshold in sorted(pq_stats):
        stats = pq_stats[score_threshold]
        denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
        pq = stats["sum_iou"] / denom if denom > 0 else 0.0
        print(
            f"PQ@score>={score_threshold:.2f}={pq:.4f} "
            f"tp={int(stats['tp'])} fp={int(stats['fp'])} fn={int(stats['fn'])}"
        )


if __name__ == "__main__":
    main()
