"""
Ship detection on satellite images using YOLO11x-OBB with SAHI-style tiling.

Input:  single satellite image (PNG)
Output: .txt file with one detection per line (OBB format)

Each line: class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence
Coordinates are in pixels relative to the original full image.

Usage:
    python detect_ships.py <image_path> [--output <output.txt>]
    python detect_ships.py satellite_images/mumbai/mumbai_2020_03.png
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH = "yolo11x-obb.pt"
SHIP_CLASS_ID = 1
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.3       # NMS IoU threshold for merging overlapping detections

# Tiling config
TILE_SIZE = 640            # YOLO native input size
OVERLAP_RATIO = 0.4        # 40% overlap — catches ships at tile edges
UPSCALE_FACTOR = 3         # 3x bicubic upscale before tiling


# ─── Tiling ──────────────────────────────────────────────────────────────────

def generate_tiles(img_h, img_w, tile_size, overlap_ratio):
    """Generate (x_start, y_start, x_end, y_end) for each tile."""
    stride = int(tile_size * (1 - overlap_ratio))
    tiles = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tiles.append((x_start, y_start, x_end, y_end))
    # Deduplicate
    return list(set(tiles))


# ─── OBB NMS ─────────────────────────────────────────────────────────────────

def polygon_iou(poly1, poly2):
    """Compute IoU between two rotated bounding boxes using OpenCV."""
    poly1 = np.array(poly1, dtype=np.float32).reshape(-1, 2)
    poly2 = np.array(poly2, dtype=np.float32).reshape(-1, 2)

    ret, region = cv2.intersectConvexConvex(poly1, poly2)
    if ret <= 0 or region is None:
        return 0.0

    inter_area = cv2.contourArea(region)
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def obb_nms(detections, iou_threshold):
    """
    Non-Maximum Suppression for oriented bounding boxes.
    detections: list of (xyxyxyxy, confidence, class_id)
    Returns filtered list.
    """
    if not detections:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d[1], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining = []
        for det in detections:
            iou = polygon_iou(best[0], det[0])
            if iou < iou_threshold:
                remaining.append(det)
        detections = remaining

    return keep


# ─── Main ────────────────────────────────────────────────────────────────────

def detect_ships(image_path, output_path=None, visualize=False):
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.with_suffix(".txt")
    else:
        output_path = Path(output_path)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    print(f"Original image: {orig_w}x{orig_h}")

    # Upscale
    if UPSCALE_FACTOR > 1:
        img = cv2.resize(
            img, None,
            fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )
        print(f"Upscaled {UPSCALE_FACTOR}x: {img.shape[1]}x{img.shape[0]}")

    print("No preprocessing — feeding raw upscaled image to YOLO")

    up_h, up_w = img.shape[:2]

    # Generate tiles
    tiles = generate_tiles(up_h, up_w, TILE_SIZE, OVERLAP_RATIO)
    print(f"Tiles: {len(tiles)} ({TILE_SIZE}x{TILE_SIZE}, {int(OVERLAP_RATIO*100)}% overlap)")

    # Load model
    model = YOLO(MODEL_PATH)

    # Run inference on each tile
    all_detections = []
    for i, (x1, y1, x2, y2) in enumerate(tiles):
        tile_img = img[y1:y2, x1:x2]

        results = model.predict(
            tile_img,
            conf=CONF_THRESHOLD,
            verbose=False,
            imgsz=TILE_SIZE,
        )

        for r in results:
            if r.obb is None or len(r.obb) == 0:
                continue

            for j in range(len(r.obb)):
                cls_id = int(r.obb.cls[j])
                if cls_id != SHIP_CLASS_ID:
                    continue

                conf = float(r.obb.conf[j])
                # xyxyxyxy: 4 corner points of the OBB
                points = r.obb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)

                # Offset points from tile coords to full upscaled image coords
                points[:, 0] += x1
                points[:, 1] += y1

                # Scale back to original image coords
                if UPSCALE_FACTOR > 1:
                    points /= UPSCALE_FACTOR

                all_detections.append((points.flatten().tolist(), conf, cls_id))

    print(f"Detections before NMS: {len(all_detections)}")

    # NMS to merge overlapping detections from adjacent tiles
    filtered = obb_nms(all_detections, IOU_THRESHOLD)
    print(f"Detections after NMS:  {len(filtered)}")

    # Write output
    with open(output_path, "w") as f:
        for points, conf, cls_id in filtered:
            coords_str = " ".join(f"{c:.1f}" for c in points)
            f.write(f"{cls_id} {coords_str} {conf:.4f}\n")

    print(f"Ship count: {len(filtered)}")
    print(f"Output: {output_path}")

    # Optional: save annotated image
    if visualize:
        vis_img = cv2.imread(str(image_path))
        for points, conf, cls_id in filtered:
            pts = np.array(points, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.drawContours(vis_img, [pts], -1, (0, 255, 0), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(
                vis_img, f"{conf:.2f}",
                (cx - 15, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
        vis_path = image_path.with_name(image_path.stem + "_detections.png")
        cv2.imwrite(str(vis_path), vis_img)
        print(f"Visualization: {vis_path}")

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ship detection with SAHI tiling + YOLO OBB")
    parser.add_argument("image", help="Path to satellite image")
    parser.add_argument("--output", "-o", help="Output .txt path (default: same name as image)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Save annotated image")
    args = parser.parse_args()

    detect_ships(args.image, args.output, args.visualize)
