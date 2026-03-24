#!/usr/bin/env python3
"""
Convert COLMAP sparse reconstruction to S3Gaussian roadside format.

Input (per scene):
    scene_dir/
        images/          — original images (any naming)
        sparse/0/        — cameras.bin, images.bin, points3D.bin

Output (in-place, adds to scene_dir):
    scene_dir/
        images/          — renamed to {frame:03d}_{cam_id}.{ext}
        intrinsics/      — {cam_id}.txt (fx fy cx cy)
        extrinsics/      — {cam_id}.txt (4x4 camera-to-world)
        ego_pose/        — 000.txt (identity)
        lidar/           — 000.bin (Nx10 float32, from points3D)
        frame_info.json

Usage:
    # Single scene
    python scripts/roadside/convert_colmap_to_roadside.py --scene_dir /path/to/scene003_far

    # Batch: all subdirs under a root
    python scripts/roadside/convert_colmap_to_roadside.py --data_root /path/to/car_road
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, qvec2rotmat,
    read_extrinsics_text, read_intrinsics_text,
    read_points3D_text,
)


def convert_scene(scene_dir):
    """Convert one COLMAP scene to roadside format."""
    scene_dir = os.path.abspath(scene_dir)
    sparse_dir = os.path.join(scene_dir, "sparse", "0")

    # Skip if already converted
    if os.path.exists(os.path.join(scene_dir, "frame_info.json")):
        print(f"  SKIP (already converted): {scene_dir}")
        return True

    if not os.path.isdir(sparse_dir):
        print(f"  SKIP (no sparse/0/): {scene_dir}")
        return False

    print(f"  Converting: {scene_dir}")

    # --- Read COLMAP data ---
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))

    try:
        pts3d_xyz, pts3d_rgb, _ = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))
    except Exception:
        pts3d_xyz, pts3d_rgb, _ = read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))

    # Sort images by name for consistent camera ordering
    sorted_images = sorted(cam_extrinsics.values(), key=lambda x: x.name)
    num_cameras = len(sorted_images)

    # --- Create output directories ---
    for subdir in ["intrinsics", "extrinsics", "ego_pose", "lidar"]:
        os.makedirs(os.path.join(scene_dir, subdir), exist_ok=True)

    # Temporary dir for renamed images
    images_dir = os.path.join(scene_dir, "images")
    images_tmp = os.path.join(scene_dir, "images_tmp")

    # --- Rename images to {frame:03d}_{cam_id}.{ext} ---
    # Single frame (frame=000), camera IDs = 0,1,2,...
    os.rename(images_dir, images_tmp)
    os.makedirs(images_dir, exist_ok=True)

    original_sizes = []
    for cam_idx, img_data in enumerate(sorted_images):
        old_name = img_data.name
        old_path = os.path.join(images_tmp, old_name)
        ext = os.path.splitext(old_name)[1]  # .jpg, .png, etc.
        new_name = f"000_{cam_idx}{ext}"
        new_path = os.path.join(images_dir, new_name)

        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
        else:
            # Try finding in subdirectories
            for root, dirs, files in os.walk(images_tmp):
                if old_name in files:
                    shutil.copy2(os.path.join(root, old_name), new_path)
                    break

        # Get image size from COLMAP intrinsics
        intr = cam_intrinsics[img_data.camera_id]
        original_sizes.append([intr.height, intr.width])

    # Clean up temp dir
    shutil.rmtree(images_tmp)

    # --- Write intrinsics ---
    for cam_idx, img_data in enumerate(sorted_images):
        intr = cam_intrinsics[img_data.camera_id]
        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        elif intr.model == "SIMPLE_RADIAL":
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model == "RADIAL":
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        np.savetxt(
            os.path.join(scene_dir, "intrinsics", f"{cam_idx}.txt"),
            np.array([fx, fy, cx, cy]),
            fmt="%.10f",
        )

    # --- Write extrinsics (camera-to-world 4x4) ---
    for cam_idx, img_data in enumerate(sorted_images):
        # COLMAP stores world-to-camera: R (from qvec) and t
        R_w2c = qvec2rotmat(img_data.qvec)  # 3x3
        t_w2c = np.array(img_data.tvec)       # 3

        # Build w2c 4x4
        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c

        # Invert to get c2w (camera-to-world)
        c2w = np.linalg.inv(w2c)

        np.savetxt(
            os.path.join(scene_dir, "extrinsics", f"{cam_idx}.txt"),
            c2w,
            fmt="%.10f",
        )

    # --- Write ego_pose (identity for single frame static) ---
    np.savetxt(
        os.path.join(scene_dir, "ego_pose", "000.txt"),
        np.eye(4),
        fmt="%.10f",
    )

    # --- Write lidar bin (Nx10 float32 from COLMAP points3D) ---
    num_points = len(pts3d_xyz)
    lidar_data = np.zeros((num_points, 10), dtype=np.float32)
    # origins = 0 (columns 0-2)
    lidar_data[:, 3:6] = pts3d_xyz.astype(np.float32)  # points (columns 3-5)
    # columns 6-8 unused, column 9 = laser_id = 0
    lidar_data.tofile(os.path.join(scene_dir, "lidar", "000.bin"))

    # --- Write frame_info.json ---
    frame_info = {
        "scene_type": "roadside",
        "num_cameras": num_cameras,
        "original_sizes": original_sizes,
        "coordinate_system": "colmap",
        "num_points": num_points,
    }
    with open(os.path.join(scene_dir, "frame_info.json"), "w") as f:
        json.dump(frame_info, f, indent=2)

    print(f"  Done: {num_cameras} cameras, {num_points} points")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP data to S3Gaussian roadside format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_dir", type=str, help="Path to a single scene directory")
    group.add_argument("--data_root", type=str, help="Path to root containing multiple scene directories")
    args = parser.parse_args()

    if args.scene_dir:
        convert_scene(args.scene_dir)
    else:
        scenes = sorted([
            os.path.join(args.data_root, d)
            for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d, "sparse", "0"))
        ])
        print(f"Found {len(scenes)} scenes with COLMAP data")
        success, skip, fail = 0, 0, 0
        for scene in scenes:
            result = convert_scene(scene)
            if result:
                success += 1
            else:
                fail += 1
        print(f"\nDone: {success} converted, {fail} failed")


if __name__ == "__main__":
    main()
