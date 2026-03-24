#!/usr/bin/env python3
"""
Convert COLMAP sparse reconstruction to S3Gaussian roadside format.

IMPORTANT: This script creates a NEW output directory with symlinks to
original images. It does NOT modify the source data.

Input (per scene):
    scene_dir/
        images/          — original images (any naming)
        sparse/0/        — cameras.bin, images.bin, points3D.bin

Output (new directory):
    output_dir/scene_name/
        images/          — symlinks: {frame:03d}_{cam_id}.{ext} → original
        intrinsics/      — {cam_id}.txt (fx fy cx cy)
        extrinsics/      — {cam_id}.txt (4x4 camera-to-world)
        ego_pose/        — 000.txt (identity)
        lidar/           — 000.bin (Nx10 float32, from points3D)
        frame_info.json

Usage:
    # Batch: all subdirs under a root
    python scripts/roadside/convert_colmap_to_roadside.py \
        --data_root /path/to/car_road \
        --output_root /path/to/roadside_data
"""
import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from collections import namedtuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, qvec2rotmat,
    read_extrinsics_text, read_intrinsics_text,
    read_points3D_text,
)


def read_points3D_binary_raw(path_to_model_file):
    """
    Read points3D.bin directly, handling the case where the standard
    reader returns empty results.
    Returns xyz (Nx3) and rgb (Nx3) arrays.
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("Q", fid.read(8))[0]
            xyz = struct.unpack("ddd", fid.read(24))
            rgb = struct.unpack("BBB", fid.read(3))
            error = struct.unpack("d", fid.read(8))[0]
            track_length = struct.unpack("Q", fid.read(8))[0]
            track_elems = struct.unpack("ii" * track_length,
                                        fid.read(8 * track_length))
            points3D[point3D_id] = {
                "xyz": np.array(xyz),
                "rgb": np.array(rgb),
            }

    if len(points3D) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    xyz = np.stack([p["xyz"] for p in points3D.values()])
    rgb = np.stack([p["rgb"] for p in points3D.values()])
    return xyz, rgb


def convert_scene(scene_dir, output_dir):
    """Convert one COLMAP scene to roadside format in a separate output directory."""
    scene_dir = os.path.abspath(scene_dir)
    output_dir = os.path.abspath(output_dir)
    sparse_dir = os.path.join(scene_dir, "sparse", "0")

    # Skip if already converted
    if os.path.exists(os.path.join(output_dir, "frame_info.json")):
        print(f"  SKIP (already converted): {os.path.basename(scene_dir)}")
        return True

    if not os.path.isdir(sparse_dir):
        print(f"  SKIP (no sparse/0/): {os.path.basename(scene_dir)}")
        return False

    print(f"  Converting: {os.path.basename(scene_dir)}")

    # --- Read COLMAP data ---
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))

    # Read points3D - try standard reader first, fall back to raw reader
    pts3d_bin = os.path.join(sparse_dir, "points3D.bin")
    pts3d_txt = os.path.join(sparse_dir, "points3D.txt")
    pts3d_xyz = np.zeros((0, 3))
    pts3d_rgb = np.zeros((0, 3))

    if os.path.exists(pts3d_bin) and os.path.getsize(pts3d_bin) > 8:
        try:
            pts3d_xyz, pts3d_rgb, _ = read_points3D_binary(pts3d_bin)
        except Exception:
            try:
                pts3d_xyz, pts3d_rgb = read_points3D_binary_raw(pts3d_bin)
            except Exception as e:
                print(f"    Warning: could not read points3D.bin: {e}")
    elif os.path.exists(pts3d_txt):
        try:
            pts3d_xyz, pts3d_rgb, _ = read_points3D_text(pts3d_txt)
        except Exception as e:
            print(f"    Warning: could not read points3D.txt: {e}")

    # Sort images by name for consistent camera ordering
    sorted_images = sorted(cam_extrinsics.values(), key=lambda x: x.name)
    num_cameras = len(sorted_images)

    # --- Create output directories ---
    for subdir in ["images", "intrinsics", "extrinsics", "ego_pose", "lidar"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # --- Create symlinks for images: 000_{cam_id}.{ext} → original ---
    original_sizes = []
    for cam_idx, img_data in enumerate(sorted_images):
        original_name = img_data.name
        original_path = os.path.join(scene_dir, "images", original_name)
        ext = os.path.splitext(original_name)[1]
        link_name = f"000_{cam_idx}{ext}"
        link_path = os.path.join(output_dir, "images", link_name)

        # Create symlink (remove if exists)
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(original_path, link_path)

        # Get image size from COLMAP intrinsics
        intr = cam_intrinsics[img_data.camera_id]
        original_sizes.append([intr.height, intr.width])

    # --- Write intrinsics ---
    for cam_idx, img_data in enumerate(sorted_images):
        intr = cam_intrinsics[img_data.camera_id]
        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        np.savetxt(
            os.path.join(output_dir, "intrinsics", f"{cam_idx}.txt"),
            np.array([fx, fy, cx, cy]),
            fmt="%.10f",
        )

    # --- Write extrinsics (camera-to-world 4x4) ---
    for cam_idx, img_data in enumerate(sorted_images):
        R_w2c = qvec2rotmat(img_data.qvec)
        t_w2c = np.array(img_data.tvec)
        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        c2w = np.linalg.inv(w2c)

        np.savetxt(
            os.path.join(output_dir, "extrinsics", f"{cam_idx}.txt"),
            c2w,
            fmt="%.10f",
        )

    # --- Write ego_pose (identity for single frame static) ---
    np.savetxt(
        os.path.join(output_dir, "ego_pose", "000.txt"),
        np.eye(4),
        fmt="%.10f",
    )

    # --- Write lidar bin (Nx10 float32 from COLMAP points3D) ---
    num_points = len(pts3d_xyz)
    lidar_data = np.zeros((num_points, 10), dtype=np.float32)
    if num_points > 0:
        lidar_data[:, 3:6] = pts3d_xyz.astype(np.float32)
    lidar_data.tofile(os.path.join(output_dir, "lidar", "000.bin"))

    # --- Write frame_info.json ---
    frame_info = {
        "scene_type": "roadside",
        "num_cameras": num_cameras,
        "original_sizes": original_sizes,
        "coordinate_system": "colmap",
        "num_points": num_points,
    }
    with open(os.path.join(output_dir, "frame_info.json"), "w") as f:
        json.dump(frame_info, f, indent=2)

    print(f"    {num_cameras} cameras, {num_points} points")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP data to S3Gaussian roadside format")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root containing multiple COLMAP scene directories")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output root for converted roadside data (separate from source)")
    args = parser.parse_args()

    scenes = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d, "sparse", "0"))
    ])
    print(f"Found {len(scenes)} scenes with COLMAP data")

    success, fail = 0, 0
    for scene_name in scenes:
        scene_dir = os.path.join(args.data_root, scene_name)
        output_dir = os.path.join(args.output_root, scene_name)
        result = convert_scene(scene_dir, output_dir)
        if result:
            success += 1
        else:
            fail += 1
    print(f"\nDone: {success} converted, {fail} failed")


if __name__ == "__main__":
    main()
