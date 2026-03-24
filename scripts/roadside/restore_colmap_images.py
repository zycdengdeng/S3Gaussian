#!/usr/bin/env python3
"""
Restore original COLMAP image filenames after convert_colmap_to_roadside.py
renamed them to 000_X.{ext} format.

Reads sparse/0/images.bin to get original filenames, then renames back.
Also removes generated roadside files (frame_info.json, intrinsics/, extrinsics/, ego_pose/, lidar/).
"""
import os
import sys
import json
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text


def restore_scene(scene_dir):
    scene_dir = os.path.abspath(scene_dir)
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    images_dir = os.path.join(scene_dir, "images")

    if not os.path.exists(os.path.join(scene_dir, "frame_info.json")):
        print(f"  SKIP (not converted): {os.path.basename(scene_dir)}")
        return

    # Read original filenames from COLMAP
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))

    sorted_images = sorted(cam_extrinsics.values(), key=lambda x: x.name)

    # Rename 000_X.{ext} back to original names
    for cam_idx, img_data in enumerate(sorted_images):
        original_name = img_data.name
        ext = os.path.splitext(original_name)[1]
        renamed = f"000_{cam_idx}{ext}"
        renamed_path = os.path.join(images_dir, renamed)
        original_path = os.path.join(images_dir, original_name)

        if os.path.exists(renamed_path) and not os.path.exists(original_path):
            os.rename(renamed_path, original_path)

    # Remove generated roadside files
    for name in ["frame_info.json", "ds-points3d.ply"]:
        path = os.path.join(scene_dir, name)
        if os.path.exists(path):
            os.remove(path)

    for subdir in ["intrinsics", "extrinsics", "ego_pose", "lidar"]:
        path = os.path.join(scene_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)

    print(f"  Restored: {os.path.basename(scene_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    scenes = sorted([
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])

    print(f"Restoring {len(scenes)} scenes...")
    for scene in scenes:
        restore_scene(scene)
    print("Done.")


if __name__ == "__main__":
    main()
