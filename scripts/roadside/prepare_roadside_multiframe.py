"""
路侧数据集多帧 -> S3Gaussian Waymo 格式转换脚本

以 road_labels/interpolation_labels/ 中的标注时间戳为时间轴，
对每帧找到对应的 PCD（精确匹配）和图像（最近邻匹配），
并从 3D bounding box 标注生成 dynamic_mask。

用法:
    python scripts/roadside/prepare_roadside_multiframe.py \
        --scene_dir /mnt/car_road_data_TianJin/053_car0402_road0402_t31 \
        --output_dir ./data/roadside/scene_053_multi

    # 限制帧数 (取前 10 帧)
    python scripts/roadside/prepare_roadside_multiframe.py \
        --scene_dir /mnt/car_road_data_TianJin/053_car0402_road0402_t31 \
        --output_dir ./data/roadside/scene_053_multi \
        --max_frames 10

输出目录结构 (Waymo 格式):
    output_dir/
        images/         {frame:03d}_{cam_id}.jpg
        intrinsics/     {cam_id}.txt           (shared across frames)
        extrinsics/     {cam_id}.txt           (shared across frames)
        ego_pose/       {frame:03d}.txt        (all identity for static infra)
        lidar/          {frame:03d}.bin
        dynamic_masks/  {frame:03d}_{cam_id}.png
        frame_info.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ============================================================
# Default camera key mapping
# ============================================================
DEFAULT_PINHOLE_TO_CALIB_KEY = {
    "pinhole0": "3",
    "pinhole1": "6",
    "pinhole2": "9",
    "pinhole3": "0",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert multi-frame roadside data to S3Gaussian Waymo format")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene folder (e.g., .../053_car0402_road0402_t31)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory in S3GS Waymo format")
    parser.add_argument("--calib_json", type=str, default=None,
                        help="Path to calib.json. Auto-detected if not specified.")
    parser.add_argument("--cameras", nargs="+",
                        default=["pinhole0", "pinhole1", "pinhole2", "pinhole3"])
    parser.add_argument("--cam_key_mapping", type=str, default=None,
                        help='JSON string for pinhole->calib key mapping')
    parser.add_argument("--undistort_alpha", type=float, default=0.0)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max number of frames to process (default: all)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process every N-th frame (default: 1 = all frames)")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Override label directory (default: scene_dir/road_labels/interpolation_labels)")
    parser.add_argument("--bbox_expand", type=float, default=1.1,
                        help="Expand bbox by this factor for dynamic mask (default: 1.1)")
    return parser.parse_args()


# ============================================================
# calib.json parsing (same as single-frame script)
# ============================================================

def load_calib_json(calib_path):
    with open(calib_path, 'r') as f:
        return json.load(f)


def get_camera_params(calib, cam_key):
    cam_data = calib["camera"][cam_key]
    is_fish = cam_data.get("isFish", 0) == 1
    K = np.array(cam_data["intri"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(cam_data["distor"], dtype=np.float64)
    vl2c = cam_data["virtualLidarToCam"]
    rvec = np.array(vl2c["rotate"], dtype=np.float64)
    t_w2c = np.array(vl2c["trans"], dtype=np.float64)
    R_w2c, _ = cv2.Rodrigues(rvec)
    return K, dist_coeffs, R_w2c, t_w2c, is_fish


def undistort_and_get_new_K(K, dist_coeffs, img_size, alpha=0.0):
    w, h = img_size
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K_new, (w, h), cv2.CV_32FC1)
    return K_new, roi, map1, map2


# ============================================================
# Auto-detect paths
# ============================================================

def find_calib_json(scene_dir):
    parent_dir = os.path.dirname(os.path.abspath(scene_dir))
    candidates = [
        os.path.join(scene_dir, "calib.json"),
        os.path.join(scene_dir, "support_info", "calib.json"),
        os.path.join(parent_dir, "support_info", "calib.json"),
        os.path.join(parent_dir, "calib.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ============================================================
# Timestamp and nearest-neighbor matching
# ============================================================

def get_label_timestamps(label_dir):
    """Get sorted list of timestamps from label JSON files."""
    timestamps = []
    for f in sorted(os.listdir(label_dir)):
        if f.endswith(".json"):
            ts = f.rsplit(".", 1)[0]
            timestamps.append(ts)
    return sorted(timestamps, key=int)


def get_image_timestamps(cam_dir):
    """Get sorted list of (timestamp_str, filename) from camera directory."""
    entries = []
    if not os.path.exists(cam_dir):
        return entries
    for f in sorted(os.listdir(cam_dir)):
        if f.endswith(".png") or f.endswith(".jpg"):
            name = f.rsplit(".", 1)[0]
            # Handle "cam5_1742877031036.png" format
            parts = name.split("_")
            ts = parts[-1]
            entries.append((int(ts), f))
    return sorted(entries, key=lambda x: x[0])


def find_nearest_image(target_ts_int, image_entries):
    """Find the image with the closest timestamp to target."""
    if not image_entries:
        return None, None
    # Binary search
    ts_list = [e[0] for e in image_entries]
    idx = np.searchsorted(ts_list, target_ts_int)
    # Check neighbors
    best_idx = idx
    best_diff = abs(ts_list[min(idx, len(ts_list) - 1)] - target_ts_int)
    if idx > 0:
        diff = abs(ts_list[idx - 1] - target_ts_int)
        if diff < best_diff:
            best_idx = idx - 1
            best_diff = diff
    if idx < len(ts_list):
        diff = abs(ts_list[idx] - target_ts_int)
        if diff < best_diff:
            best_idx = idx
    best_idx = min(best_idx, len(ts_list) - 1)
    return image_entries[best_idx][1], abs(ts_list[best_idx] - target_ts_int)


# ============================================================
# 3D Bounding Box -> Dynamic Mask
# ============================================================

def load_label_json(label_path):
    """Load 3D bounding box annotations from label JSON."""
    with open(label_path, 'r') as f:
        data = json.load(f)
    return data.get("object", [])


def bbox3d_corners(obj):
    """
    Compute 8 corners of a 3D bounding box in world (virtualLidar) coordinates.

    obj: dict with x, y, z, length, width, height, yaw
    Returns: (8, 3) array of corner positions
    """
    cx, cy, cz = obj["x"], obj["y"], obj["z"]
    l, w, h = obj["length"], obj["width"], obj["height"]
    yaw = obj["yaw"]

    # 8 corners in object frame (centered at origin)
    # length along x, width along y, height along z
    dx = l / 2
    dy = w / 2
    dz = h / 2
    corners = np.array([
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
    ])

    # Rotation around Z axis (yaw only, roll=pitch=0 for most objects)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])

    # Transform to world frame
    corners_world = (R @ corners.T).T + np.array([cx, cy, cz])
    return corners_world


def project_bbox3d_to_mask(objects, K_new, R_w2c, t_w2c, img_w, img_h, expand=1.1):
    """
    Project 3D bounding boxes to 2D image and create binary mask.

    Args:
        objects: list of object dicts from label JSON
        K_new: 3x3 undistorted intrinsic matrix
        R_w2c: 3x3 world-to-camera rotation
        t_w2c: 3x1 world-to-camera translation
        img_w, img_h: image dimensions
        expand: factor to expand projected bbox (account for projection approximation)

    Returns:
        mask: (img_h, img_w) uint8, 255 = dynamic, 0 = static
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for obj in objects:
        corners = bbox3d_corners(obj)  # (8, 3)

        # Transform to camera frame
        corners_cam = (R_w2c @ corners.T).T + t_w2c  # (8, 3)

        # Filter out boxes entirely behind camera
        if np.all(corners_cam[:, 2] <= 0):
            continue

        # Project to image (only points in front of camera)
        valid = corners_cam[:, 2] > 0.1
        if not np.any(valid):
            continue

        corners_valid = corners_cam[valid]
        u = K_new[0, 0] * corners_valid[:, 0] / corners_valid[:, 2] + K_new[0, 2]
        v = K_new[1, 1] * corners_valid[:, 1] / corners_valid[:, 2] + K_new[1, 2]

        # Get 2D bounding rect of projected corners
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        # Expand slightly
        u_center = (u_min + u_max) / 2
        v_center = (v_min + v_max) / 2
        u_half = (u_max - u_min) / 2 * expand
        v_half = (v_max - v_min) / 2 * expand
        u_min = int(max(0, u_center - u_half))
        u_max = int(min(img_w - 1, u_center + u_half))
        v_min = int(max(0, v_center - v_half))
        v_max = int(min(img_h - 1, v_center + v_half))

        if u_min >= u_max or v_min >= v_max:
            continue

        # Fill rectangle in mask
        mask[v_min:v_max + 1, u_min:u_max + 1] = 255

    return mask


# ============================================================
# PCD and LiDAR bin
# ============================================================

def load_pcd_to_numpy(pcd_path):
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pcd.points, dtype=np.float32)
    else:
        return _parse_pcd_manual(pcd_path)


def _parse_pcd_manual(pcd_path):
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('DATA'):
                data_type = line.split()[1]
                break
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    header[parts[0]] = parts[1:]
        num_points = int(header.get('POINTS', ['0'])[0])
        fields = header.get('FIELDS', [])
        types = header.get('TYPE', [])
        sizes = header.get('SIZE', [])
        if data_type == 'ascii':
            points = []
            for _ in range(num_points):
                line = f.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                vals = line.split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
            return np.array(points, dtype=np.float32)
        else:
            dtype_map = {'F': 'f', 'U': 'u', 'I': 'i'}
            dt_list = []
            for field, typ, size in zip(fields, types, sizes):
                dt_char = dtype_map.get(typ, 'f')
                dt_list.append((field, f'{dt_char}{size}'))
            dt = np.dtype(dt_list)
            arr = np.frombuffer(f.read(num_points * dt.itemsize), dtype=dt, count=num_points)
            return np.stack([arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32)


def save_lidar_bin(points, output_path):
    N = len(points)
    lidar_data = np.zeros((N, 10), dtype=np.float32)
    lidar_data[:, 3:6] = points
    lidar_data.tofile(output_path)
    return N


# ============================================================
# Main conversion
# ============================================================

def main():
    args = parse_args()
    scene_dir = args.scene_dir
    output_dir = args.output_dir
    cameras = args.cameras
    num_cameras = len(cameras)

    cam_key_map = json.loads(args.cam_key_mapping) if args.cam_key_mapping else DEFAULT_PINHOLE_TO_CALIB_KEY

    # Find calib.json
    calib_path = args.calib_json or find_calib_json(scene_dir)
    if calib_path is None or not os.path.exists(calib_path):
        print("ERROR: calib.json not found.")
        sys.exit(1)
    calib = load_calib_json(calib_path)
    print(f"Calibration: {calib_path}")

    img_size_wh = calib.get("imgSize", {}).get("notFish", [1280, 720])
    img_w, img_h = img_size_wh[0], img_size_wh[1]

    # Find label directory
    label_dir = args.label_dir or os.path.join(scene_dir, "road_labels", "interpolation_labels")
    if not os.path.exists(label_dir):
        print(f"ERROR: Label directory not found: {label_dir}")
        sys.exit(1)

    # Get all timestamps from labels
    all_timestamps = get_label_timestamps(label_dir)
    print(f"Total label timestamps: {len(all_timestamps)}")

    # Apply skip and max_frames
    all_timestamps = all_timestamps[::args.skip_frames]
    if args.max_frames:
        all_timestamps = all_timestamps[:args.max_frames]
    num_frames = len(all_timestamps)
    print(f"Selected frames: {num_frames} (skip={args.skip_frames})")

    # Build image timestamp lookup for each camera (for nearest-neighbor matching)
    cam_image_entries = {}
    for cam_name in cameras:
        cam_dir = os.path.join(scene_dir, "road", "cameras", cam_name)
        cam_image_entries[cam_name] = get_image_timestamps(cam_dir)
        print(f"  {cam_name}: {len(cam_image_entries[cam_name])} images available")

    # Create output directories
    for subdir in ["images", "intrinsics", "extrinsics", "ego_pose", "lidar", "dynamic_masks"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # ========================================
    # Process camera calibration (once, shared across frames)
    # ========================================
    all_K_new = []
    all_R_w2c = []
    all_t_w2c = []
    all_undistort_maps = []

    for cam_idx, cam_name in enumerate(cameras):
        calib_key = cam_key_map[cam_name]
        K, dist_coeffs, R_w2c, t_w2c, is_fish = get_camera_params(calib, calib_key)
        K_new, roi, map1, map2 = undistort_and_get_new_K(K, dist_coeffs, (img_w, img_h), args.undistort_alpha)

        cam_center = -R_w2c.T @ t_w2c
        print(f"  {cam_name} (cam{calib_key}): K_new fx={K_new[0,0]:.1f} fy={K_new[1,1]:.1f}, "
              f"center=[{cam_center[0]:.1f}, {cam_center[1]:.1f}, {cam_center[2]:.1f}]")

        all_K_new.append(K_new)
        all_R_w2c.append(R_w2c)
        all_t_w2c.append(t_w2c)
        all_undistort_maps.append((map1, map2))

        # Save intrinsic (shared across all frames)
        intrinsic_arr = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
        np.savetxt(os.path.join(output_dir, "intrinsics", f"{cam_idx}.txt"), intrinsic_arr, fmt="%.10f")

        # Save extrinsic: camera-to-world (inverse of w2c)
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c
        T_c2w = np.linalg.inv(T_w2c)
        np.savetxt(os.path.join(output_dir, "extrinsics", f"{cam_idx}.txt"), T_c2w, fmt="%.10f")

    # ========================================
    # Process each frame
    # ========================================
    total_img_time_diffs = []
    missing_pcd = 0
    missing_img = 0

    for frame_idx, ts in enumerate(all_timestamps):
        ts_int = int(ts)
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            print(f"\r  Processing frame {frame_idx + 1}/{num_frames} (ts={ts})...", end="", flush=True)

        # --- ego_pose (identity for static roadside) ---
        np.savetxt(os.path.join(output_dir, "ego_pose", f"{frame_idx:03d}.txt"), np.eye(4), fmt="%.10f")

        # --- LiDAR PCD ---
        pcd_path = os.path.join(scene_dir, "road", "lidar", "merged_pcd", f"{ts}.pcd")
        if os.path.exists(pcd_path):
            points = load_pcd_to_numpy(pcd_path)
            save_lidar_bin(points, os.path.join(output_dir, "lidar", f"{frame_idx:03d}.bin"))
        else:
            missing_pcd += 1

        # --- Load labels for dynamic mask ---
        label_path = os.path.join(label_dir, f"{ts}.json")
        objects = load_label_json(label_path) if os.path.exists(label_path) else []

        # --- Images and dynamic masks for each camera ---
        for cam_idx, cam_name in enumerate(cameras):
            # Find nearest image
            img_file, time_diff = find_nearest_image(ts_int, cam_image_entries[cam_name])
            if img_file is None:
                missing_img += 1
                continue

            if time_diff is not None:
                total_img_time_diffs.append(time_diff)

            # Read, undistort, save image
            img_path = os.path.join(scene_dir, "road", "cameras", cam_name, img_file)
            img = cv2.imread(img_path)
            if img is None:
                missing_img += 1
                continue

            map1, map2 = all_undistort_maps[cam_idx]
            img_undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(
                os.path.join(output_dir, "images", f"{frame_idx:03d}_{cam_idx}.jpg"),
                img_undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            # Generate dynamic mask from 3D bboxes
            if objects:
                mask = project_bbox3d_to_mask(
                    objects, all_K_new[cam_idx], all_R_w2c[cam_idx], all_t_w2c[cam_idx],
                    img_w, img_h, expand=args.bbox_expand
                )
            else:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.imwrite(os.path.join(output_dir, "dynamic_masks", f"{frame_idx:03d}_{cam_idx}.png"), mask)

    print()  # newline after progress

    # ========================================
    # Save frame_info.json
    # ========================================
    frame_info = {
        "scene_type": "roadside",
        "num_cameras": num_cameras,
        "cameras": cameras,
        "cam_key_mapping": cam_key_map,
        "num_frames": num_frames,
        "timestamps": all_timestamps,
        "original_sizes": [[img_h, img_w]] * num_cameras,
        "coordinate_system": "virtualLidar",
        "multi_frame": True,
        "notes": "Multi-frame roadside reconstruction. "
                 "Ego-pose is identity (static infrastructure). "
                 "Dynamic masks generated from 3D bbox annotations."
    }
    with open(os.path.join(output_dir, "frame_info.json"), 'w') as f:
        json.dump(frame_info, f, indent=2, ensure_ascii=False)

    # ========================================
    # Summary
    # ========================================
    avg_diff = np.mean(total_img_time_diffs) if total_img_time_diffs else 0
    max_diff = np.max(total_img_time_diffs) if total_img_time_diffs else 0

    print(f"\n{'=' * 60}")
    print(f"Multi-frame conversion complete!")
    print(f"Output: {output_dir}")
    print(f"  Frames:         {num_frames}")
    print(f"  Cameras:        {num_cameras}")
    print(f"  Total images:   {num_frames * num_cameras}")
    print(f"  Missing PCD:    {missing_pcd}")
    print(f"  Missing images: {missing_img}")
    print(f"  Image-label time offset: avg={avg_diff:.1f}ms, max={max_diff:.1f}ms")
    print(f"  Dynamic masks:  {num_frames * num_cameras} files")
    print(f"{'=' * 60}")
    print(f"\nTo train:")
    print(f"  bash scripts/roadside/run_roadside_multi.sh 0 {output_dir}")


if __name__ == "__main__":
    main()
