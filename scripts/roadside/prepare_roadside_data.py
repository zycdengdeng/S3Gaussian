"""
路侧数据集 -> S3Gaussian Waymo 格式转换脚本

将车路协同数据集中路侧（road）pinhole 相机数据和 LiDAR 数据
转换为 S3Gaussian 所需的 Waymo 目录格式。

支持两种输入布局:
  A) car_road 布局 (天津数据集等):
       data_root/
         support_info/calib.json          # 所有场景共享的标定文件
         {scene_name}/road/cameras/pinhole{0-3}/cam{N}_{timestamp}.png
         {scene_name}/road/lidar/merged_pcd/{timestamp}.pcd
       --scene_dir 指向具体场景文件夹, calib.json 会自动从父目录搜索

  B) self_Dataset/ 布局 (单场景):
       calib.json, {timestamp}.pcd, img/pinhole{0-3}/{timestamp}.png

用法:
    # 天津数据集 (car_road 布局, calib.json 自动从父目录搜索)
    python scripts/roadside/prepare_roadside_data.py \
        --scene_dir /mnt/car_road_data_TianJin/001_car0325_road0327_t1 \
        --output_dir ./data/roadside/scene_001 \
        --timestamp 1742877031036

    # 或手动指定 calib.json
    python scripts/roadside/prepare_roadside_data.py \
        --scene_dir /mnt/car_road_data_TianJin/001_car0325_road0327_t1 \
        --output_dir ./data/roadside/scene_001 \
        --calib_json /mnt/car_road_data_TianJin/support_info/calib.json \
        --timestamp 1742877031036
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: open3d not installed, will use manual PCD parser")


# ============================================================
# Default camera key mapping: pinhole folder name -> calib.json camera key
# This mapping is dataset-specific and hardcoded.
# ============================================================
DEFAULT_PINHOLE_TO_CALIB_KEY = {
    "pinhole0": "3",   # calib.json camera["3"]
    "pinhole1": "6",   # calib.json camera["6"]
    "pinhole2": "9",   # calib.json camera["9"]
    "pinhole3": "0",   # calib.json camera["0"]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert roadside data to S3Gaussian Waymo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # car_road layout (calib.json auto-detected from parent dir)
  python scripts/roadside/prepare_roadside_data.py \\
      --scene_dir /mnt/car_road_data_TianJin/001_car0325_road0327_t1 \\
      --output_dir ./data/roadside/scene_001 \\
      --timestamp 1742877031036

  # self_Dataset layout
  python scripts/roadside/prepare_roadside_data.py \\
      --scene_dir /path/to/self_Dataset \\
      --output_dir ./data/roadside/scene_001
        """)
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene folder")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory in S3GS Waymo format")
    parser.add_argument("--calib_json", type=str, default=None,
                        help="Path to calib.json. Auto-detected if not specified.")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Timestamp to use. Auto-detected if not specified.")
    parser.add_argument("--cameras", nargs="+",
                        default=["pinhole0", "pinhole1", "pinhole2", "pinhole3"],
                        help="Pinhole camera folder names (default: pinhole0-3)")
    parser.add_argument("--cam_key_mapping", type=str, default=None,
                        help='JSON string for pinhole->calib key mapping, e.g. '
                             '\'{"pinhole0":"3","pinhole1":"6","pinhole2":"9","pinhole3":"0"}\'')
    parser.add_argument("--undistort_alpha", type=float, default=0.0,
                        help="Alpha for cv2.getOptimalNewCameraMatrix (0=no black, 1=keep all pixels)")
    parser.add_argument("--filter_visible", action="store_true", default=True,
                        help="Only keep LiDAR points visible from at least one camera (default: True)")
    parser.add_argument("--no_filter_visible", dest="filter_visible", action="store_false",
                        help="Keep all LiDAR points")
    return parser.parse_args()


# ============================================================
# calib.json parsing
# ============================================================

def load_calib_json(calib_path):
    """Load and parse calib.json."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    return calib


def get_camera_params(calib, cam_key):
    """
    Extract camera parameters from calib.json.

    Returns:
        K: 3x3 intrinsic matrix (original, with distortion)
        dist_coeffs: distortion coefficients [k1, k2, p1, p2, k3]
        R_w2c: 3x3 rotation matrix (world-to-camera, from Rodrigues vector)
        t_w2c: 3x1 translation vector (world-to-camera)
        is_fish: bool
    """
    cam_data = calib["camera"][cam_key]
    is_fish = cam_data.get("isFish", 0) == 1

    # Intrinsic: 9 floats, row-major 3x3
    K = np.array(cam_data["intri"], dtype=np.float64).reshape(3, 3)

    # Distortion coefficients
    dist_coeffs = np.array(cam_data["distor"], dtype=np.float64)

    # Extrinsic: virtualLidarToCam (world-to-camera)
    vl2c = cam_data["virtualLidarToCam"]
    rvec = np.array(vl2c["rotate"], dtype=np.float64)  # Rodrigues 3D vector
    t_w2c = np.array(vl2c["trans"], dtype=np.float64)

    # Convert Rodrigues vector to rotation matrix
    R_w2c, _ = cv2.Rodrigues(rvec)

    return K, dist_coeffs, R_w2c, t_w2c, is_fish


def undistort_and_get_new_K(K, dist_coeffs, img_size, alpha=0.0):
    """
    Compute optimal new camera matrix for undistortion.

    Args:
        K: 3x3 original intrinsic matrix
        dist_coeffs: distortion coefficients
        img_size: (width, height)
        alpha: 0 = no black borders, 1 = keep all pixels

    Returns:
        K_new: 3x3 undistorted intrinsic matrix
        roi: valid region of interest
        map1, map2: undistortion remap tables
    """
    w, h = img_size
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K_new, (w, h), cv2.CV_32FC1)
    return K_new, roi, map1, map2


# ============================================================
# PCD loading
# ============================================================

def load_pcd_to_numpy(pcd_path):
    """Load a PCD file and return Nx3 numpy array of points (and intensities if available)."""
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        return points
    else:
        return _parse_pcd_manual(pcd_path)


def _parse_pcd_manual(pcd_path):
    """Parse ASCII or binary PCD files manually."""
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
            # Binary PCD
            dtype_map = {'F': 'f', 'U': 'u', 'I': 'i'}
            dt_list = []
            for field, typ, size in zip(fields, types, sizes):
                dt_char = dtype_map.get(typ, 'f')
                dt_list.append((field, f'{dt_char}{size}'))
            dt = np.dtype(dt_list)
            arr = np.frombuffer(f.read(num_points * dt.itemsize), dtype=dt, count=num_points)
            x = arr['x'].astype(np.float32)
            y = arr['y'].astype(np.float32)
            z = arr['z'].astype(np.float32)
            return np.stack([x, y, z], axis=-1)


# ============================================================
# Point cloud visibility filtering and depth map generation
# ============================================================

def project_points_to_camera(points, K, R_w2c, t_w2c, img_w, img_h):
    """
    Project 3D points to camera image plane.

    Returns:
        pixel_coords: Nx2 (u, v)
        depths: N (depth in camera frame, Zc)
        valid_mask: N bool
    """
    # Transform to camera frame: P_cam = R @ P_world + t
    P_cam = (R_w2c @ points.T).T + t_w2c  # Nx3

    depths = P_cam[:, 2]

    # Project to image: u = fx * Xc/Zc + cx, v = fy * Yc/Zc + cy
    valid = depths > 0.5  # minimum depth threshold

    u = K[0, 0] * P_cam[:, 0] / (depths + 1e-8) + K[0, 2]
    v = K[1, 1] * P_cam[:, 1] / (depths + 1e-8) + K[1, 2]

    valid = valid & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    pixel_coords = np.stack([u, v], axis=-1)
    return pixel_coords, depths, valid


def filter_visible_points(points, cameras_params, img_w, img_h):
    """Filter points visible from at least one camera."""
    visible_mask = np.zeros(len(points), dtype=bool)
    for K, R_w2c, t_w2c in cameras_params:
        _, _, valid = project_points_to_camera(points, K, R_w2c, t_w2c, img_w, img_h)
        visible_mask |= valid
    return visible_mask


# ============================================================
# Waymo format conversion
# ============================================================

def convert_pcd_to_waymo_lidar_bin(points, output_path):
    """
    Convert Nx3 point cloud to S3GS Waymo lidar format (Nx10 float32 binary).

    Columns: origins[3] + points[3] + unused[3] + laser_id[1]
    """
    N = len(points)
    lidar_data = np.zeros((N, 10), dtype=np.float32)
    lidar_data[:, 0:3] = 0.0       # origins (roadside LiDAR at origin)
    lidar_data[:, 3:6] = points    # 3D points in virtualLidar frame
    lidar_data[:, 6:9] = 0.0       # unused
    lidar_data[:, 9] = 0           # laser_id
    lidar_data.tofile(output_path)
    return N


# ============================================================
# Input layout detection
# ============================================================

def detect_layout(scene_dir):
    """
    Detect input data layout.

    Returns: "self_dataset", "car_road", or "support_info"
    """
    # self_Dataset layout: scene_dir/img/pinhole0/...
    if os.path.exists(os.path.join(scene_dir, "img")):
        return "self_dataset"
    # car_road layout: scene_dir/road/cameras/pinhole0/...
    if os.path.exists(os.path.join(scene_dir, "road", "cameras")):
        return "car_road"
    # support_info layout: scene_dir/support_info/calib.json + pinhole{N}/ at top level
    if os.path.exists(os.path.join(scene_dir, "support_info", "calib.json")):
        return "support_info"
    # Fallback: check for calib.json at top level
    if os.path.exists(os.path.join(scene_dir, "calib.json")):
        return "self_dataset"
    return "unknown"


def find_calib_json(scene_dir, layout):
    """Auto-detect calib.json location.

    Searches in scene_dir and its parent directory (for shared calib files).
    """
    parent_dir = os.path.dirname(os.path.abspath(scene_dir))
    candidates = [
        os.path.join(scene_dir, "calib.json"),
        os.path.join(scene_dir, "support_info", "calib.json"),
        os.path.join(scene_dir, "road", "calib.json"),
        os.path.join(scene_dir, "road", "calib", "calib.json"),
        os.path.join(scene_dir, "calib", "calib.json"),
        # Also search parent directory (calib shared across scenes)
        os.path.join(parent_dir, "support_info", "calib.json"),
        os.path.join(parent_dir, "calib.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_image_dir(scene_dir, layout, cam_name):
    """Get image directory for a given camera name."""
    if layout == "self_dataset":
        return os.path.join(scene_dir, "img", cam_name)
    else:  # car_road
        return os.path.join(scene_dir, "road", "cameras", cam_name)


def find_pcd_path(scene_dir, layout, timestamp):
    """Get PCD file path."""
    if layout == "self_dataset":
        return os.path.join(scene_dir, f"{timestamp}.pcd")
    else:  # car_road
        return os.path.join(scene_dir, "road", "lidar", "merged_pcd", f"{timestamp}.pcd")


def find_image_file(img_dir, timestamp):
    """Find image file matching timestamp in directory."""
    # Try direct match first
    for ext in [".png", ".jpg"]:
        direct = os.path.join(img_dir, f"{timestamp}{ext}")
        if os.path.exists(direct):
            return direct

    # Search for files containing timestamp
    if os.path.exists(img_dir):
        for f in sorted(os.listdir(img_dir)):
            if timestamp in f and (f.endswith(".png") or f.endswith(".jpg")):
                return os.path.join(img_dir, f)
    return None


def find_timestamp(scene_dir, layout, cameras):
    """Auto-detect a valid timestamp from available data."""
    cam_dir = find_image_dir(scene_dir, layout, cameras[0])
    if not os.path.exists(cam_dir):
        return None
    for f in sorted(os.listdir(cam_dir)):
        if f.endswith(".png") or f.endswith(".jpg"):
            # Extract timestamp
            name = f.rsplit(".", 1)[0]
            # Handle both "1743583131842.png" and "cam5_1742877031036.png"
            parts = name.split("_")
            ts = parts[-1]  # last part is always the timestamp
            return ts
    return None


# ============================================================
# Main conversion
# ============================================================

def prepare_roadside_data(args):
    scene_dir = args.scene_dir
    output_dir = args.output_dir
    cameras = args.cameras
    num_cameras = len(cameras)

    # Parse camera key mapping
    if args.cam_key_mapping:
        cam_key_map = json.loads(args.cam_key_mapping)
    else:
        cam_key_map = DEFAULT_PINHOLE_TO_CALIB_KEY

    # Detect layout
    layout = detect_layout(scene_dir)
    print(f"Detected layout: {layout}")
    print(f"Scene: {scene_dir}")
    print(f"Output: {output_dir}")
    print(f"Cameras: {cameras}")
    print(f"Camera key mapping: {cam_key_map}")

    # Find calib.json
    calib_path = args.calib_json or find_calib_json(scene_dir, layout)
    if calib_path is None or not os.path.exists(calib_path):
        print(f"ERROR: calib.json not found. Searched in scene_dir and subdirectories.")
        print(f"  Please specify --calib_json /path/to/calib.json")
        sys.exit(1)
    print(f"Calibration: {calib_path}")
    calib = load_calib_json(calib_path)

    # Find timestamp
    timestamp = args.timestamp or find_timestamp(scene_dir, layout, cameras)
    if timestamp is None:
        print("ERROR: Could not auto-detect timestamp. Please specify --timestamp")
        sys.exit(1)
    print(f"Timestamp: {timestamp}")

    # Get image size from calib
    img_size_hw = calib.get("imgSize", {}).get("notFish", [1280, 720])
    # imgSize in calib.json is [width, height] for notFish
    if len(img_size_hw) == 2:
        img_w, img_h = img_size_hw[0], img_size_hw[1]
    else:
        img_w, img_h = 1280, 720
    print(f"Image size: {img_w}x{img_h}")

    # Create output directories
    for subdir in ["images", "intrinsics", "extrinsics", "ego_pose", "lidar"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # ========================================
    # Process each camera
    # ========================================
    all_K_new = []
    all_R_w2c = []
    all_t_w2c = []
    all_c2w = []

    for cam_idx, cam_name in enumerate(cameras):
        calib_key = cam_key_map.get(cam_name)
        if calib_key is None:
            print(f"ERROR: No calib key mapping for {cam_name}")
            print(f"  Available mappings: {cam_key_map}")
            sys.exit(1)

        if calib_key not in calib["camera"]:
            print(f"ERROR: Camera key '{calib_key}' not found in calib.json")
            print(f"  Available keys: {list(calib['camera'].keys())}")
            sys.exit(1)

        # Extract camera parameters
        K, dist_coeffs, R_w2c, t_w2c, is_fish = get_camera_params(calib, calib_key)
        if is_fish:
            print(f"WARNING: {cam_name} (calib key={calib_key}) is a fisheye camera, skipping")
            continue

        print(f"\n--- {cam_name} (calib key=cam{calib_key}) ---")
        print(f"  K (original): fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        print(f"  Distortion: {dist_coeffs}")

        # Compute undistorted camera matrix
        K_new, roi, map1, map2 = undistort_and_get_new_K(
            K, dist_coeffs, (img_w, img_h), alpha=args.undistort_alpha)

        print(f"  K (undistorted): fx={K_new[0,0]:.2f} fy={K_new[1,1]:.2f} "
              f"cx={K_new[0,2]:.2f} cy={K_new[1,2]:.2f}")

        # Compute camera center in world frame
        cam_center = -R_w2c.T @ t_w2c
        print(f"  Camera center (virtualLidar): [{cam_center[0]:.1f}, {cam_center[1]:.1f}, {cam_center[2]:.1f}]")

        all_K_new.append(K_new)
        all_R_w2c.append(R_w2c)
        all_t_w2c.append(t_w2c)

        # Compute camera-to-world (c2w) 4x4 matrix = inv(w2c)
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c
        T_c2w = np.linalg.inv(T_w2c)
        all_c2w.append(T_c2w)

        # ---- Save intrinsic: [fx, fy, cx, cy] ----
        intrinsic_arr = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
        intr_path = os.path.join(output_dir, "intrinsics", f"{cam_idx}.txt")
        np.savetxt(intr_path, intrinsic_arr, fmt="%.10f")
        print(f"  -> intrinsics/{cam_idx}.txt")

        # ---- Save extrinsic: 4x4 camera-to-world matrix ----
        # S3GS readRoadsideInfo expects camera-to-ego (= camera-to-world for static roadside)
        extr_path = os.path.join(output_dir, "extrinsics", f"{cam_idx}.txt")
        np.savetxt(extr_path, T_c2w, fmt="%.10f")
        print(f"  -> extrinsics/{cam_idx}.txt")

        # ---- Undistort and save image ----
        img_dir = find_image_dir(scene_dir, layout, cam_name)
        img_path = find_image_file(img_dir, timestamp)

        if img_path is None:
            print(f"  WARNING: Image not found for {cam_name} at timestamp {timestamp}")
            print(f"    Searched in: {img_dir}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: Cannot read image: {img_path}")
            continue

        # Undistort image
        img_undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        dst_path = os.path.join(output_dir, "images", f"000_{cam_idx}.jpg")
        cv2.imwrite(dst_path, img_undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  -> images/000_{cam_idx}.jpg (undistorted)")

    # ========================================
    # Save ego_pose (identity for static roadside)
    # ========================================
    ego_pose = np.eye(4)
    ego_pose_path = os.path.join(output_dir, "ego_pose", "000.txt")
    np.savetxt(ego_pose_path, ego_pose, fmt="%.10f")
    print(f"\n  -> ego_pose/000.txt (identity)")

    # ========================================
    # Process LiDAR
    # ========================================
    pcd_path = find_pcd_path(scene_dir, layout, timestamp)
    if os.path.exists(pcd_path):
        print(f"\nLoading PCD: {pcd_path}")
        points = load_pcd_to_numpy(pcd_path)
        print(f"  Total points: {len(points)}")
        print(f"  XYZ range: [{points.min(axis=0)}] ~ [{points.max(axis=0)}]")

        # Visibility filtering: keep only points seen by at least one camera
        if args.filter_visible and all_K_new:
            cam_params = list(zip(all_K_new, all_R_w2c, all_t_w2c))
            visible_mask = filter_visible_points(points, cam_params, img_w, img_h)
            n_visible = visible_mask.sum()
            print(f"  Visible points: {n_visible}/{len(points)} ({n_visible/len(points)*100:.1f}%)")
            points = points[visible_mask]
            print(f"  XYZ range (filtered): [{points.min(axis=0)}] ~ [{points.max(axis=0)}]")

        # Save as Waymo bin format
        bin_path = os.path.join(output_dir, "lidar", "000.bin")
        n_saved = convert_pcd_to_waymo_lidar_bin(points, bin_path)
        print(f"  -> lidar/000.bin ({n_saved} points)")
    else:
        print(f"\nWARNING: PCD not found: {pcd_path}")
        print("  Training will fall back to random point initialization.")

    # ========================================
    # Save frame_info.json
    # ========================================
    frame_info = {
        "scene_type": "roadside",
        "num_cameras": num_cameras,
        "cameras": cameras,
        "cam_key_mapping": cam_key_map,
        "timestamp": timestamp,
        "num_frames": 1,
        "original_sizes": [[img_h, img_w]] * num_cameras,
        "coordinate_system": "virtualLidar",
        "notes": "Single-frame roadside reconstruction. "
                 "Ego-pose is identity (static infrastructure). "
                 "Extrinsics are camera-to-world (inverse of virtualLidarToCam). "
                 "Intrinsics are undistorted K_new (alpha=0). "
                 "Images are undistorted."
    }
    frame_info_path = os.path.join(output_dir, "frame_info.json")
    with open(frame_info_path, 'w') as f:
        json.dump(frame_info, f, indent=2, ensure_ascii=False)
    print(f"\n  -> frame_info.json")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {output_dir}")
    print(f"  images/       {num_cameras} undistorted images")
    print(f"  intrinsics/   {num_cameras} files (undistorted K_new: fx fy cx cy)")
    print(f"  extrinsics/   {num_cameras} files (camera-to-world 4x4)")
    print(f"  ego_pose/     1 file (identity 4x4)")
    if os.path.exists(os.path.join(output_dir, "lidar", "000.bin")):
        print(f"  lidar/        1 file (Nx10 float32 bin)")
    print(f"  frame_info.json")

    if all_c2w:
        centers = np.array([-R.T @ t for R, t in zip(all_R_w2c, all_t_w2c)])
        cam_spread = np.max(centers, axis=0) - np.min(centers, axis=0)
        print(f"\nCamera spread: X={cam_spread[0]:.1f}m  Y={cam_spread[1]:.1f}m  Z={cam_spread[2]:.1f}m")
        print(f"Camera center mean: {centers.mean(axis=0)}")

    print(f"\n{'=' * 60}")
    print("Next: run training with")
    print(f"  bash scripts/roadside/run_roadside.sh 0 {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    prepare_roadside_data(args)
