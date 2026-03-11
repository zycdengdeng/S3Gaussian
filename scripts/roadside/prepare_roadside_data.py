"""
路侧数据集 -> S3Gaussian Waymo 格式转换脚本

将车路协同数据集中路侧（road）的 pinhole 相机数据和 LiDAR 数据
转换为 S3Gaussian 所需的 Waymo 目录格式。

用法:
    python scripts/roadside/prepare_roadside_data.py \
        --scene_dir /mnt/car_road_data_fix/001_car0325_road0327_t1 \
        --output_dir ./data/roadside/scene_001 \
        --calib_dir /path/to/calibration \
        --timestamp 1742877031036 \
        --cameras pinhole0 pinhole1 pinhole2 pinhole3

输入目录结构:
    scene_dir/road/
        cameras/pinhole{0-3}/cam{N}_TIMESTAMP.png
        lidar/merged_pcd/TIMESTAMP.pcd

输出目录结构 (Waymo 格式):
    output_dir/
        images/         000_{cam_id}.jpg
        intrinsics/     {cam_id}.txt
        extrinsics/     {cam_id}.txt
        ego_pose/       000.txt
        lidar/          000.bin
        frame_info.json
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("WARNING: opencv-python not installed, image operations will fail")
    cv2 = None

try:
    import open3d as o3d
except ImportError:
    print("WARNING: open3d not installed, PCD loading will fail")
    o3d = None


def parse_args():
    parser = argparse.ArgumentParser(description="Convert roadside data to S3Gaussian Waymo format")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene folder, e.g., /mnt/car_road_data_fix/001_car0325_road0327_t1")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory in Waymo format")
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="Path to calibration files directory. "
                             "If not specified, will look for calib/ under scene_dir/road/")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Specific timestamp to use (roadside ms format). "
                             "If not specified, uses the first available timestamp.")
    parser.add_argument("--cameras", nargs="+", default=["pinhole0", "pinhole1", "pinhole2", "pinhole3"],
                        help="Camera names to use (default: pinhole0-3)")
    parser.add_argument("--image_size", nargs=2, type=int, default=None,
                        help="Resize images to [height, width]. If not specified, keep original size.")
    parser.add_argument("--lidar_source", type=str, default="merged_pcd",
                        choices=["merged_pcd", "lidar0", "lidar1", "lidar2", "lidar3"],
                        help="Which lidar source to use (default: merged_pcd)")
    return parser.parse_args()


def find_available_timestamp(scene_dir, cameras, lidar_source="merged_pcd"):
    """Find the first timestamp that has both images and LiDAR data available."""
    # Collect image timestamps per camera
    cam_timestamps = {}
    for cam_name in cameras:
        cam_dir = os.path.join(scene_dir, "road", "cameras", cam_name)
        if not os.path.exists(cam_dir):
            print(f"WARNING: Camera directory not found: {cam_dir}")
            continue
        timestamps = set()
        for f in os.listdir(cam_dir):
            if f.endswith(".png") or f.endswith(".jpg"):
                # Format: cam{N}_TIMESTAMP.png
                ts = f.split("_", 1)[1].rsplit(".", 1)[0]
                timestamps.add(ts)
        cam_timestamps[cam_name] = timestamps

    # Collect LiDAR timestamps
    lidar_dir = os.path.join(scene_dir, "road", "lidar", lidar_source)
    lidar_timestamps = set()
    if os.path.exists(lidar_dir):
        for f in os.listdir(lidar_dir):
            if f.endswith(".pcd"):
                ts = f.rsplit(".", 1)[0]  # Remove .pcd
                lidar_timestamps.add(ts)

    # Find common timestamps
    if cam_timestamps:
        common_ts = set.intersection(*cam_timestamps.values())
    else:
        common_ts = set()

    common_ts = common_ts & lidar_timestamps if lidar_timestamps else common_ts

    if not common_ts:
        print("WARNING: No common timestamp found across all sensors.")
        print(f"  Camera timestamps: {[len(v) for v in cam_timestamps.values()]}")
        print(f"  LiDAR timestamps: {len(lidar_timestamps)}")
        # Fallback: use the first camera timestamp
        if cam_timestamps:
            first_cam = list(cam_timestamps.values())[0]
            if first_cam:
                return sorted(first_cam)[0]
        return None

    return sorted(common_ts)[0]


def load_pcd_to_numpy(pcd_path):
    """Load a PCD file and return Nx3 numpy array of points."""
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        return points
    else:
        # Fallback: manual PCD parsing
        return _parse_pcd_manual(pcd_path)


def _parse_pcd_manual(pcd_path):
    """Manually parse ASCII/binary PCD files."""
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
                vals = line.split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
            return np.array(points, dtype=np.float32)
        else:
            # Binary PCD
            # Calculate point size
            point_size = sum(int(s) for s in sizes)
            data = f.read(num_points * point_size)
            # Find x, y, z field indices
            dtype_map = {'F': 'f', 'U': 'u', 'I': 'i'}
            dt_list = []
            for field, typ, size in zip(fields, types, sizes):
                dt_char = dtype_map.get(typ, 'f')
                dt_list.append((field, f'{dt_char}{size}'))
            dt = np.dtype(dt_list)
            arr = np.frombuffer(data, dtype=dt, count=num_points)
            x = arr['x'].astype(np.float32)
            y = arr['y'].astype(np.float32)
            z = arr['z'].astype(np.float32)
            return np.stack([x, y, z], axis=-1)


def convert_pcd_to_waymo_lidar_bin(points, output_path):
    """
    Convert Nx3 point cloud to S3GS Waymo lidar format (Nx10 float32 binary).

    Waymo lidar format columns:
      [0:3] = ray origins (we set to zeros since roadside LiDAR is static)
      [3:6] = 3D points (x, y, z)
      [6:9] = unused (zeros)
      [9]   = laser_id (all zeros for merged)
    """
    N = len(points)
    lidar_data = np.zeros((N, 10), dtype=np.float32)
    # Ray origins = LiDAR sensor position (for roadside, use zeros or actual position)
    # Setting origins to zero means the LiDAR is at the origin of the coordinate system
    lidar_data[:, 0:3] = 0.0  # origins
    lidar_data[:, 3:6] = points  # 3D points
    lidar_data[:, 6:9] = 0.0  # unused
    lidar_data[:, 9] = 0  # laser_id
    lidar_data.tofile(output_path)
    print(f"  Saved lidar bin: {output_path} ({N} points)")


def load_calibration(calib_dir, cam_name):
    """
    Load calibration for a specific camera.

    Expected calibration file format (JSON):
    {
        "intrinsic": {
            "fx": ..., "fy": ..., "cx": ..., "cy": ...,
            "k1": ..., "k2": ..., "p1": ..., "p2": ..., "k3": ...
        },
        "extrinsic": {
            // 4x4 matrix as list of lists, camera-to-world or camera-to-reference
            "transform": [[...], [...], [...], [...]],
        },
        "image_width": ...,
        "image_height": ...
    }

    Also supports simple text format:
        intrinsic file: fx fy cx cy [k1 k2 p1 p2 k3]
        extrinsic file: 4x4 matrix (4 rows, 4 columns)

    Adjust this function based on your actual calibration file format.
    """
    intrinsic = None
    extrinsic = None
    image_size = None  # [height, width]

    # Try JSON format first
    json_path = os.path.join(calib_dir, f"{cam_name}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            calib = json.load(f)
        intr = calib["intrinsic"]
        intrinsic = np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]])
        if "extrinsic" in calib:
            extrinsic = np.array(calib["extrinsic"]["transform"], dtype=np.float64)
        if "image_width" in calib and "image_height" in calib:
            image_size = [calib["image_height"], calib["image_width"]]
        return intrinsic, extrinsic, image_size

    # Try separate txt files
    intr_path = os.path.join(calib_dir, f"{cam_name}_intrinsic.txt")
    extr_path = os.path.join(calib_dir, f"{cam_name}_extrinsic.txt")

    if os.path.exists(intr_path):
        intrinsic = np.loadtxt(intr_path).flatten()[:4]  # fx, fy, cx, cy

    if os.path.exists(extr_path):
        extrinsic = np.loadtxt(extr_path).reshape(4, 4)

    # Try yaml format
    yaml_path = os.path.join(calib_dir, f"{cam_name}.yaml")
    if os.path.exists(yaml_path) and intrinsic is None:
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                calib = yaml.safe_load(f)
            if "camera_matrix" in calib:
                K = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
                intrinsic = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            if "extrinsic_matrix" in calib:
                extrinsic = np.array(calib["extrinsic_matrix"]["data"]).reshape(4, 4)
            if "image_width" in calib:
                image_size = [calib["image_height"], calib["image_width"]]
        except ImportError:
            print("WARNING: pyyaml not installed, cannot load yaml calibration")

    return intrinsic, extrinsic, image_size


def prepare_roadside_data(args):
    """Main conversion function."""
    scene_dir = args.scene_dir
    output_dir = args.output_dir
    cameras = args.cameras
    num_cameras = len(cameras)

    print(f"Scene: {scene_dir}")
    print(f"Output: {output_dir}")
    print(f"Cameras: {cameras}")

    # ---- Determine calibration directory ----
    calib_dir = args.calib_dir
    if calib_dir is None:
        # Look for common calibration locations
        candidates = [
            os.path.join(scene_dir, "road", "calib"),
            os.path.join(scene_dir, "road", "calibration"),
            os.path.join(scene_dir, "calib"),
            os.path.join(scene_dir, "calibration"),
        ]
        for c in candidates:
            if os.path.exists(c):
                calib_dir = c
                break
        if calib_dir is None:
            print("ERROR: Calibration directory not found. Please specify --calib_dir")
            print(f"  Searched: {candidates}")
            sys.exit(1)
    print(f"Calibration dir: {calib_dir}")

    # ---- Determine timestamp ----
    timestamp = args.timestamp
    if timestamp is None:
        timestamp = find_available_timestamp(scene_dir, cameras, args.lidar_source)
        if timestamp is None:
            print("ERROR: Could not find a valid timestamp. Please specify --timestamp")
            sys.exit(1)
    print(f"Using timestamp: {timestamp}")

    # ---- Create output directories ----
    for subdir in ["images", "intrinsics", "extrinsics", "ego_pose", "lidar"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # ---- Load and save calibration ----
    original_sizes = []
    for cam_idx, cam_name in enumerate(cameras):
        intrinsic, extrinsic, img_size = load_calibration(calib_dir, cam_name)

        if intrinsic is None:
            print(f"ERROR: Could not load intrinsic for {cam_name}")
            print(f"  Please provide calibration files in one of these formats:")
            print(f"    - {calib_dir}/{cam_name}.json")
            print(f"    - {calib_dir}/{cam_name}_intrinsic.txt (fx fy cx cy)")
            print(f"    - {calib_dir}/{cam_name}.yaml")
            sys.exit(1)

        if extrinsic is None:
            print(f"ERROR: Could not load extrinsic for {cam_name}")
            sys.exit(1)

        # Save intrinsic: [fx, fy, cx, cy] (same as Waymo format)
        intr_path = os.path.join(output_dir, "intrinsics", f"{cam_idx}.txt")
        np.savetxt(intr_path, intrinsic.flatten(), fmt="%.10f")
        print(f"  Saved intrinsic [{cam_name} -> cam_id={cam_idx}]: {intr_path}")

        # Save extrinsic: 4x4 camera-to-ego matrix
        # For roadside setup, "ego" is the roadside reference coordinate system.
        # The extrinsic should be camera-to-reference(world) transform.
        extr_path = os.path.join(output_dir, "extrinsics", f"{cam_idx}.txt")
        np.savetxt(extr_path, extrinsic, fmt="%.10f")
        print(f"  Saved extrinsic [{cam_name} -> cam_id={cam_idx}]: {extr_path}")

        if img_size:
            original_sizes.append(img_size)

    # ---- Save ego pose (identity for static roadside) ----
    # For single-frame roadside reconstruction, ego_pose is identity
    # because the "ego vehicle" (road infrastructure) doesn't move.
    ego_pose = np.eye(4)
    ego_pose_path = os.path.join(output_dir, "ego_pose", "000.txt")
    np.savetxt(ego_pose_path, ego_pose, fmt="%.10f")
    print(f"  Saved ego_pose (identity): {ego_pose_path}")

    # ---- Copy and rename images ----
    # Camera name -> cam_id mapping for the filename
    cam_name_to_id = {}  # maps e.g. "pinhole0" to Waymo cam_id number
    for cam_idx, cam_name in enumerate(cameras):
        cam_name_to_id[cam_name] = cam_idx

        # Find the image file
        cam_dir = os.path.join(scene_dir, "road", "cameras", cam_name)
        if not os.path.exists(cam_dir):
            print(f"WARNING: Camera directory not found: {cam_dir}")
            continue

        # Look for image with this timestamp
        # Roadside image format: cam{N}_TIMESTAMP.png
        # Extract camera number from cam_name (e.g., pinhole0 -> probably cam4 or cam5)
        img_file = None
        for f in os.listdir(cam_dir):
            if timestamp in f and (f.endswith(".png") or f.endswith(".jpg")):
                img_file = f
                break

        if img_file is None:
            print(f"WARNING: No image found for {cam_name} at timestamp {timestamp}")
            print(f"  Searched in: {cam_dir}")
            continue

        # Read, optionally resize, and save
        src_path = os.path.join(cam_dir, img_file)
        dst_path = os.path.join(output_dir, "images", f"000_{cam_idx}.jpg")

        if cv2 is not None:
            img = cv2.imread(src_path)
            if img is None:
                print(f"ERROR: Cannot read image: {src_path}")
                continue

            if not original_sizes:
                original_sizes.append([img.shape[0], img.shape[1]])

            if args.image_size is not None:
                img = cv2.resize(img, (args.image_size[1], args.image_size[0]))

            cv2.imwrite(dst_path, img)
        else:
            # Just copy
            import shutil
            shutil.copy2(src_path, dst_path)

        print(f"  Saved image [{cam_name}]: {dst_path}")

    # ---- Convert LiDAR data ----
    lidar_dir = os.path.join(scene_dir, "road", "lidar", args.lidar_source)
    pcd_file = os.path.join(lidar_dir, f"{timestamp}.pcd")

    if os.path.exists(pcd_file):
        print(f"  Loading PCD: {pcd_file}")
        points = load_pcd_to_numpy(pcd_file)
        print(f"  Loaded {len(points)} points")

        # Save as Waymo-format binary
        bin_path = os.path.join(output_dir, "lidar", "000.bin")
        convert_pcd_to_waymo_lidar_bin(points, bin_path)
    else:
        print(f"WARNING: LiDAR PCD not found: {pcd_file}")
        print("  Will fall back to random initialization during training.")

    # ---- Create frame_info.json (trigger for Waymo loader) ----
    frame_info = {
        "scene_type": "roadside",
        "num_cameras": num_cameras,
        "cameras": cameras,
        "cam_id_mapping": cam_name_to_id,
        "timestamp": timestamp,
        "num_frames": 1,
        "original_sizes": original_sizes if original_sizes else [[1080, 1920]] * num_cameras,
        "notes": "Single-frame roadside reconstruction. Ego-pose is identity (static infrastructure)."
    }
    frame_info_path = os.path.join(output_dir, "frame_info.json")
    with open(frame_info_path, 'w') as f:
        json.dump(frame_info, f, indent=2)
    print(f"  Saved frame_info.json: {frame_info_path}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("Data conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  images/       : {num_cameras} images (000_0.jpg ~ 000_{num_cameras-1}.jpg)")
    print(f"  intrinsics/   : {num_cameras} files")
    print(f"  extrinsics/   : {num_cameras} files")
    print(f"  ego_pose/     : 1 file (identity)")
    print(f"  lidar/        : 1 file")
    print(f"  frame_info.json")
    print("=" * 50)
    print("\nNext steps:")
    print(f"  1. Verify calibration files are correct")
    print(f"  2. Run training:")
    print(f"     bash scripts/roadside/run_roadside.sh 0 {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    prepare_roadside_data(args)
