"""
Batch render trained S3Gaussian roadside scenes from VEHICLE camera viewpoints.

Transform chain (vehicle cameras):
  Vehicle Camera --(cam2lidar)--> Vehicle LiDAR --(inv world2lidar)--> World (virtualLiDAR)

Data sources:
  - vehicle_calib/camera/camera_XX_{intrinsics,extrinsics}.yaml: vehicle camera calibration
  - transform_json/{scene_id}/*.json: per-scene world2lidar transforms
  - scene_timestamps.json: maps scene_name -> target timestamp (ms)
  - models/{scene_name}/chkpnt_fine_30000.pth: trained S3Gaussian model

Usage:
  python scripts/roadside/render_roadside_batch.py \
    --model_root /path/to/work_dir/models \
    --vehicle_calib /path/to/vehicle/calibration \
    --transform_root /path/to/transform_json \
    --output_root /path/to/work_dir/renders

  Render a specific scene:
    --scene_name scene003_far
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import cv2
import json
import yaml
import glob
import re
import math
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from arguments import ModelHiddenParams, PipelineParams
from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
import torchvision


# Vehicle camera definitions
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "front narrow 30°",  "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "front wide 120°",   "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "front-left 120°",   "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "front-right 120°",  "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "rear-left 60°",     "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "rear-right 60°",    "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "rear narrow 60°",   "resolution": (1920, 1080)},
}


def quaternion_to_rotation_matrix(q):
    """Quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])


def load_vehicle_camera(calib_folder, cam_id):
    """Load vehicle camera intrinsics and extrinsics from YAML files.

    Returns:
        K (3x3), D (distortion), R_cam2lidar (3x3), t_cam2lidar (3,), resolution (w, h)
    """
    calib_folder = Path(calib_folder)
    cam_subdir = calib_folder / "camera"
    base = cam_subdir if cam_subdir.is_dir() else calib_folder

    # Intrinsics
    intr_path = base / f"camera_{cam_id:02d}_intrinsics.yaml"
    with open(intr_path, 'r') as f:
        intrinsics = yaml.safe_load(f)
    K = np.array(intrinsics['K']).reshape(3, 3)
    D = np.array(intrinsics['D'])

    # Extrinsics (camera -> lidar)
    extr_path = base / f"camera_{cam_id:02d}_extrinsics.yaml"
    with open(extr_path, 'r') as f:
        extrinsics = yaml.safe_load(f)
    transform = extrinsics['transform']
    q = [transform['rotation']['x'], transform['rotation']['y'],
         transform['rotation']['z'], transform['rotation']['w']]
    t = np.array([transform['translation']['x'],
                  transform['translation']['y'],
                  transform['translation']['z']])

    R_cam2lidar = quaternion_to_rotation_matrix(q)
    resolution = VEHICLE_CAMERAS[cam_id]["resolution"]
    return K, D, R_cam2lidar, t, resolution


def compute_undistorted_intrinsics(K, D, cam_id, resolution):
    """Compute new camera matrix after undistortion."""
    w, h = resolution
    if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D[:4], (w, h), np.eye(3), balance=0.0
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    return new_K


def load_scene_transform(transform_root, scene_id, target_ts_ms):
    """Load the world2lidar transform closest to target_ts_ms.

    Returns:
        (timestamp, R_w2l (3x3), t_w2l (3,))
    """
    scene_dir = os.path.join(transform_root, scene_id)
    json_files = glob.glob(os.path.join(scene_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {scene_dir}")

    with open(json_files[0], 'r') as f:
        transforms = json.load(f)

    best = None
    best_diff = float('inf')
    for entry in transforms:
        ts = entry["timestamp"]
        ts_ms = ts * 1000 if ts < 1e12 else ts
        diff = abs(ts_ms - target_ts_ms)
        if diff < best_diff:
            best_diff = diff
            best = entry

    if best is None:
        raise ValueError(f"No transforms found in {scene_dir}")

    rotvec = np.array(best["world2lidar"]["rotation"])
    R_w2l = Rotation.from_rotvec(rotvec).as_matrix()
    t_w2l = np.array(best["world2lidar"]["translation"])

    print(f"  world2lidar: closest ts diff = {best_diff:.0f}ms")
    return best["timestamp"], R_w2l, t_w2l


def extract_scene_id(scene_name):
    """Extract scene number from name: 'scene003_far' -> '003'."""
    m = re.match(r"scene(\d+)", scene_name)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract scene ID from '{scene_name}'")


def compute_vehicle_w2c(R_w2l, t_w2l, R_cam2lidar, t_cam2lidar):
    """Compute world-to-camera for vehicle camera.

    Chain: Camera --(cam2lidar)--> LiDAR --(inv world2lidar)--> World
    So: T_c2w = inv(T_w2l) @ T_c2l
        T_w2c = inv(T_c2w)

    Returns R_stored, T_stored in S3Gaussian convention (R transposed).
    """
    T_w2l = np.eye(4)
    T_w2l[:3, :3] = R_w2l
    T_w2l[:3, 3] = t_w2l

    T_c2l = np.eye(4)
    T_c2l[:3, :3] = R_cam2lidar
    T_c2l[:3, 3] = t_cam2lidar

    T_l2w = np.linalg.inv(T_w2l)
    T_c2w = T_l2w @ T_c2l
    T_w2c = np.linalg.inv(T_c2w)

    R_stored = T_w2c[:3, :3].T
    T_stored = T_w2c[:3, 3]
    return R_stored, T_stored


def create_minicam(R_stored, T_stored, fovx, fovy, width, height, time_val):
    """Create a MiniCam for rendering."""
    world_view_transform = torch.tensor(
        getWorld2View2(R_stored, T_stored)
    ).transpose(0, 1).cuda()

    znear, zfar = 0.01, 100.0
    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).cuda()

    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    cam = MiniCam(
        width=width, height=height,
        fovy=fovy, fovx=fovx,
        znear=znear, zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )
    cam.time = time_val
    return cam


def load_trained_model(model_path, iteration, hyper_args):
    """Load a trained S3Gaussian model from checkpoint."""
    sh_degree = 3
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = eval(f.read())
            if hasattr(cfg, 'sh_degree'):
                sh_degree = cfg.sh_degree

    gaussians = GaussianModel(sh_degree, hyper_args)

    chkpnt_path = os.path.join(model_path, f"chkpnt_fine_{iteration}.pth")
    print(f"Loading checkpoint: {chkpnt_path}")
    (model_params, _) = torch.load(chkpnt_path, map_location="cuda")

    (active_sh_degree, xyz, deform_state, deformation_table,
     features_dc, features_rest, scaling, rotation, opacity,
     max_radii2D, _xyz_gradient_accum, _denom, _opt_dict,
     spatial_lr_scale) = model_params

    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = torch.nn.Parameter(xyz)
    gaussians._features_dc = torch.nn.Parameter(features_dc)
    gaussians._features_rest = torch.nn.Parameter(features_rest)
    gaussians._scaling = torch.nn.Parameter(scaling)
    gaussians._rotation = torch.nn.Parameter(rotation)
    gaussians._opacity = torch.nn.Parameter(opacity)
    gaussians.max_radii2D = max_radii2D
    gaussians.spatial_lr_scale = spatial_lr_scale
    gaussians._deformation_table = deformation_table
    gaussians._deformation.load_state_dict(deform_state, strict=False)
    gaussians._deformation = gaussians._deformation.to("cuda")
    gaussians._deformation.eval()

    print(f"Model loaded: {gaussians._xyz.shape[0]} points, sh_degree={active_sh_degree}")
    return gaussians


def render_scene(model_path, vehicle_calib, camera_ids, R_w2l, t_w2l,
                 scene_name, pipe, output_dir, render_scale=4, time_val=0.0):
    """Render one scene from vehicle cameras at ONE timestamp."""
    gaussians = load_trained_model(model_path, 30000, pipe["hyper"])
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Rendering {scene_name}: {len(camera_ids)} vehicle cameras")

    scene_out = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    for cam_id in tqdm(camera_ids, desc=scene_name):
        cam_info = VEHICLE_CAMERAS.get(cam_id)
        if cam_info is None:
            print(f"  Unknown camera ID: {cam_id}, skipping")
            continue

        cam_name = cam_info["name"]

        # Load vehicle camera calibration
        K, D, R_cam2lidar, t_cam2lidar, resolution = load_vehicle_camera(
            vehicle_calib, cam_id
        )
        w, h = resolution

        # Apply render scale
        if render_scale != 1:
            w = w // render_scale
            h = h // render_scale

        # Compute undistorted intrinsics
        new_K = compute_undistorted_intrinsics(K, D, cam_id, resolution)

        # Scale intrinsics
        if render_scale != 1:
            scale_x = w / resolution[0]
            scale_y = h / resolution[1]
            new_K[0, :] *= scale_x
            new_K[1, :] *= scale_y

        fx, fy = new_K[0, 0], new_K[1, 1]
        fovx = focal2fov(fx, w)
        fovy = focal2fov(fy, h)

        # Compute vehicle camera pose in world coordinates
        R_stored, T_stored = compute_vehicle_w2c(
            R_w2l, t_w2l, R_cam2lidar, t_cam2lidar
        )

        cam = create_minicam(R_stored, T_stored, fovx, fovy, w, h, time_val)

        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe["pipe"], bg_color, stage="fine")

        rendering = render_pkg["render"]

        # Save: {scene}/{cam_name}.png (e.g. scene003_far/FN.png)
        torchvision.utils.save_image(
            rendering, os.path.join(scene_out, f"{cam_name}.png")
        )

    print(f"Saved to {scene_out}")


def main():
    parser = ArgumentParser(description="Batch render roadside scenes from vehicle camera viewpoints")
    pipeline_params = PipelineParams(parser)
    hyper_params = ModelHiddenParams(parser)

    parser.add_argument("--model_root", type=str, required=True,
                        help="Root of trained models (work_dir/models)")
    parser.add_argument("--vehicle_calib", type=str, required=True,
                        help="Path to vehicle calibration folder (contains camera/ subdir with YAML files)")
    parser.add_argument("--transform_root", type=str, required=True,
                        help="Root of transform_json directories")
    parser.add_argument("--timestamp_map", type=str,
                        default=os.path.join(os.path.dirname(__file__), "scene_timestamps.json"),
                        help="JSON mapping scene_name -> timestamp_ms")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for rendered images")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Render specific scene (default: all)")
    parser.add_argument("--camera_ids", type=int, nargs='+', default=[1, 5, 6, 7],
                        help="Vehicle camera IDs to render (default: 1 5 6 7, non-fisheye)")
    parser.add_argument("--render_scale", type=int, default=4,
                        help="Downscale factor for rendering resolution (default: 4)")
    parser.add_argument("--time", type=float, default=0.0,
                        help="Time value for deformation network (0.0 = first frame)")

    args = parser.parse_args()
    pipe_args = pipeline_params.extract(args)
    hyper = hyper_params.extract(args)

    # Load timestamp mapping
    with open(args.timestamp_map, 'r') as f:
        ts_map = json.load(f)
    print(f"Loaded timestamp mapping: {len(ts_map)} entries")

    # Verify vehicle calibration folder
    calib_path = Path(args.vehicle_calib)
    cam_subdir = calib_path / "camera"
    calib_base = cam_subdir if cam_subdir.is_dir() else calib_path
    print(f"Vehicle calibration: {calib_base}")
    for cid in args.camera_ids:
        cam_name = VEHICLE_CAMERAS[cid]["name"]
        res = VEHICLE_CAMERAS[cid]["resolution"]
        w, h = res[0] // args.render_scale, res[1] // args.render_scale
        print(f"  cam{cid} ({cam_name}): {w}x{h}")

    # Discover scenes
    if args.scene_name:
        scenes = [args.scene_name]
    else:
        scenes = sorted([
            d for d in os.listdir(args.model_root)
            if os.path.isfile(os.path.join(args.model_root, d, "chkpnt_fine_30000.pth"))
        ])

    print(f"Scenes to render: {len(scenes)}")

    for scene_name in scenes:
        if scene_name not in ts_map:
            print(f"SKIP {scene_name}: not in timestamp mapping")
            continue

        target_ts_ms = ts_map[scene_name]
        model_path = os.path.join(args.model_root, scene_name)
        scene_id = extract_scene_id(scene_name)

        print(f"\n{scene_name} (scene_id={scene_id}): target timestamp {target_ts_ms}")

        try:
            _, R_w2l, t_w2l = load_scene_transform(
                args.transform_root, scene_id, target_ts_ms
            )
        except FileNotFoundError as e:
            print(f"SKIP {scene_name}: {e}")
            continue

        render_scene(
            model_path=model_path,
            vehicle_calib=args.vehicle_calib,
            camera_ids=args.camera_ids,
            R_w2l=R_w2l,
            t_w2l=t_w2l,
            scene_name=scene_name,
            pipe={"pipe": pipe_args, "hyper": hyper},
            output_dir=args.output_root,
            render_scale=args.render_scale,
            time_val=args.time,
        )


if __name__ == "__main__":
    main()
