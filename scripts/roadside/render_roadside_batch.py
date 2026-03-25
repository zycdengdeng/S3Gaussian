"""
Batch render trained S3Gaussian roadside scenes from vehicle cameras at 1280x720.

Transform chain:
  World --(world2lidar)--> VirtualLiDAR --(virtualLidarToCam)--> Camera

Data sources:
  - calib.json: vehicle camera intrinsics + virtualLidarToCam extrinsics
  - transform_json/{scene_id}/*.json: per-scene world2lidar transforms (all timestamps)
  - models/{scene_name}/chkpnt_fine_30000.pth: trained S3Gaussian model

Usage:
  python scripts/roadside/render_roadside_batch.py \
    --model_root /path/to/work_dir/models \
    --calib_json /path/to/calib.json \
    --transform_root /path/to/transform_json \
    --output_root /path/to/work_dir/renders

  Or render a specific scene:
  python scripts/roadside/render_roadside_batch.py \
    --model_root /path/to/work_dir/models \
    --calib_json /path/to/calib.json \
    --transform_root /path/to/transform_json \
    --output_root /path/to/work_dir/renders \
    --scene_name scene003_far
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import json
import glob
import re
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from arguments import ModelHiddenParams, PipelineParams
from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
import torchvision


# Target render resolution
RENDER_W = 1280
RENDER_H = 720


def load_calib(calib_json_path):
    """Load vehicle camera calibrations from calib.json.

    Returns dict: cam_id -> {K (3x3), distor, R_vl2cam (3x3), t_vl2cam (3,), isFish}
    """
    with open(calib_json_path, 'r') as f:
        calib = json.load(f)

    cameras = {}
    for cam_id_str, cam_data in calib["camera"].items():
        cam_id = int(cam_id_str)
        is_fish = cam_data["isFish"]

        # Intrinsic matrix (row-major flattened 3x3)
        K = np.array(cam_data["intri"]).reshape(3, 3)
        distor = np.array(cam_data["distor"])

        # virtualLidarToCam: Rodrigues rotation vector + translation
        vl2cam = cam_data["virtualLidarToCam"]
        rotvec = np.array(vl2cam["rotate"])
        R_vl2cam = Rotation.from_rotvec(rotvec).as_matrix()
        t_vl2cam = np.array(vl2cam["trans"])

        cameras[cam_id] = {
            "K": K,
            "distor": distor,
            "R_vl2cam": R_vl2cam,
            "t_vl2cam": t_vl2cam,
            "isFish": is_fish,
            "name": cam_data.get("name", f"cam{cam_id}"),
        }

    return cameras


def load_scene_transforms(transform_root, scene_id):
    """Load all world2lidar transforms for a scene.

    Args:
        transform_root: root dir containing scene subdirs (001/, 002/, ...)
        scene_id: scene number string (e.g. "003")

    Returns:
        list of (timestamp, R_w2vl (3x3), t_w2vl (3,))
    """
    scene_dir = os.path.join(transform_root, scene_id)
    json_files = glob.glob(os.path.join(scene_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {scene_dir}")

    with open(json_files[0], 'r') as f:
        transforms = json.load(f)

    results = []
    for entry in transforms:
        ts = entry["timestamp"]
        rotvec = np.array(entry["world2lidar"]["rotation"])
        R_w2vl = Rotation.from_rotvec(rotvec).as_matrix()
        t_w2vl = np.array(entry["world2lidar"]["translation"])
        results.append((ts, R_w2vl, t_w2vl))

    return results


def extract_scene_id(scene_name):
    """Extract scene number from name: 'scene003_far' -> '003'."""
    m = re.match(r"scene(\d+)", scene_name)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract scene ID from '{scene_name}'")


def compute_w2c(R_w2vl, t_w2vl, R_vl2cam, t_vl2cam):
    """World -> VirtualLiDAR -> Camera.

    Returns R_stored, T_stored in S3Gaussian convention.
    """
    # Build 4x4 transforms
    T_w2vl = np.eye(4)
    T_w2vl[:3, :3] = R_w2vl
    T_w2vl[:3, 3] = t_w2vl

    T_vl2cam = np.eye(4)
    T_vl2cam[:3, :3] = R_vl2cam
    T_vl2cam[:3, 3] = t_vl2cam

    # World -> Camera = VL2Cam @ W2VL
    T_w2c = T_vl2cam @ T_w2vl

    # S3Gaussian convention: R stored transposed
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
        from argparse import Namespace
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


def render_scene(model_path, cam_calibs, transforms, scene_name,
                 pipe, output_dir, time_val=0.0):
    """Render one scene from all vehicle cameras at all timestamps."""
    gaussians = load_trained_model(model_path, 30000, pipe["hyper"])
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Filter to non-fish cameras only (1280x720)
    render_cams = {cid: c for cid, c in cam_calibs.items() if not c["isFish"]}

    total_renders = len(transforms) * len(render_cams)
    print(f"Rendering {scene_name}: {len(transforms)} timestamps x {len(render_cams)} cameras = {total_renders} views at {RENDER_W}x{RENDER_H}")

    scene_out = os.path.join(output_dir, scene_name)

    with tqdm(total=total_renders, desc=scene_name) as pbar:
        for ts, R_w2vl, t_w2vl in transforms:
            # Timestamp as string for filename
            ts_str = f"{ts:.6f}"

            for cam_id, calib in render_cams.items():
                R_stored, T_stored = compute_w2c(
                    R_w2vl, t_w2vl, calib["R_vl2cam"], calib["t_vl2cam"]
                )

                fx, fy = calib["K"][0, 0], calib["K"][1, 1]
                fovx = focal2fov(fx, RENDER_W)
                fovy = focal2fov(fy, RENDER_H)

                cam = create_minicam(
                    R_stored, T_stored, fovx, fovy,
                    RENDER_W, RENDER_H, time_val,
                )

                with torch.no_grad():
                    render_pkg = render(cam, gaussians, pipe["pipe"], bg_color, stage="fine")

                rendering = render_pkg["render"]

                # Save: {scene}/cam{id}/{timestamp}.png
                cam_dir = os.path.join(scene_out, f"cam{cam_id}")
                os.makedirs(cam_dir, exist_ok=True)
                torchvision.utils.save_image(
                    rendering, os.path.join(cam_dir, f"{ts_str}.png")
                )
                pbar.update(1)

    print(f"Saved to {scene_out}")


def main():
    parser = ArgumentParser(description="Batch render roadside scenes from vehicle cameras at 1280x720")
    pipeline_params = PipelineParams(parser)
    hyper_params = ModelHiddenParams(parser)

    parser.add_argument("--model_root", type=str, required=True,
                        help="Root of trained models (work_dir/models)")
    parser.add_argument("--calib_json", type=str, required=True,
                        help="Path to calib.json")
    parser.add_argument("--transform_root", type=str, required=True,
                        help="Root of transform_json directories")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for rendered images")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Render specific scene (default: all)")
    parser.add_argument("--time", type=float, default=0.0,
                        help="Time value for deformation network (0.0 = first frame)")

    args = parser.parse_args()
    pipe_args = pipeline_params.extract(args)
    hyper = hyper_params.extract(args)

    # Load vehicle camera calibrations
    cam_calibs = load_calib(args.calib_json)
    non_fish = [cid for cid, c in cam_calibs.items() if not c["isFish"]]
    print(f"Loaded {len(cam_calibs)} cameras ({len(non_fish)} non-fish: {sorted(non_fish)})")

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
        model_path = os.path.join(args.model_root, scene_name)
        scene_id = extract_scene_id(scene_name)

        try:
            transforms = load_scene_transforms(args.transform_root, scene_id)
        except FileNotFoundError as e:
            print(f"SKIP {scene_name}: {e}")
            continue

        print(f"\n{scene_name} (scene_id={scene_id}): {len(transforms)} timestamps")

        render_scene(
            model_path=model_path,
            cam_calibs=cam_calibs,
            transforms=transforms,
            scene_name=scene_name,
            pipe={"pipe": pipe_args, "hyper": hyper},
            output_dir=args.output_root,
            time_val=args.time,
        )


if __name__ == "__main__":
    main()
