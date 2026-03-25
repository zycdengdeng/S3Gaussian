"""
Batch render trained S3Gaussian roadside scenes at 1280x720.

Loads camera parameters from the roadside scene data format
(intrinsics/, extrinsics/, ego_pose/, frame_info.json) and renders
using the same projection logic as render_vehicle.py.

Usage:
  python scripts/roadside/render_roadside_batch.py \
    --data_root /path/to/work_dir/data \
    --model_root /path/to/work_dir/models \
    --output_root /path/to/work_dir/renders \
    --scene_name scene1

  Or render all scenes:
  python scripts/roadside/render_roadside_batch.py \
    --data_root /path/to/work_dir/data \
    --model_root /path/to/work_dir/models \
    --output_root /path/to/work_dir/renders
"""

import torch
import numpy as np
import json
import os
import math
import glob
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from arguments import ModelHiddenParams, PipelineParams
from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
import torchvision


# Target render resolution
RENDER_W = 1280
RENDER_H = 720


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


def create_minicam(R_stored, T_stored, fovx, fovy, width, height, time_val):
    """Create a MiniCam with the given parameters (same as render_vehicle.py)."""
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


def load_scene_cameras(data_dir):
    """Load camera parameters from roadside scene data directory.

    Reads intrinsics/, extrinsics/, ego_pose/, frame_info.json
    and constructs camera poses for all timestamps and cameras.

    Returns:
        list of dicts: [{cam_id, timestamp, R_stored, T_stored, fovx, fovy, time_val}, ...]
    """
    with open(os.path.join(data_dir, "frame_info.json")) as f:
        frame_info = json.load(f)

    num_cameras = frame_info["num_cameras"]
    original_sizes = frame_info["original_sizes"]  # [[h, w], ...]

    # Load per-camera intrinsics and extrinsics (cam-to-ego)
    intrinsics = []
    cam_to_egos = []
    for i in range(num_cameras):
        intr = np.loadtxt(os.path.join(data_dir, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]

        # Scale intrinsics from original resolution to 1280x720
        orig_h, orig_w = original_sizes[i]
        fx_scaled = fx * RENDER_W / orig_w
        fy_scaled = fy * RENDER_H / orig_h
        intrinsics.append((fx_scaled, fy_scaled))

        cam_to_ego = np.loadtxt(os.path.join(data_dir, "extrinsics", f"{i}.txt"))
        cam_to_egos.append(cam_to_ego)

    # Discover timestamps from ego_pose files
    ego_files = sorted(glob.glob(os.path.join(data_dir, "ego_pose", "*.txt")))
    timestamps = [int(Path(f).stem) for f in ego_files]

    if not timestamps:
        raise ValueError(f"No ego_pose files found in {data_dir}")

    # Time normalization (same as readRoadsideInfo)
    start_time = timestamps[0]
    time_length = max(timestamps[-1] - start_time, 1)

    # Reference ego pose (first timestamp)
    ego_start = np.loadtxt(os.path.join(data_dir, "ego_pose", f"{start_time:03d}.txt"))

    cameras = []
    for t in timestamps:
        ego_current = np.loadtxt(os.path.join(data_dir, "ego_pose", f"{t:03d}.txt"))
        ego_to_world = np.linalg.inv(ego_start) @ ego_current
        time_val = (t - start_time) / time_length

        for cam_idx in range(num_cameras):
            # cam2world = ego2world @ cam2ego
            cam2world = ego_to_world @ cam_to_egos[cam_idx]

            # world2cam
            w2c = np.linalg.inv(cam2world)

            # S3Gaussian convention: R stored transposed
            R_stored = w2c[:3, :3].T
            T_stored = w2c[:3, 3]

            fx, fy = intrinsics[cam_idx]
            fovx = focal2fov(fx, RENDER_W)
            fovy = focal2fov(fy, RENDER_H)

            cameras.append({
                "cam_id": cam_idx,
                "timestamp": t,
                "R_stored": R_stored,
                "T_stored": T_stored,
                "fovx": fovx,
                "fovy": fovy,
                "time_val": time_val,
            })

    return cameras


def render_scene(data_dir, model_path, output_dir, pipe, hyper_args, iteration=30000):
    """Render a single scene: all timestamps x all cameras at 1280x720."""
    scene_name = os.path.basename(data_dir)

    if not os.path.exists(os.path.join(model_path, f"chkpnt_fine_{iteration}.pth")):
        print(f"SKIP {scene_name}: no checkpoint found")
        return

    # Load model
    gaussians = load_trained_model(model_path, iteration, hyper_args)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Load cameras from scene data
    cameras = load_scene_cameras(data_dir)
    print(f"Rendering {scene_name}: {len(cameras)} views at {RENDER_W}x{RENDER_H}")

    scene_out = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    for cam_info in tqdm(cameras, desc=scene_name):
        cam = create_minicam(
            cam_info["R_stored"], cam_info["T_stored"],
            cam_info["fovx"], cam_info["fovy"],
            RENDER_W, RENDER_H, cam_info["time_val"],
        )

        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe, bg_color, stage="fine")

        rendering = render_pkg["render"]

        # Save: {scene}/{timestamp:03d}_{cam_id}.png
        fname = f"{cam_info['timestamp']:03d}_{cam_info['cam_id']}.png"
        torchvision.utils.save_image(rendering, os.path.join(scene_out, fname))

    print(f"Saved {len(cameras)} renders to {scene_out}")


def main():
    parser = ArgumentParser(description="Batch render roadside scenes at 1280x720")
    pipeline_params = PipelineParams(parser)
    hyper_params = ModelHiddenParams(parser)

    parser.add_argument("--data_root", type=str, required=True,
                        help="Root of converted scene data (work_dir/data)")
    parser.add_argument("--model_root", type=str, required=True,
                        help="Root of trained models (work_dir/models)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for rendered images")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Render specific scene (default: all)")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Checkpoint iteration to load")

    args = parser.parse_args()
    pipe = pipeline_params.extract(args)
    hyper = hyper_params.extract(args)

    if args.scene_name:
        scenes = [args.scene_name]
    else:
        # Find all scenes with frame_info.json
        scenes = sorted([
            d for d in os.listdir(args.data_root)
            if os.path.isfile(os.path.join(args.data_root, d, "frame_info.json"))
        ])

    print(f"Scenes to render: {len(scenes)}")

    for scene_name in scenes:
        data_dir = os.path.join(args.data_root, scene_name)
        model_path = os.path.join(args.model_root, scene_name)
        render_scene(data_dir, model_path, args.output_root, pipe, hyper,
                     iteration=args.iteration)


if __name__ == "__main__":
    main()
