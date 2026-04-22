# S3Gaussian Roadside Pipeline — Context for New Sessions

## Overview

This repo is a fork of [S3Gaussian](https://github.com/hzxie/S3Gaussian) (3D Gaussian Splatting with deformation network), adapted for **roadside infrastructure camera** scenes. The pipeline:

1. Takes COLMAP sparse reconstructions from roadside cameras
2. Converts to S3Gaussian's "roadside" data format
3. Trains 3DGS models (30K iterations)
4. Renders trained models from **vehicle camera viewpoints** (7 cameras) at 1280x720

## Cloud Server Data Paths

All data lives on the cloud server (not this repo):

```
# Source COLMAP data (original, read-only)
/mnt/zyc_wzh/SparseGS/data/car_road/
    scene003_far/
        images/          # original photos from roadside cameras
        sparse/0/        # cameras.bin, images.bin, points3D.ply (COLMAP output)
    scene003_middle/
    scene003_near/
    scene004_far/
    ...                  # ~54 scenes total (19 scene groups x near/middle/far)

# Working directory (all outputs go here)
/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/
    data/                # Converted roadside-format data (symlinks to original images)
        scene003_far/
            images/      # symlinks: 000_0.jpg, 000_1.jpg, ...
            intrinsics/  # {cam_id}.txt (fx fy cx cy)
            extrinsics/  # {cam_id}.txt (4x4 camera-to-world)
            ego_pose/    # 000.txt (identity for single-frame)
            lidar/       # 000.bin (Nx10 float32, XYZ from COLMAP points3D)
            frame_info.json
    models/              # Trained 3DGS checkpoints
        scene003_far/
            chkpnt_fine_30000.pth   # Final checkpoint (indicates training complete)
            cfg_args
            point_cloud/
    renders/             # Vehicle-view rendered images
        scene003_far/
            FN.png FW.png FL.png FR.png RL.png RR.png RN.png
    logs/                # Training logs per scene
    logs_render/         # Rendering logs per scene

# Vehicle camera calibration (YAML format)
/mnt/car_road_data_TianJin/support_info/NoEER705_v3/
    camera/
        camera_01_intrinsics.yaml   # K, D, width, height, type (pinhole/fisheye)
        camera_01_extrinsics.yaml   # quaternion (xyzw) + translation (cam->lidar)
        camera_02_intrinsics.yaml
        ...                         # cameras 01-07

# World2Lidar transforms (per scene, per timestamp)
/mnt/car_road_data_TianJin/support_info/transform_json/
    003/                 # scene_id (3-digit number from scene name)
        *.json           # array of {timestamp, world2lidar: {rotation (rotvec), translation}}
    004/
    ...
```

## Scene Naming Convention

Scenes are named `scene{NNN}_{distance}` where:
- `NNN` = 3-digit scene group ID (003, 004, 009, 015, 020, 031, 035, 039, 050, 055, 056, 059, 063, 076, 082, 085, 086, 088)
- `distance` = `near`, `middle`, or `far` (three capture distances per scene group)

Each scene is a **single-frame static reconstruction** from ~4 roadside pinhole cameras.

## Key Scripts (all in `scripts/roadside/`)

| Script | Purpose |
|--------|---------|
| `convert_colmap_to_roadside.py` | COLMAP sparse/0/ → roadside format (symlinks, no copy) |
| `run_batch_colmap.sh` | Batch: convert + parallel train all scenes |
| `run_batch_render.sh` | Batch: parallel vehicle-view rendering |
| `render_roadside_batch.py` | Core rendering: vehicle camera projection chain |
| `scene_timestamps.json` | Maps scene_name → timestamp (ms) for world2lidar lookup |

## Training Commands

```bash
# Full batch (convert + train all unconverted scenes)
bash scripts/roadside/run_batch_colmap.sh <GPU_ID> /mnt/zyc_wzh/SparseGS/data/car_road \
    /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap <PARALLEL>

# Train specific scenes manually
cd /path/to/S3Gaussian
CUDA_VISIBLE_DEVICES=1 python train.py \
    -s /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/data/scene056_middle \
    --model_path /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/models/scene056_middle \
    --expname roadside \
    --configs arguments/roadside.py
```

Training config (`arguments/roadside.py`):
- Single frame: `start_time=0, end_time=0`
- All cameras for training: `stride=0`
- 30K iterations, coarse 3K + fine 27K
- No sky/panoptic/dynamic masks, no DINOv2 features
- Depth maps generated internally from LiDAR projection (not external input)

## Rendering Commands (Vehicle Viewpoints)

```bash
# Batch render all trained scenes
bash scripts/roadside/run_batch_render.sh <GPU_ID> \
    /mnt/car_road_data_TianJin/support_info/NoEER705_v3 \
    /mnt/car_road_data_TianJin/support_info/transform_json \
    /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap <PARALLEL>

# Render specific scene
CUDA_VISIBLE_DEVICES=1 python scripts/roadside/render_roadside_batch.py \
    --model_root /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/models \
    --vehicle_calib /mnt/car_road_data_TianJin/support_info/NoEER705_v3 \
    --transform_root /mnt/car_road_data_TianJin/support_info/transform_json \
    --output_root /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/renders \
    --scene_name scene056_middle
```

## Vehicle Camera Rendering — Transform Chain

```
Vehicle Camera --(cam2lidar)--> Vehicle LiDAR --(inv world2lidar)--> World (virtualLiDAR)

T_c2w = inv(T_w2l) @ T_c2l
T_w2c = inv(T_c2w)
```

- **cam2lidar**: from vehicle YAML extrinsics (quaternion xyzw → rotation matrix)
- **world2lidar**: from transform_json, matched by timestamp from `scene_timestamps.json`
- **Rendering**: equivalent to undistorted pinhole camera (D used only to compute new_K via `cv2.getOptimalNewCameraMatrix`, 3DGS renders with pinhole projection, no actual distortion applied to output)
- **Output resolution**: 1280x720 (scaled from original calibrated resolution)

7 vehicle cameras:

| ID | Name | Original Resolution | Type |
|----|------|-------------------|------|
| 1  | FN (front narrow)  | 3840x2160 | pinhole |
| 2  | FW (front wide)    | 3840x2160 | pinhole |
| 3  | FL (front left)    | 3840x2160 | pinhole |
| 4  | FR (front right)   | 3840x2160 | pinhole |
| 5  | RL (rear left)     | 1920x1080 | pinhole |
| 6  | RR (rear right)    | 1920x1080 | pinhole |
| 7  | RN (rear narrow)   | 1920x1080 | pinhole |

## Data Format Details

### Roadside format (S3Gaussian input)

```
scene_dir/
    images/          # {frame:03d}_{cam_id}.{jpg|png}  (frame=000 for single-frame)
    intrinsics/      # {cam_id}.txt → 4 values: fx fy cx cy
    extrinsics/      # {cam_id}.txt → 4x4 camera-to-world matrix
    ego_pose/        # {frame:03d}.txt → 4x4 identity (single-frame static)
    lidar/           # {frame:03d}.bin → Nx10 float32: origins[3] + points[3] + unused[3] + laser_id[1]
    frame_info.json  # {"scene_type": "roadside", "num_cameras": N, "original_sizes": [...]}
```

### Scene type dispatch (`scene/__init__.py`)

The loader checks for `frame_info.json` → reads `scene_type` field:
- `"roadside"` → calls `readRoadsideInfo` (no OPENCV2DATASET transform, single-frame support)
- Otherwise → Waymo loader

### LiDAR bin format

Nx10 float32 array. For COLMAP-converted data:
- Columns 0-2: origins (zeros, since COLMAP points have no sensor origin)
- Columns 3-5: XYZ coordinates (from COLMAP points3D.ply)
- Columns 6-8: unused (zeros)
- Column 9: laser_id (zeros)

Depth maps are generated internally during training by projecting LiDAR points onto camera planes — no external depth input needed.

## Known Issues & Fixes Applied

1. **Shell worker distribution bug** (FIXED): `IFS='\n' read -ra` only reads the first line. Both `run_batch_colmap.sh` and `run_batch_render.sh` now use `mapfile -t` instead.

2. **OOM with too many parallel workers**: On 80GB GPU, use 5-8 parallel workers max. 18 parallel will OOM.

3. **COLMAP points3D format**: The converter tries PLY first (most reliable), then .bin, then .txt. Our data has `points3D.ply` in `sparse/0/`.

4. **No world coordinate transform during training**: COLMAP points and camera poses are used as-is in their original coordinate system (virtualLiDAR / world frame). The world2lidar transform is only needed at render time for vehicle viewpoint projection.

5. **`ds-points3d.ply`** is a training byproduct (downsampled point cloud with random colors), not an input file. The actual input points come from `lidar/000.bin` which contains COLMAP's `points3D.ply` coordinates.

## Useful Check Commands

```bash
# List all trained scenes
find /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/models -maxdepth 2 \
    -name "chkpnt_fine_30000.pth" | sed 's|.*/models/||;s|/.*||' | sort

# List all rendered scenes
ls /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/renders/

# Count images in a rendered scene
ls /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/renders/scene003_far/

# Check training log for errors
tail -50 /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/logs/scene003_far.log
```
