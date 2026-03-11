"""
Configuration for roadside multi-frame reconstruction.

Key differences from single-frame:
- start_time=0, end_time=-1: use all frames (auto-detected)
- load_dynamic_mask=True: use 3D bbox projected masks
- feat_head=False: no DINOv2 features
- Deformation network enabled for dynamic objects
"""

ModelParams = dict(
    # Multi-frame: end_time=-1 means use all available frames
    start_time=0,
    end_time=-1,
    original_start_time=0,
    # Train/test split: every 5th frame is test
    stride=5,
    # Point cloud
    num_pts=500000,
    # Enable dynamic masks (from 3D bbox annotations)
    load_dynamic_mask=True,
    # Disable other masks
    load_sky_mask=False,
    load_panoptic_mask=False,
    load_sam_mask=False,
    load_feat_map=False,
    # Occupancy grid
    save_occ_grid=True,
    occ_voxel_size=0.4,
    recompute_occ_grid=False,
)

ModelHiddenParams = dict(
    # Disable DINOv2 feature head
    feat_head=False,
    # Enable deformation for dynamic objects
    no_dx=False,
    no_ds=True,
    no_dr=True,
    no_do=True,
    no_dshs=False,
)

OptimizationParams = dict(
    # Two-stage training
    coarse_iterations=5000,
    iterations=30000,
    # Densification
    densify_until_iter=15000,
    max_points=500000,
    # Loss weights
    lambda_dssim=0.2,
    lambda_depth=0.5,
    # Deformation regularization
    lambda_dx=0.001,
    lambda_dshs=0.001,
)
