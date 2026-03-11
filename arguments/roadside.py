"""
Configuration for roadside single-frame reconstruction.

Key differences from Waymo NVS config:
- stride=0: use all cameras for training (no test split for single frame)
- start_time=0, end_time=0: single frame
- load_dynamic_mask=False: static scene
- load_panoptic_mask=False: no panoptic segmentation needed
- load_feat_map=False: disable DINOv2 features (can enable later)
"""

ModelParams = dict(
    # Single frame: start=0, end=0
    start_time=0,
    end_time=0,
    original_start_time=0,
    # No train/test split for single frame (use all 4 cameras for training)
    stride=0,
    # Point cloud
    num_pts=500000,
    # Disable masks not available for roadside
    load_sky_mask=False,
    load_panoptic_mask=False,
    load_sam_mask=False,
    load_dynamic_mask=False,
    # DINOv2 features (set True if you want self-supervised features)
    load_feat_map=False,
    # Occupancy grid
    save_occ_grid=True,
    occ_voxel_size=0.4,
    recompute_occ_grid=False,
)

OptimizationParams = dict(
    # For single-frame static scene, fewer iterations may suffice
    coarse_iterations=3000,
    iterations=30000,
    # Densification
    densify_until_iter=15000,
    max_points=300000,
    # Loss weights
    lambda_dssim=0.2,
    lambda_depth=0.5,
)
