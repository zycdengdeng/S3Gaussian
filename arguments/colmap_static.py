"""
Configuration for static scene reconstruction from COLMAP data.

For scenes with sparse/ directory (cameras.bin, images.bin, points3D.bin).
Tuned for few-view (e.g. 4-view) static reconstruction.

Key settings:
- No depth supervision (lambda_depth=0): COLMAP reader doesn't load depth maps
- No deformation regularization: static scene
- feat_head=False: no DINOv2 features
- 30k iterations matching roadside config
"""

ModelParams = dict(
    # Static scene: single time step
    start_time=0,
    end_time=0,
    original_start_time=0,
    # No train/test split (use all views for training)
    stride=0,
    eval=False,
    # Downscale large images to save GPU memory (2 = half resolution)
    resolution=2,
    # Keep images on CPU, move to GPU per-batch during training
    data_device="cpu",
    # Disable masks not available
    load_sky_mask=False,
    load_panoptic_mask=False,
    load_sam_mask=False,
    load_dynamic_mask=False,
    load_feat_map=False,
)

ModelHiddenParams = dict(
    feat_head=False,
)

OptimizationParams = dict(
    coarse_iterations=3000,
    iterations=30000,
    # Densification
    densify_until_iter=15000,
    max_points=300000,
    # Loss weights
    lambda_dssim=0.2,
    lambda_depth=0,  # No depth supervision for COLMAP data
)
