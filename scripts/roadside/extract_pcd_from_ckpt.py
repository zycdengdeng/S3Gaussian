"""Extract point cloud PLY from a training checkpoint (.pth file)."""
import argparse
import os
import sys
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output PLY path (default: same dir as ckpt)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.ckpt), "point_cloud.ply")

    print(f"Loading checkpoint: {args.ckpt}")
    model_params, iteration = torch.load(args.ckpt, map_location="cpu")

    # Unpack model_params following GaussianModel.capture() order
    (active_sh_degree, _xyz, deform_state, _deformation_table,
     _features_dc, _features_rest, _scaling, _rotation, _opacity,
     max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = model_params

    xyz = _xyz.numpy()
    normals = np.zeros_like(xyz)
    f_dc = _features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    f_rest = _features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    opacities = _opacity.numpy()
    scale = _scaling.numpy()
    rotation = _rotation.numpy()

    # Build attribute names
    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        attr_names.append(f'f_dc_{i}')
    for i in range(f_rest.shape[1]):
        attr_names.append(f'f_rest_{i}')
    attr_names.append('opacity')
    for i in range(scale.shape[1]):
        attr_names.append(f'scale_{i}')
    for i in range(rotation.shape[1]):
        attr_names.append(f'rot_{i}')

    dtype_full = [(name, 'f4') for name in attr_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    PlyData([el]).write(args.output)
    print(f"Saved {xyz.shape[0]} points to {args.output}")

if __name__ == "__main__":
    main()
