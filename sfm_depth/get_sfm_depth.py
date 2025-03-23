import os, numpy as np, cv2, matplotlib.pyplot as plt, sys
from tqdm import tqdm
import argparse

from colmap_util import read_model, get_intrinsics, get_hws, get_extrinsic


def get_rescale_crop_tgthw(original_res, processing_res):
    original_height, original_width = original_res
    downscale_factor = min(
        processing_res / original_width, processing_res / original_height
    )
    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)
    crop_h = new_height - new_height % 16
    crop_w = new_width - new_width % 16
    return downscale_factor, crop_h, crop_w, new_height, new_width


def rescale_intrinsic(ixt, scale):
    ixt[:2] *= scale
    return ixt


def read_ixt_ext_hw_pointid(cams, images, points):
    # get image ids
    name2imageid = {img.name:img.id for img in images.values()}
    names = sorted([img.name for img in images.values()])
    imageids = [name2imageid[name] for name in names]

    # ixts
    ixts = np.asarray([get_intrinsics(cams[images[imageid].camera_id]) for imageid in imageids])
    # exts
    exts = np.asarray([get_extrinsic(images[imageid]) for imageid in imageids])
    # hws
    hws = np.asarray([get_hws(cams[images[imageid].camera_id]) for imageid in imageids])
    # point ids
    point_ids = [images[imageid].point3D_ids for imageid in imageids]

    return ixts, exts, hws, point_ids, names


def get_sparse_depth(points3d, ixt, ext, point3D_ids, h, w):
    # sparse_depth: Nx3 array, uvd
    if [id for id in point3D_ids if id != -1] == []:
        return []
    points = np.asarray([points3d[id].xyz for id in point3D_ids if id != -1])
    errs = np.asarray([points3d[id].error for id in point3D_ids if id != -1])
    num_views = np.asarray([len(points3d[id].image_ids) for id in point3D_ids if id != -1])
    sparse_points = points @ ext[:3, :3].T + ext[:3, 3:].T
    sparse_points = sparse_points @ ixt.T
    sparse_points[:, :2] = sparse_points[:, :2] / sparse_points[:, 2:]
    sparse_points = np.concatenate([sparse_points, errs[:, None], num_views[:, None]], axis=1)

    sdpt = np.zeros((h, w, 3))
    for x, y, z, error, num_views in sparse_points:
        x, y = int(x), int(y)
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        sdpt[y, x, 0] = z
        sdpt[y, x, 1] = error
        sdpt[y, x, 2] = num_views

    return sdpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_sfm_dir",
        type=str,
        required=True,
        help="Path to the sfm folder, sfm outputs should be organized in the format of colmap.",
    )

    parser.add_argument(
        "--output_sfm_dir",
        type=str,
        required=True,
        help="Path to the output folder.",
    )

    parser.add_argument(
        "--processing_res",
        type=int,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )

    args = parser.parse_args()
    input_sfm_dir = args.input_sfm_dir
    output_sfm_dir = args.output_sfm_dir
    processing_res = np.array(args.processing_res)

    cams, images, points = read_model(input_sfm_dir)
    ixts, exts, hws, point_ids, names = read_ixt_ext_hw_pointid(cams, images, points)

    os.makedirs(os.path.join(output_sfm_dir, 'sparse_depth'), exist_ok=True)
    os.makedirs(os.path.join(output_sfm_dir, 'intrinsic'), exist_ok=True)
    os.makedirs(os.path.join(output_sfm_dir, 'pose'), exist_ok=True)

    for i, name in tqdm(enumerate(names), desc=f'extracting depth'):
        img_id = name.split('.')[0]
        ixt = ixts[i]
        ext = exts[i]
        original_res = hws[i]
        point_id = point_ids[i]
        scale, crop_h, crop_w, tgt_h, tgt_w = get_rescale_crop_tgthw(original_res, processing_res)
        
        ixt = rescale_intrinsic(ixt, scale)
        sparse_depth = get_sparse_depth(points, ixt, ext, point_id, h=tgt_h, w=tgt_w)
        sparse_depth = sparse_depth[:crop_h, :crop_w]

        np.savetxt(os.path.join(output_sfm_dir, 'intrinsic', f'{img_id}.txt'), ixt)
        np.savetxt(os.path.join(output_sfm_dir, 'pose', f'{img_id}.txt'), ext)
        np.savez_compressed(os.path.join(output_sfm_dir, 'sparse_depth', f'{img_id}.npz'), sparse_depth)
