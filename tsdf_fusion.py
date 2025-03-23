import os, numpy as np, cv2, matplotlib.pyplot as plt, trimesh, argparse
from tqdm import tqdm, trange
# import os, numpy as np, trimesh
import open3d as o3d
from sklearn.neighbors import KDTree

import open3d as o3d
import open3d.core as o3c


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Input image directory.')
    parser.add_argument('--depth_dir', type=str, help='Input depth directory. Depth maps will be fused according be camera parameters.')
    parser.add_argument('--intrinsic_dir', type=str, help='Input camera intrinsics directory.')
    parser.add_argument('--pose_dir', type=str, help='Input camera pose directory.')
    parser.add_argument('--output_dir', type=str, default='output_mesh', help='Output directory of the fused mesh.')
    parser.add_argument('--save_tag', type=str, default='demo', help='Mesh file name to be saved.')
    parser.add_argument('--res', type=float, default=10., help='Resolution of the fused geometry.')
    parser.add_argument('--depth_max', type=float, default=9., help='Maximum depth values where the depth maps will be clipped.')
    args = parser.parse_args()

    image_dir = args.image_dir
    depth_dir = args.depth_dir
    intrinsic_dir = args.intrinsic_dir
    pose_dir = args.pose_dir
    output_dir = args.output_dir
    save_tag = args.save_tag
    res = args.res
    depth_max = args.depth_max

    ixt_files = sorted(os.listdir(intrinsic_dir))
    ixts = []
    for ixt_file in ixt_files:
        ixts.append(np.loadtxt(os.path.join(intrinsic_dir, ixt_file)))

    ext_files = sorted(os.listdir(pose_dir))
    exts = []
    for ext_file in ext_files:
        exts.append(np.loadtxt(os.path.join(pose_dir, ext_file)))

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('_pred.npy')])
    dpts = []
    for depth_file in depth_files:
        dpts.append(np.load(os.path.join(depth_dir, depth_file)))

    h, w = round(ixts[0][1, 2] * 2), round(ixts[0][0, 2] * 2)
    image_files = sorted(os.listdir(image_dir))
    imgs = []
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_dir, image_file))
        img = cv2.resize(img, (w, h))
        crop_h, crop_w = h - h % 16, w - w % 16
        img = img[:crop_h, :crop_w, :]
        imgs.append(img)

    voxel_size = res / 512.
    depth_scale=1.0

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d.core.Device('CUDA:0')
    )

    intrinsic = ixts[0].copy()
    intrinsic = o3c.Tensor(intrinsic[:3, :3], o3d.core.Dtype.Float64)
    color_intrinsic = depth_intrinsic = intrinsic

    for i in trange(len(dpts), desc=f'tsdf integrate'):
        extrinsic = exts[i]
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)
        img = imgs[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img = o3d.t.geometry.Image(img).cuda()
        dpt = dpts[i]
        depth = dpt.astype(np.float32)
        if ((depth > 0) & (depth < depth_max)).sum() < 50:
            print(i, end=',')
            continue
        
        depth = o3d.t.geometry.Image(depth).cuda()
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic, extrinsic, depth_scale, depth_max)

        vbg.integrate(frustum_block_coords, depth, img,
            depth_intrinsic, color_intrinsic, extrinsic,
            depth_scale, depth_max)
        
    mesh_no_check = vbg.extract_triangle_mesh(weight_threshold=0.0).to_legacy()

    check_threshold = 3

    print(f'resolution: {res}, check threshold: {check_threshold}')

    mesh_check = vbg.extract_triangle_mesh(weight_threshold=float(check_threshold)).to_legacy()
    vertices_no_check = np.asarray(mesh_no_check.vertices)
    vertices_check = np.asarray(mesh_check.vertices)
    assert nn_correspondance(vertices_no_check, vertices_check).max() == 0

    nn = nn_correspondance(vertices_check, vertices_no_check)
    msk = nn != 0
    visible_num = np.zeros((msk.sum(), ), np.int32)

    for i, ext in tqdm(enumerate(exts)):
        ixt = np.eye(4)
        ixt[:3, :3] = ixts[i].copy()
        homo_points = np.concatenate([vertices_no_check[msk], np.ones((msk.sum(), 1), np.float32)], axis=1)
        pt = (ixt @ (ext @ homo_points.T))[:3]
        u = pt[0] / pt[2]
        v = pt[1] / pt[2]
        z = pt[2]
        valid = ((z > 0) & (u >= 0) & (u < 1200) & (v >= 0) & (v < 680))
        visible_num += valid.astype(np.int32)

    msk_keep = (~msk).copy()
    msk_keep[msk] = visible_num <= check_threshold

    m = trimesh.Trimesh(vertices=vertices_no_check, faces=np.asarray(mesh_no_check.triangles), process=False)
    msk_keep_face = msk_keep[np.asarray(mesh_no_check.triangles)].all(-1)
    m.update_vertices(msk_keep)
    m.update_faces(msk_keep_face)       
    m.export(f'{output_dir}/{save_tag}.obj')

    print(f'Done! Mesh saved to {output_dir}/{save_tag}.obj')