import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def get_point_cloud(data_dir, data_mode):
    if data_mode == 'mesh':
        data_mesh = o3d.io.read_triangle_mesh(data_dir)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)
    
    elif data_mode == 'pcd':
        data_pcd_o3d = o3d.io.read_point_cloud(data_dir)
        data_pcd = np.asarray(data_pcd_o3d.points)

    else:
        raise ValueError(f'Invalid data mode: {data_mode}')

    return data_pcd


if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to the data')
    parser.add_argument('--gt_dir', type=str, help='path to the gt')
    parser.add_argument('--data_mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--gt_mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--max_dist', type=float, default=20)
    args = parser.parse_args()

    thresh = args.downsample_density
    
    pbar = tqdm(total=7)
    pbar.update(1)
    pbar.set_description(f'get data from {args.data_mode}')
    data_pcd = get_point_cloud(args.data_dir, args.data_mode)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('read GT pcd')
    stl = get_point_cloud(args.gt_dir, args.gt_mode)

    pbar.update(1)
    pbar.set_description('compute data2GT')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute GT2data')
    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    print('+'+'-'*40+'+')
    print('|{:^40}|'.format('Chamfer Distance Evaluation Result'))
    print('+'+'-'*40+'+')
    print('|{:^40}|'.format('data2GT: {:.6f}'.format(mean_d2s)))
    print('|{:^40}|'.format('GT2data: {:.6f}'.format(mean_s2d)))
    print('|{:^40}|'.format('overall: {:.6f}'.format(over_all)))
    print('+'+'-'*40+'+')