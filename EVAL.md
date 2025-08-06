## Evaluation Instructions

### DTU & Replica

- **Dataset Download**: Please refer to [MonoSDF](https://github.com/autonomousvision/monosdf#dataset)  
- **Evaluation**: Please refer to [MonoSDF](https://github.com/autonomousvision/monosdf#evaluations)

### ScanNet

- **Dataset Download**: Please refer to [Manhattan-SDF](https://github.com/zju3dv/manhattan_sdf/tree/main#data-preparation)  
- **Evaluation**: Please refer to [Manhattan-SDF](https://github.com/zju3dv/manhattan_sdf/tree/main#evaluation)

### Waymo

We follow [StreetSurf](https://github.com/waymo-research/streetsurf) and select static scenes from the [Waymo dataset](https://waymo.com/open/) (refer to Table 1 in [StreetSurf](https://github.com/waymo-research/streetsurf)). The scene IDs we used are listed as follows:

```python
scene_ids = [
    '003', '019', '036', '069', '081', '126', '139', '140',
    '146', '148', '157', '181', '200', '204', '226', '232',
    '237', '241', '245', '246', '271', '297', '302', '312',
    '314', '362', '482', '495', '524', '527', '753', '780'
]
```

We compute the Root Mean Square Error (RMSE) between predicted depth and LiDAR depth for each frame within each scene, using the following core snippet:

```python
dpt = np.clip(dpt, 0, 80)
msk = lidar_dpt > 0
rmse = np.sqrt(np.mean(np.square(dpt[msk] - lidar_dpt[msk])))
```

Finally, we average the per-frame RMSE across each scene, and report the final metric as the mean RMSE over all selected scenes.

