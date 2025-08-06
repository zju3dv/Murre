# Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
### [Project Page](https://zju3dv.github.io/murre) | [Paper](https://arxiv.org/pdf/2503.14483)

![teaser](./assets/teaser.jpg)

> [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](https://zju3dv.github.io/murre)  
> [Haoyu Guo](https://github.com/ghy0324)<sup>\*</sup>, [He Zhu](https://ada4321.github.io/)<sup>\*</sup>, [Sida Peng](https://pengsida.net), [Haotong Lin](https://haotongl.github.io/), [Yunzhi Yan](https://yunzhiy.github.io/), [Tao Xie](https://github.com/xbillowy), [Wenguan Wang](https://sites.google.com/view/wenguanwang), [Xiaowei Zhou](https://xzhou.me), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/)  
> CVPR 2025 Oral

## Installation

### Clone this repository
```
git clone https://github.com/zju3dv/Murre.git
```

### Create the environment

```
conda create -n murre python=3.10
conda activate murre
```

### Installing dependencies

```
conda install cudatoolkit=11.8 pytorch==2.0.1 torchvision=0.15.2 torchtriton=2.0.0 -c pytorch -c nvidia  # use the correct version of cuda for your system
```

### Installing other requirements

```
pip install -r requirements.txt
```

## Checkpoint

The pretrained model weights can be downloaded from [here](https://drive.google.com/file/d/1gcThkgOQRmjAxhGJRV7SwzwXKBWP1cDa/view?usp=sharing).


## Inference

### Parse SfM output

```
cd sfm_depth
python get_sfm_depth.py --input_sfm_dir ${your_input_path} --output_sfm_dir ${your_output_path} --processing_res ${your_desired_resolution}
```
Make sure that the input is organized in the format of COLMAP results.
You can specify the processing resolution to trade off between inference speed and reconstruction precision.

The parsed sparse depth maps, camera intrinsics, camera poses will be stored in ` ${your_output_path}/sparse_depth`, `${your_output_path}/intrinsic`, and `${your_output_path}/pose` respectively.

### SfM-guided monocular depth estimation

Run the Murre model to perform SfM-guided monocular depth estimation:
```
python run.py --checkpoint ${your_ckpt_path} --input_rgb_dir ${your_rgb_path} --input_sdpt_dir ${your_sparse_depth_path} --output_dir ${your_output_path} --denoise_steps 10 --ensemble_size 5 --processing_res ${your_desired_resolution} --max_depth 10.0
```
For ​indoor scenes, we recommend setting `--max_depth=10.0`. For ​outdoor scenes, consider increasing this value (for example, 80.0).

To filter unreliable SfM depth estimates, adjust:

`--err_thr=${your_error_thresh}` (reprojection error threshold)

`--nviews_thr=${your_nviews_thresh}` (minimum co-visible views)

This ensures robustness by excluding noisy depth values with high errors or insufficient observations.

Make sure that the same processing resolution is used as the first step.

### TSDF fusion

Run the following to perform TSDF fusion on depth maps produced by Murre:

```
python tsdf_fusion.py --image_dir ${your_rgb_path} --depth_dir ${your_depth_path} --intrinsic_dir ${your_intrinsic_path} --pose_dir ${your_pose_path}
```

Please pass in the depth maps produced by Murre and camera parameters parsed in the first step.

Adjust `--res` to balance reconstruction resolution with performance. Set `--depth_max` to clip depth maps based on your scene type (e.g., lower values for indoor scenes, higher for outdoor).

## Evaluation

Please refer to [here](./EVAL.md).

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{guo2025murre,
  title={Multi-view Reconstruction via SfM-guided Monocular Depth Estimation},
  author={Guo, Haoyu and Zhu, He and Peng, Sida and Lin, Haotong and Yan, Yunzhi and Xie, Tao and Wang, Wenguan and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2025},
}
```

## Acknowledgement

We sincerely thank the following excellent projects, from which our work has greatly benefited.

- [Diffusers](https://huggingface.co/docs/diffusers)
- [Marigold](https://marigoldmonodepth.github.io/)
- [COLMAP](https://colmap.github.io/)
- [Detector-Free SfM](https://zju3dv.github.io/DetectorFreeSfM/)
