# Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
### [Project Page](https://zju3dv.github.io/murre) | [Paper](https://arxiv.org/pdf/2503.14483)

![teaser](./assets/teaser.jpg)

> [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](https://zju3dv.github.io/murre)  
> [Haoyu Guo](https://github.com/ghy0324)<sup>\*</sup>, [He Zhu](https://ada4321.github.io/)<sup>\*</sup>, [Sida Peng](https://pengsida.net), [Haotong Lin](https://haotongl.github.io/), [Yunzhi Yan](https://yunzhiy.github.io/), [Tao Xie](https://github.com/xbillowy), [Wenguan Wang](https://sites.google.com/view/wenguanwang), [Xiaowei Zhou](https://xzhou.me), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/)  
> CVPR 2025

## Installation

### Clone this repository
```
git clone https://github.com/zju3dv/Murre.git
```

### Set up the environment

You can create a conda environment named 'murre' by running:
```
conda env create -f environment.yml
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
python run.py --checkpoint ${your_ckpt_path} --input_rgb_dir ${your_rgb_path} --input_sdpt_dir ${your_sparse_depth_path} --output_dir ${your_output_path} --denoise_steps 10 --ensemble_size 5 --processing_res ${your desired resolution}
```

Make sure that the same proccesing resolution is used as the first step.

### TSDF fusion

Run the following to perform TSDF fusion on depth maps produced by Murre:

```
python tsdf_fusion.py --image_dir ${your_rgb_path} --depth_dir ${your_depth_path} --intrinsic_dir ${your_intrinsic_path} --pose_dir ${your_pose_path}
```

Please pass in the depth maps produced by Murre and camera parameters parsed in the first step.

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
