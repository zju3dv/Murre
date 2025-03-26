import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from murre.pipeline import MurrePipeline

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
SDPT_EXTENSION_LIST = [".npz"]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Murre."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt",
        help="Checkpoint path.",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--input_sdpt_dir",
        type=str,
        required=True,
        help="Path to the sparse depth map folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    parser.add_argument(
        "--scale_invariant",
        action="store_true",
        help="Whether the diffusion model outputs scale-invariant depth.",
    )

    parser.add_argument(
        "--shift_invariant",
        action="store_true",
        help="Whether the diffusion model outputs shift-invariant depth.",
    )

    # sparse depth
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Maximum depth value(larger depth will be clipped at max_depth).",
    )

    parser.add_argument(
        "--err_thr",
        type=float,
        default=None,
        help="SfM depth values with error higher than err_thr will be filtered.",
    )

    parser.add_argument(
        "--nviews_thr",
        type=int,
        default=None,
        help="SfM depth values with number of visible views fewer than nviews_thr will be filtered.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    input_sdpt_dir = args.input_sdpt_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    max_depth = args.max_depth
    err_thr = args.err_thr
    nviews_thr = args.nviews_thr

    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    scale_invariant = args.scale_invariant
    shift_invariant = args.shift_invariant
    err_thr = args.err_thr
    nviews_thr = args.nviews_thr

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    # rgb
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)
    # sparse depth
    sdpt_filename_list = glob(os.path.join(input_sdpt_dir, "*"))
    sdpt_filename_list = [
        f for f in sdpt_filename_list if os.path.splitext(f)[1].lower() in SDPT_EXTENSION_LIST
    ]
    sdpt_filename_list = sorted(sdpt_filename_list)
    if not len(sdpt_filename_list) == n_images:
        logging.error(f'Number of sparse depth maps({len(sdpt_filename_list)}) is not the same as that of images({n_images})')
        exit(1)
    

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe: MurrePipeline = MurrePipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    # force specily
    pipe.scale_invariant = scale_invariant
    pipe.shift_invariant = shift_invariant
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path, sdpt_path in tqdm(zip(rgb_filename_list, sdpt_filename_list), desc="Estimating depth", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)
            # Read input sparse depth
            sdpt = np.load(sdpt_path, allow_pickle=True)['arr_0'].astype(np.float32)
            sdpt, err, nviews = sdpt[..., 0], sdpt[..., 1], sdpt[..., 2]
            if np.isnan(sdpt).any(): sdpt[np.isnan(sdpt)] = 0
            input_sparse_depth = np.clip(sdpt, None, max_depth)
            if err_thr is not None:
                sdpt[err > err_thr] = 0.
            if nviews_thr is not None:
                sdpt[nviews <= nviews_thr] = 0.

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out = pipe(
                input_image,
                input_sparse_depth,
                max_depth=max_depth,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                batch_size=batch_size,
                model_dtype=dtype,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            depth_pred: np.ndarray = pipe_out.depth_np  # NOTE: depth here should be re-normed and aligned
            depth_colored: Image.Image = pipe_out.depth_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            # Save as 16-bit uint png
            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # Colorize
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            depth_colored.save(colored_save_path)
