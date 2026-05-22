## Setup MURRE

Create python environment:
```
screen -r
conda create -p .conda/envs/murre python=3.10
conda env list
```

Start new jupyter server:
```
./start_jupyter_server.sh [with image 'docker://nvcr.io#nvidia/pytorch:26.03-py3']
```

Open command prompt in the jupyter server and run:
```
conda activate murre
conda install mamba
mamba install cudatoolkit=11.8 pytorch==2.0.1 torchvision=0.15.2 torchtriton=2.0.0 -c pytorch -c nvidia
```

Download MURRE:
```
git clone https://github.com/zju3dv/Murre.git
```

## Majore changes

make change to get_sfm_depth.py in function get_rescale_crop_tgthw-
```
def get_rescale_crop_tgthw(original_res, processing_res):
    original_height, original_width = original_res

    downscale_factor = 1.0
    if processing_res > 0:
        downscale_factor = min(
            processing_res / original_width, processing_res / original_height
        )
        
    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)
    crop_h = new_height - new_height % 8 #16
    crop_w = new_width - new_width % 8 #16
    return downscale_factor, crop_h, crop_w, new_height, new_width
```

add run.py after line 223-
```
    rgb_filename_list, sdpt_filename_list = filter_common_files(rgb_filename_list, sdpt_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Matched {n_images} RGB images.")
    else:
        logging.error(f"No image matched between '{input_rgb_dir}' and '{input_sdpt_dir}'")
        exit(1)
```
and remove line 211-216
and these functions in the beginning-
```
def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

def filter_common_files(set1, set2):
    stems2 = {stem(p) for p in set2}
    filtered1 = [p for p in set1 if stem(p) in stems2]

    stems1 = {stem(p) for p in set1}
    filtered2 = [p for p in set2 if stem(p) in stems1]

    return filtered1, filtered2
```

## Run MURRE

Note: Set --max_depth 10.0 for indoor and 80.0 for outdoor scenes

Running on Scannet++:
```
cd /workspace/minhas/mono_depth/murre/
conda activate murre

mkdir -p /workspace/minhas/dataset/scannetpp/data/0b031f3119/dslr/murre_resized_undistorted_images/

python sfm_depth/get_sfm_depth.py \
    --input_sfm_dir  /workspace/minhas/dataset/scannetpp/data/0b031f3119/dslr/colmap_resized_undistorted_images/ \
    --output_sfm_dir /workspace/minhas/dataset/scannetpp/data/0b031f3119/dslr/murre_resized_undistorted_images/ \
    --processing_res 0

python run.py \
    --checkpoint     /workspace/minhas/mono_depth/murre/murre-ckpt/ \
    --input_rgb_dir  /workspace/minhas/dataset/scannetpp/data/0b031f3119/dslr/resized_undistorted_images/ \
    --input_sdpt_dir /workspace/minhas/dataset/scannetpp/data/0b031f3119/dslr/murre_resized_undistorted_images/sparse_depth/ \
    --output_dir     /workspace/minhas/dataset/scannetpp/data/0b031f3119/dslr/murre_resized_undistorted_images/ \
    --denoise_steps 10 --ensemble_size 5 --processing_res 0 --max_depth 10.0
```

Running on MipNeRF360:
```
cd /workspace/minhas/mono_depth/murre/
conda activate murre

img_id="garden"

mkdir -p /workspace/minhas/dataset/mipnerf/${img_id}/murre_images_2/

python sfm_depth/get_sfm_depth.py \
    --input_sfm_dir  /workspace/minhas/dataset/mipnerf/${img_id}/colmap_images_2/ \
    --output_sfm_dir /workspace/minhas/dataset/mipnerf/${img_id}/murre_images_2/ \
    --processing_res 0

python run.py \
    --checkpoint     /workspace/minhas/mono_depth/murre/murre-ckpt/ \
    --input_rgb_dir  /workspace/minhas/dataset/mipnerf/${img_id}/images_2/ \
    --input_sdpt_dir /workspace/minhas/dataset/mipnerf/${img_id}/murre_images_2/sparse_depth/ \
    --output_dir     /workspace/minhas/dataset/mipnerf/${img_id}/murre_images_2/ \
    --denoise_steps 10 --ensemble_size 5 --processing_res 0 --max_depth 80.0
```


## Running GLOMAP

On Scannet++:

```
cd /workspace/minhas/splat/glomap/build/glomap/
conda activate glomap

mkdir -p glomap_resized_undistorted_images
mkdir -p colmap_resized_undistorted_images

img_id="0a5c013435"

colmap feature_extractor \
    --image_path    /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/resized_undistorted_images/ \
    --database_path /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/glomap_resized_undistorted_images/database.db \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE

#colmap exhaustive_matcher \
colmap sequential_matcher \
    --database_path /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/glomap_resized_undistorted_images/database.db

./glomap mapper \
    --database_path /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/glomap_resized_undistorted_images/database.db \
    --image_path    /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/resized_undistorted_images/ \
    --output_path   /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/glomap_resized_undistorted_images/

colmap model_converter \
    --input_path  /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/glomap_resized_undistorted_images/0/ \
    --output_path /workspace/minhas/dataset/scannetpp/data/${img_id}/dslr/colmap_resized_undistorted_images/ \
    --output_type TXT
```

Running on MipNeRF360:
```
cd /workspace/minhas/splat/glomap/build/glomap/
conda activate glomap

img_id="room"

mkdir -p /workspace/minhas/dataset/mipnerf/${img_id}/glomap_images_2/
mkdir -p /workspace/minhas/dataset/mipnerf/${img_id}/colmap_images_2/

colmap feature_extractor \
    --image_path    /workspace/minhas/dataset/mipnerf/${img_id}/images_2/ \
    --database_path /workspace/minhas/dataset/mipnerf/${img_id}/glomap_images_2/database.db \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE

colmap sequential_matcher \
    --database_path /workspace/minhas/dataset/mipnerf/${img_id}/glomap_images_2/database.db

./glomap mapper \
    --database_path /workspace/minhas/dataset/mipnerf/${img_id}/glomap_images_2/database.db \
    --image_path    /workspace/minhas/dataset/mipnerf/${img_id}/images_2/ \
    --output_path   /workspace/minhas/dataset/mipnerf/${img_id}/glomap_images_2/

colmap model_converter \
    --input_path  /workspace/minhas/dataset/mipnerf/${img_id}/glomap_images_2/0/ \
    --output_path /workspace/minhas/dataset/mipnerf/${img_id}/colmap_images_2/ \
    --output_type TXT

```
