set -e

export PATH="/nfs/project/opt/miniconda3/bin::$PATH"
conda config --add envs_dirs /nfs/project/opt/miniconda3/envs
source activate py38_torch2
export PYTHONPATH=./src

CUDA_VISIBLE_DEVICES=0
python main_pretrain.py \
    --output_dir /nfs/volume-100001-6/zhoutongzt/MGeo/mae_torch_output \
    --log_dir /nfs/volume-100001-6/zhoutongzt/MGeo/mae_torch_output \
    --mask_ratio 0.5 \
    --input_size 20 \
    --patch_size 2 \
    --data_path /nfs/volume-100001-6/zhoutongzt/MGeo/LinkFeatureMap_images

# /nfs/volume-100001-6/zhoutongzt/MGeo/CIFAR10_images 32x32 p4
# /nfs/volume-100001-6/zhoutongzt/MGeo/LinkFeatureMap_images 20x20 p2
conda deactivate
