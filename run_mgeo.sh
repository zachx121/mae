set -e

export PATH="/nfs/project/opt/miniconda3/bin::$PATH"
conda config --add envs_dirs /nfs/project/opt/miniconda3/envs
source activate py38_torch2
export PYTHONPATH=./src

CUDA_VISIBLE_DEVICES=0
python main_pretrain.py \
    --save_ckpt_epoch 20 \
    --mask_ratio 0.5 \
    --input_size 20 --patch_size 1 \
    --embed_dim 1024 --depth 4 --num_heads 16\
    --decoder_embed_dim 64 --decoder_depth 2 --decoder_num_heads 16\
    --output_dir /nfs/volume-100001-6/zhoutongzt/MGeo/mgeo_torch_ps1enc4dec2 \
    --log_dir /nfs/volume-100001-6/zhoutongzt/MGeo/mgeo_torch_ps1enc4dec2 \
    --data_path /nfs/volume-100001-6/zhoutongzt/MGeo/feature_fm


# /nfs/volume-100001-6/zhoutongzt/MGeo/CIFAR10_images 32x32 p4
# /nfs/volume-100001-6/zhoutongzt/MGeo/LinkFeatureMap_images 20x20 p2
conda deactivate
