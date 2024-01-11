set -e

export PATH="/nfs/project/opt/miniconda3/bin::$PATH"
conda config --add envs_dirs /nfs/project/opt/miniconda3/envs
source activate py38_torch2
export PYTHONPATH=./src

CUDA_VISIBLE_DEVICES=0
python main_pretrain.py \
    --save_ckpt_epoch 10 \
    --mask_ratio 0.5 \
    --input_size 20 --patch_size 2 \
    --embed_dim 768 --depth 12 --num_heads 12 \
    --decoder_embed_dim 256 --decoder_depth 8 --decoder_num_heads 16 \
    --output_dir /nfs/volume-100001-6/zhoutongzt/MGeo/mgeo_torch_ps2enc12dec8 \
    --log_dir /nfs/volume-100001-6/zhoutongzt/MGeo/mgeo_torch_ps2enc12dec8 \
    --data_path /nfs/volume-100001-6/zhoutongzt/MGeo/feature_fm


# /nfs/volume-100001-6/zhoutongzt/MGeo/CIFAR10_images 32x32 p4
# /nfs/volume-100001-6/zhoutongzt/MGeo/LinkFeatureMap_images 20x20 p2
conda deactivate
