import logging

logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import torch
import torch.nn as nn
import argparse
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import datetime
import time
from functools import partial
import models_mae
import math

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class HParams:
    img_dir = "/nfs/volume-100001-6/zhoutongzt/MGeo/feature_fm"
    mask_ratio = 0.5
    num_workers = 8
    epochs = 400
    batch_size = 64
    save_ckpt_epoch = 10
    log_step = 1000
    summary_step = 1000

    lr = 1e-3
    input_size = 20
    patch_size = 2
    embed_dim = 768
    depth = 12
    num_heads = 12
    decoder_embed_dim = 256
    decoder_depth = 8
    decoder_num_heads = 12


if __name__ == '__main__':
    hps = HParams()
    ckpt_dir = "/nfs/volume-100001-6/zhoutongzt/MGeo/model_torch/ps{}_enc{}dec{}_maskr{}".format(hps.patch_size, hps.depth, hps.decoder_depth, hps.mask_ratio)
    tbd_writer = SummaryWriter(log_dir=ckpt_dir)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(hps.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(hps.img_dir, transform=transform_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=torch.utils.data.RandomSampler(dataset_train),
        batch_size=hps.batch_size,
        num_workers=hps.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = models_mae.MaskedAutoencoderViT(img_size=hps.input_size,
                                            patch_size=hps.patch_size,
                                            embed_dim=hps.embed_dim,
                                            depth=hps.depth,
                                            num_heads=hps.num_heads,
                                            decoder_embed_dim=hps.decoder_embed_dim,
                                            decoder_depth=hps.decoder_depth,
                                            decoder_num_heads=hps.decoder_num_heads,
                                            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(DEVICE)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=hps.lr)

    logging.info(f"Start training for {hps.epochs} epochs")
    start_time = time.time()
    for epoch in range(0, hps.epochs):
        model.train(True)
        optimizer.zero_grad()
        for step, (batch_samples, lbl) in tqdm(enumerate(data_loader_train)):
            loss, pred, mask = model(batch_samples, mask_ratio=hps.mask_ratio)
            loss.backward()

            if step % hps.summary_step == 0:
                tbd_writer.add_scalar("loss", loss, global_step=step)
                tbd_writer.add_image('input', batch_samples[0], step)
                tbd_writer.add_image('pred', model.unpatchify(pred)[0], step)

            optimizer.step()

        if epoch % hps.save_ckpt_epoch == 0 or epoch + 1 == hps.epochs:
            logging.info(f">>> Saving ckpt at epoch-{epoch}")

            pass

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))

