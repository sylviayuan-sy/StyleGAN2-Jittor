from train import get_config
from training.misc import define_G
from tqdm import tqdm
import jittor as jt
import io
import os
import PIL
import numpy as np
from subprocess import run

def generate():
    cfg = get_config()
    G_ema = define_G(cfg)
    G_ema.load(cfg.g_ema_pretrained)
    if cfg.gpu:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    for i in tqdm(range(cfg.eval_size // cfg.batch_size + 1)):
        sample_z = jt.randn(cfg.batch_size, 512)
        eval_images, _ = G_ema([sample_z])
        for j in range(eval_images.shape[0]):
            jt.save_image(eval_images[j], 
                      f'/mnt/disk/yuanlu/eval/FFHQ/{i*cfg.batch_size + j}.png',
                      nrow=1,
                      normalize=True,
                      range=(-1, 1),
            )
    run('python3.7 -m pytorch_fid /mnt/disk/yuanlu/eval/FFHQ_orig /mnt/disk/yuanlu/eval/FFHQ --device cuda:0'.split())
    
    
if __name__ == "__main__":
    generate()