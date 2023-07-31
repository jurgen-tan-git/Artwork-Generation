# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator
import torchvision.transforms as transforms
import io

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def generate_style_image(dpath: PIL.Image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    checkpoint = "./pretrained/CycleGAN_Generator_50.pt"
    model = torch.jit.load(checkpoint)

    with open(dpath, "rb") as f:
        img = f.read()
    img = io.BytesIO(img)
    im = PIL.Image.open(img).convert("RGB")
    w, h = im.size
    if h or w >= 500:
        scale_factor = 500 / max(h, w)
        height = int(h * scale_factor)
        width = int(w * scale_factor)
        im = transforms.Resize((height, width))(im)

    input = transform(im)
    model.eval()
    output = model(input.unsqueeze(0))
    output = output / 2 + 0.5
    return (transforms.ToPILImage()(output.squeeze(0)))
    
def generate_images(

    dpath: PIL.Image
):
    network_pkl = './pretrained/FFHQ_512.pkl'
    resolution = 512
    
    """
    Generate images using pretrained network pickle.
    """
    seed = random.randint(0,10000)  # pick up a random number
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = dpath


    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:

        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
        print(f)
        
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)


    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = PIL.Image.open(f)
                if image.size != (512,512):
                    image = image.resize((512,512))
                image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        print(f'Processing: {dpath}')
        image = read_image(dpath)
        image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)
        
        mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        output = G(image, mask, z, label, truncation_psi=0.5, noise_mode='const')
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()
        output_image = PIL.Image.fromarray(output, 'RGB')
    return output_image

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
