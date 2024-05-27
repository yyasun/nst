import argparse
import os
from collections import namedtuple
from typing import Tuple

import cv2 as cv
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
from torchvision import models, transforms

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

class Utils:
    def __init__(self, device):
        self.device = device

    def load_image(self, img_path, target_shape=None):
        if not os.path.exists(img_path):
            raise Exception(f'Path does not exist: {img_path}')
        img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

        if target_shape is not None:  # resize section
            if isinstance(target_shape, int) and target_shape != -1:
                current_height, current_width = img.shape[:2]
                new_height = target_shape
                new_width = int(current_width * (new_height / current_height))
                img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
            else:
                img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

        img = img.astype(np.float32)
        img /= 255.0
        return img

    def prepare_img(self, img_path, target_shape):
        img = self.load_image(img_path, target_shape=target_shape)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
        ])

        img = transform(img).to(self.device).unsqueeze(0)

        return img

    def prepare_init_img(self, content_img, style_img, init_method):
        if init_method == 'random':
            gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
            init_img = torch.from_numpy(gaussian_noise_img).float().to(self.device)
        elif init_method == 'content':
            init_img = content_img
        else:
            style_img_resized = self.prepare_img(style_img, np.asarray(content_img.shape[2:]))
            init_img = style_img_resized

        return init_img

    def create_dump_path(self, config):
        out_dir_name = 'combined_' + os.path.split(config['content_img_path'])[1].split('.')[0] + '_' + os.path.split(config['style_img_path'])[1].split('.')[0]
        dump_path = os.path.join(config['output_img_dir'], out_dir_name)
        os.makedirs(dump_path, exist_ok=True)
        return dump_path

    def save_and_display(self, optimizing_img, dump_path, config, img_id, num_iterations, should_display=False):
        saving_freq = config['saving_freq']
        out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
        out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, ch

        # Save image
        if img_id == num_iterations - 1 or (saving_freq > 0 and img_id % saving_freq == 0):
            img_format = config['img_format']
            out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else self.generate_out_img_name(config)
            dump_img = np.copy(out_img)
            dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
            dump_img = np.clip(dump_img, 0, 255).astype('uint8')
            cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])

        # Display image
        if should_display:
            plt.imshow(np.uint8(self.get_uint8_range(out_img)))
            plt.show()

    def generate_out_img_name(self, config):
        prefix = os.path.basename(config['content_img_path']).split('.')[0] + '_' + os.path.basename(config['style_img_path']).split('.')[0]
        if 'reconstruct_script' in config:
            suffix = f'_o_{config["optimizer"]}_h_{str(config["height"])}_m_{config["model"]}{config["img_format"][1]}'
        else:
            suffix = f'_o_{config["optimizer"]}_i_{config["init_method"]}_h_{str(config["height"])}_m_{config["model"]}_cw_{config["content_weight"]}_sw_{config["style_weight"]}_tv_{config["tv_weight"]}{config["img_format"][1]}'
        return prefix + suffix

    def get_uint8_range(self, x):
        if isinstance(x, np.ndarray):
            x -= np.min(x)
            x /= np.max(x)
            x *= 255
            return x
        else:
            raise ValueError(f'Expected numpy array got {type(x)}')

    def gram_matrix(self, x, should_normalize=True):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        if should_normalize:
            gram /= ch * h * w
        return gram

    def total_variation(self, y):
        return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
               torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))