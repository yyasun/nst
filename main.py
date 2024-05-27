import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
from utils import Utils

def build_loss(net, opt_img, target_reps, content_idx, style_idxs, config):
    target_content_rep = target_reps[0]
    target_style_rep = target_reps[1]

    feature_maps = net(opt_img)

    current_content_rep = feature_maps[content_idx].squeeze(0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_rep, current_content_rep)

    style_loss = 0.0
    current_style_rep = [Utils.gram_matrix(x) for i, x in enumerate(feature_maps) if i in style_idxs]
    for gram_gt, gram_hat in zip(target_style_rep, current_style_rep):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_rep)

    tv_loss = Utils.total_variation(opt_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(net, optimizer, target_reps, content_idx, style_idxs, config):
    def tuning_step(opt_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(net, opt_img, target_reps, content_idx, style_idxs, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step

def nst(config):
    content_path = os.path.join(config['content_dir'], config['content_img'])
    style_path = os.path.join(config['style_dir'], config['style_img'])

    out_dir = 'combined_' + os.path.split(content_path)[1].split('.')[0] + '_' + os.path.split(style_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_dir'], out_dir)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = Utils.prepare_img(content_path, config['height'], device)
    style_img = Utils.prepare_img(style_path, config['height'], device)

    init_img = Utils.prepare_init_img(content_img, style_img, config['init_method'], device)

    opt_img = Variable(init_img, requires_grad=True)

    net, content_idx, style_idxs = Utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_maps = net(content_img)
    style_maps = net(style_img)

    target_content_rep = content_maps[content_idx].squeeze(0)
    target_style_rep = [Utils.gram_matrix(x) for i, x in enumerate(style_maps) if i in style_idxs]
    target_reps = [target_content_rep, target_style_rep]

    num_iters = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    if config['optimizer'] == 'adam':
        optimizer = Adam((opt_img,), lr=1e1)
        tuning_step = make_tuning_step(net, optimizer, target_reps, content_idx, style_idxs, config)
        for cnt in range(num_iters[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(opt_img)
            with torch.no_grad():
                print(f'Adam | iter: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                Utils.save_and_display(opt_img, dump_path, config, cnt, num_iters[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((opt_img,), max_iter=num_iters['lbfgs'], line_search_fn='strong_wolfe')

        cnt = 0
        def closure():
            nonlocal cnt
            with torch.no_grad():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(net, opt_img, target_reps, content_idx, style_idxs, config)

            optimizer.zero_grad()
            total_loss.backward()

            print(f'L-BFGS | iter: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            Utils.save_and_display(opt_img, dump_path, config, cnt, num_iters[config['optimizer']], should_display=False)
            cnt += 1
            return total_loss

        optimizer.step(closure)

        for _ in range(100):
            optimizer.step(closure)

    return dump_path

if __name__ == "__main__":
    default_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_dir = os.path.join(default_dir, 'content-images')
    style_dir = os.path.join(default_dir, 'style-images')
    output_dir = os.path.join(default_dir, 'output-images')
    img_format = (4, '.jpg')

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img", type=str, help="content image name", default='golden_gate2.jpg')
    parser.add_argument("--style_img", type=str, help="style image name", default='vg_houses.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='adam')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg16')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=100)
    args = parser.parse_args()

    config = vars(args)
    config['content_dir'] = content_dir
    config['style_dir'] = style_dir
    config['output_dir'] = output_dir
    config['img_format'] = img_format

    results_path = nst(config)