# System libs
# import sys
#
# sys.path.append('/workspace/library')
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg


colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred, cfg, args):
    (img, info) = data
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction

    pred_color = colorEncode(data, pred, colors, names, args.result).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(args.result, os.path.splitext(os.path.basename(img_name))[0], 'org&seg.png'))
    Image.fromarray(img).save(
        os.path.join(args.result, os.path.splitext(os.path.basename(img_name))[0], 'org.png'))
    Image.fromarray(pred_color).save(
        os.path.join(args.result, os.path.splitext(os.path.basename(img_name))[0], 'seg.png'))


def test(segmentation_module, loader, gpu, gpu_flag, args, progress):
    segmentation_module.eval()
    pbar = tqdm(total=len(loader))
    process_count = 0
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            if gpu_flag:
                scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                # feed_dict['img_data'] = img
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                if gpu_flag:
                    feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                try:
                    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)
                except RuntimeError as e:
                    print('出现运行错误，假如出现CUDA OUT OF MEMORY则为爆显存，会输出错误分割结果，请尝试用CPU处理该图片。错误信息：', e)

            _, pred = torch.max(scores, dim=1)
            if gpu_flag:
                pred = as_numpy(pred.squeeze(0).cpu())
            else:
                pred = as_numpy(pred.squeeze(0))

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg,
            args
        )
        process_count += 1
        progress.setValue(int(process_count/len(loader)*100))
        pbar.update(1)


def main(cfg, gpu, args, progress):
    gpu_flag = args.gpu_flag
    if gpu_flag and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        print('使用GPU进行语义分割')
    else:
        print('未开启GPU或未安装CUDA环境，设置使用CPU进行语义分割')

    # Network Builders
    if gpu_flag:
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)
    else:
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder, gpu_flag=False)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True, gpu_flag=False)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)
    if gpu_flag:
        segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu, gpu_flag, args, progress)

    print('语义分割处理完成!')


def arg_from_ui(imgs, progress, gpu_flag=None, config_path=None, dir=None, checkpoint=None, result=None):
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'
    # args = {'cfg': 'config/ade20k-resnet50dilated-ppm_deepsup.yaml', 'gpu': 0, 'opts': None, 'gpu_flag': False,
    #         'dir': 'ade20k-resnet50dilated-ppm_deepsup', 'result': 'segmentation', 'checkpoint': 'epoch_20.pth'}
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        default=imgs,
        type=str,
        help="an image paths, or a directory name"
    )
    parser.add_argument(
        "--config_path",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--gpu_flag",
        help="open and close gpu",
        default=True,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dir",
        help="model dir",
        default="ade20k-resnet50dilated-ppm_deepsup",
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--result",
        help="segmentation result dir",
        default="segmentation",
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--checkpoint",
        help="pretrained model checkpoint",
        default="epoch_20.pth",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if gpu_flag is not None:
        args.gpu_flag = gpu_flag
    if config_path:
        args.config_path = config_path
    if dir:
        args.dir = dir
    if checkpoint:
        args.checkpoint = checkpoint
    if result:
        args.result = result

    cfg.merge_from_file(args.config_path)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)  # TODO
    logger.info("Loaded configuration file {}".format(args.config_path))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        args.dir, 'encoder_' + args.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        args.dir, 'decoder_' + args.checkpoint)
    print(cfg.MODEL.weights_encoder)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    print('-----imgs：', args.imgs)
    if os.path.isdir(args.imgs):
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(cfg, args.gpu, args, progress)
