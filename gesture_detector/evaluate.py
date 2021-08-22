import argparse
import time
from pathlib import Path
import random
import json
import os

import numpy as np
import torch

import util.misc as utils
from models import build_model

import cv2
import video

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_segment', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--segment_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='chalearn')
    parser.add_argument('--chalearn_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def plot_result_on_frames(detections, frames_path, output_path):
    for i, detection in enumerate(detections):
        im = cv2.imread(os.path.join(frames_path, str(i + 1) + '.jpg'))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if detection:
            cv2.putText(im, 'Gesture detected', (50, 50), font, 1, (127,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(im, 'No Gesture detected', (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_path, str(i + 1) + '.jpg'), im)


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # /model/detr.py build
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    postprocessors = postprocessors['segments']
    # postprocessors.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # load keypoints of a video
    root = '/home/yxz2569'
    keypoint_path = 'ellen_show/keypoints/ellen.json'
    video_keypoints = json.load(open(os.path.join(root, keypoint_path)))
    sample = torch.tensor(video_keypoints['keypoints'])
    samples = utils.nested_tensor_from_tensor_list([sample])
    samples = samples.to(device)

    # target sizes
    target_sizes = samples.tensors.size()[1]
    detections = torch.zeros(target_sizes, dtype=torch.bool)
    target_sizes = torch.tensor([[target_sizes]]).to(device)

    # prediction
    outputs = model(samples)

    # postprocessing
    results = postprocessors(outputs, target_sizes)
    scores = results[0]['scores']
    segments = results[0]['segments']
    threshold = 0.9
    predictions = segments[scores > threshold]

    # generate results on image
    for prediction in predictions:
        detections[prediction[0] : prediction[1]] = True

    plot_result_on_frames(detections, os.path.join(root, 'ellen_show/keypoints/frames'), os.path.join(root, 'ellen_show/keypoints/output'))
    video.simple_frame2video(os.path.join(root, 'ellen_show/keypoints/output'), os.path.join(root, 'ellen_show/keypoints/output/result.avi'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.epochs = 500

    args.hidden_dim = 18  # 18 keypoints
    args.dim_feedforward = 256  # feed forward intermediary in encoder
    args.nheads = 2
    args.num_queries = 10

    args.device = 'cuda'

    args.dataset_file = 'chalearn'
    args.chalearn_path = '/home/yxz2569/chalearn'

    args.batch_size = 16
    args.num_workers = 4

    args.resume = './run/checkpoint0499.pth'
    args.output_dir = './run'
    args.no_aux_loss = True

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)