from .chalearn_keypoint import build as build_chalearn


def build_dataset(video_set, args):
    if args.dataset_file == 'chalearn':
        return build_chalearn(video_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
