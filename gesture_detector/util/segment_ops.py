"""
Utilities for video segments manipulation and GIoU.
"""
import torch


def segment_SL_to_SE(x):
    # [start, length] to [start, end]
    start, length = x.unbind(-1)
    b = [start, start + length]
    return torch.stack(b, dim=-1)


def segment_iou(segments1, segments2):
    area1 = segments1[:, 1] - segments1[:, 0]
    area2 = segments2[:, 1] - segments2[:, 0]

    # left and right of the intersection
    l = torch.max(segments1[:, None, 0], segments2[:, 0])  # [N,M]
    r = torch.min(segments1[:, None, 1], segments2[:, 1])  # [N,M]

    inter = (r - l).clamp(min=0)  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_segment_iou(segments1, segments2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The segments should be in [start, end] format

    Returns a [N, M] pairwise matrix, where N = len(segments1)
    and M = len(segments2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segments1[:, 1] >= segments1[:, 0]).all()
    assert (segments2[:, 1] >= segments2[:, 0]).all()
    iou, union = segment_iou(segments1, segments2)

    # left and right of the smallest convex hull
    l = torch.min(segments1[:, None, 0], segments2[:, 0])
    r = torch.max(segments1[:, None, 1], segments2[:, 1])

    area = (r - l).clamp(min=0)  # [N,M]
    return iou - (area - union) / area