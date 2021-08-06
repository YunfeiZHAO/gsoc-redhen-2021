"""
author: Yunfei
"""
import os
from pathlib import Path
from collections import defaultdict
import time
import json
import itertools
import tqdm

import cv2
import numpy as np

import torch


# Chalearn dataset
class ChalearnKeypointDataset:
    def __init__(self, annotation_file_path=None):
        """
        Constructor of Chalear Continuous Gesture Recognition dataset anayse https://chalearnlap.cvc.uab.cat/dataset/22/description/
        :param annotation_file_path (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset, self.segments, self.videos = dict(), dict(), dict()
        self.videoToSegments = defaultdict(list)
        if annotation_file_path is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file_path, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.create_index()

    def create_index(self):
        # create index
        print('creating index...')
        segments, videos = {}, {}
        videoToSegments = defaultdict(list)
        if 'segments' in self.dataset:
            for segment in self.dataset['segments']:
                videoToSegments[segment['video_id']].append(segment)
                segments[segment['id']] = segment

        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                videos[video['id']] = video
        print('index created!')

        # create class members
        self.segments = segments
        self.videoToSegments = videoToSegments
        self.videos = videos

    def getSegIds(self, videoIds=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param videoIds  (int array)     : get anns for given videos
        :return: ids (int array)       : integer array of ann ids
        """
        videoIds = videoIds if isinstance(videoIds, list) else [videoIds]
        if len(videoIds) == 0:
            segments = self.dataset['segments']
        else:
            if not len(videoIds) == 0:
                lists = [self.videoToSegments[videoId] for videoId in videoIds if videoId in self.videoToSegments]
                segments = list(itertools.chain.from_iterable(lists))
        ids = [segment['id'] for segment in segments]
        return ids

    def loadSegs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if isinstance(ids, list):
            return [self.segments[id] for id in ids]
        else:
            return [self.segments[ids]]

    def loadVideos(self, ids=[]):
        """
        Load Videos with the specified ids.
        :param ids (int array)       : list of string ids specifying video
        :return:  video (object array) : loaded video objects
        """
        if isinstance(ids, list):
            return [self.videos[id] for id in ids]
        else:
            return [self.videos[ids]]

    def generate_label_on_frames_one_video(self, video_id, save_path):
        """Add a lebel on video frames where we have a gesture"""
        os.makedirs(save_path, exist_ok=True)
        video_data = self.loadVideos(video_id)[0]
        video_length = video_data['videoLenghth']
        frames_path = video_data['frame_path']
        SegIds = self.getSegIds(video_id)
        segments = self.loadSegs(SegIds)
        starts = []
        ends = []
        for segment in segments:
            start, end = segment['start_end']
            starts.append(start)
            ends.append(end)
        for i in range(video_length):
            frame_id = i + 1
            frame_path = os.path.join(frames_path, str(frame_id) + '.jpg')
            frame = cv2.imread(frame_path)
            if frame_id in starts:
                draw_text(frame, 'Gesture start', (0, 255, 0))
            if frame_id in ends:
                draw_text(frame, 'Gesture end', (255, 0, 0))
            cv2.imwrite(os.path.join(save_path, str(frame_id) + '.jpg'), frame)


# Chalearn dataset loader
class ChalearnLoader:
    def __init__(self, root, ann_file='train_annotations.json'):
        ann_file = os.path.join(root, 'annotations', ann_file)
        self.dataset = ChalearnKeypointDataset(ann_file)
        self.ids = list(sorted(self.dataset.videos.keys()))
        self.root = root

    def _load_keypoint(self, id: str):
        keypoint_path = self.dataset.loadVideos(id)[0]['keypoint_path']
        keypoints = json.load(open(keypoint_path))
        return keypoints

    def _load_segments(self, id):
        return self.dataset.loadSegs(self.dataset.getSegIds(id))

    def __getitem__(self, idx, prepare=True):
        video_id = self.ids[idx]
        video = self._load_keypoint(video_id)
        video['centers'] = torch.tensor(video['centers'])
        video['keypoints'] = torch.tensor(video['keypoints'])

        segments = self._load_segments(video_id)
        target = {'video_id': video_id, 'segments': segments}
        if prepare:
            video, target = self._prepare_for_detector(video, target)
        return video, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self) -> str:
        head = 'chalearn dataset'
        body = ["Number of videos: {}".format(len(self.ids))]
        _repr_indent = 4
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def extra_repr() -> str:
        return "load: video keypoints and segments"

    @staticmethod
    def _prepare_for_detector(video, segments):
        video = video['keypoints']
        segments = segments['segments']
        target = {}
        num_seg = len(segments)
        segs = torch.zeros(num_seg, 2)
        label = torch.zeros(num_seg, dtype=torch.long)
        for i, s in enumerate(segments):
            segs[i] = torch.tensor(s['normalised_start_end'])
            label[i] = torch.tensor(1, dtype=torch.long)  # all label signify that there is a gesture
        target['segments'] = segs
        target['labels'] = label
        return video, target


# tool functions
def draw_text(img, text, color):
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    # set the rectangle background to white
    rectangle_bgr = color
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


def build(video_set, args):
    root = Path(args.chalearn_path)
    assert root.exists(), f'provided chalearn path {root} does not exist'
    PATHS = {
        "train": "train_annotations.json",
        "val": "valid_annotations.json",
    }

    ann_file = PATHS[video_set]
    dataset = ChalearnLoader(root, ann_file=ann_file)
    return dataset


# if __name__ == '__main__':
#     root = '/home/yxz2569/chalearn'
#     chalearn = ChalearnKeypointDataset('chalearn/Annotations/valid_annotations.json')
#     chalearn.generate_label_on_frames_one_video('001/00001', 'test')
#     loader = ChalearnLoader(root, ann_file='valid_annotations.json')
#