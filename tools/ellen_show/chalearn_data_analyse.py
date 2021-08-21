""" Chalear Continuous Gesture Recognition dataset anayse https://chalearnlap.cvc.uab.cat/dataset/22/description/
author: Yunfei ZHAO
"""

import os
import re
import json
from tqdm import tqdm

import video2frame


def generate_folders(root):
    frame_path = os.path.join(root, 'frames')
    keypoint_path = os.path.join(root, 'keypoints')
    paths = [frame_path, keypoint_path]
    types = ['train', 'val']
    for p in paths:
        for t in types:
            os.makedirs(os.path.join(p, t), exist_ok=True)


class TemproalSegment:
    """
    Class that contains the information of a single annotated object as a temporal segmentation
    """

    def __init__(self):
        # [start frame number, end frame number]
        self.start_end = []
        # the label of the corresponding object
        self.label = -1

    def __str__(self):
        segment = ""
        segment += '[(start: {}, end: {})]'.format(
            self.start_end[0], self.start_end[1])

        text = "Object: {}\n - temporal segment {}".format(
            self.label, segment)
        return text

    def fromText(self, text):
        """
         try to load from txt string
         format: "1,43:223": label 1, from frame 43 to frame 223
        """
        start, end, label = re.split(',|:', text)
        self.start_end = [int(start), int(end)]
        self.label = int(label)

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['start_end'] = self.start_end
        return objDict


class ChalearnAnnotation:
    """The annotation of a video
        Extract annotation from text file
    """

    # Constructor
    def __init__(self):
        self.id = ''
        # paths for an annotation
        self.video_path = ''
        self.frame_path = ''
        self.keypoint_path = ''
        # video w,h
        self.imgWidth = 0
        self.imgHeight = 0
        # the leghth of video (frame number)
        self.videoLenghth = 0
        self.videoFPS = 0
        # the list of objects
        self.temporal_segments = []

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromText(self, root, type, text):
        elements = text.split(' ')
        self.id = elements[0]
        segments_txt = elements[1:]
        # set paths
        self.video_path = os.path.join(type, self.id + '.M.avi')
        self.frame_path = os.path.join('frames', type, self.id)
        self.keypoint_path = os.path.join('keypoints', type, self.id + '.json')
        # video properties
        frame_count, fps, width, height = video2frame.get_video_property(os.path.join(root, self.video_path))
        self.videoLenghth = int(frame_count)
        self.videoFPS = fps
        self.imgWidth = width
        self.imgHeight = height
        # load segmentations
        for segment_txt in segments_txt:
            TS = TemproalSegment()
            TS.fromText(segment_txt)
            self.temporal_segments.append(TS)

    def toJsonText(self):
        jsonDict = {}
        jsonDict['id'] = self.id
        jsonDict['video_path'] = self.video_path
        jsonDict['frame_path'] = self.frame_path
        jsonDict['keypoint_path'] = self.keypoint_path
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['videoLenghth'] = self.videoLenghth
        jsonDict['videoFPS'] = self.videoFPS
        jsonDict['temporal_segments'] = []
        for temporal_segment in self.temporal_segments:
            objDict = temporal_segment.toJsonText()
            jsonDict['temporal_segments'].append(objDict)
        return jsonDict


def chalearn_json_annotation_generation(root='chalearn', type='train', jsonfile_root='annotations'):
    """Generate annotations"""
    annotation_path = os.path.join(root, jsonfile_root, type + '_annotations.json')
    txt_file_path = os.path.join(root, f'{type}.txt')
    os.makedirs(os.path.join(root, jsonfile_root), exist_ok=True)
    videos = []
    segments = []
    id = 1
    with open(txt_file_path) as fp:
        for line in tqdm(fp):
            ca = ChalearnAnnotation()
            ca.fromText(root, type, line)
            ca_json = ca.toJsonText()
            s = ca_json['temporal_segments']
            del ca_json['temporal_segments']
            length = ca.videoLenghth
            videos.append(ca_json)
            for segment in s:
                start, end = segment['start_end']
                segment['id'] = id
                segment['video_id'] = ca.id
                segment['normalised_start_end'] = [start/length, end/length]
                id += 1
                segments.append(segment)
    dataset = {}
    dataset['videos'] = videos
    dataset['segments'] = segments
    with open(annotation_path, 'w') as outfile:
        json.dump(dataset, outfile)
    return


def chalearn_frames_generation(root='chalearn', type='train', frames_root='frames'):
    """Generate frames from videos"""
    annotation_path = os.path.join(root, 'annotations', type + '_annotations.json')
    data = json.load(open(annotation_path))
    videos = data['videos']
    for video in tqdm(videos):
        id = video['id']
        frame_path = os.path.join(root, frames_root, type, id)
        os.makedirs(frame_path, exist_ok=True)
        video_path = video['video_path']
        video2frame.simple_video2frame(video_path, frame_path)





    
# def main():
#     root = '/home/yxz2569/chalearn'
#     generate_folders(root)

# if __name__ == "__main__":
#     main()	

# root = '/home/yxz2569/chalearn'
# type = 'train'
# CA = ChalearnAnnotation()
# CA.fromText(root, type, '110/05452 1,33:233 34,71:71')

# Annotation generation
# chalearn_json_annotation_generation('/home/yxz2569/chalearn/', 'train', 'annotations')
# chalearn_json_annotation_generation('/home/yxz2569/chalearn/', 'valid', 'annotations')

# Frames generation
# chalearn_frames_generation(root='/home/yxz2569/chalearn/', type='train', frames_root='frames')
# chalearn_frames_generation(root='/home/yxz2569/chalearn/', type='valid', frames_root='frames')

