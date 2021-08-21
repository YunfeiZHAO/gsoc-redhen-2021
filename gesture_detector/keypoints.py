"""Keypoints generation by Openpose"""
import cv2
import os
import glob
from tqdm import tqdm

import numpy as np
import json

from openpose import pyopenpose as op
import video


def extract_keypoints(image_path, opWrapper, save_path):
    imageToProcess = cv2.imread(image_path)
    height, width, _ = imageToProcess.shape
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

    # Extract first 8 points on body
    if len(datum.poseKeypoints.shape) != 3:
        poseKeypoints = np.zeros((1, 9, 2))
        normalised_center_point_1 = poseKeypoints[0][1]
        centered_normalised_poseKeypoints = poseKeypoints[0]
        print(normalised_center_point_1)
        print(centered_normalised_poseKeypoints)
    else:
        poseKeypoints = datum.poseKeypoints[0, 0:9, 0:2]
        normalised_poseKeypoints = poseKeypoints/[width, height]
        normalised_center_point_1 = normalised_poseKeypoints[1]
        centered_normalised_poseKeypoints = normalised_poseKeypoints - normalised_center_point_1
    # display
    if save_path:
        cv2.imwrite(save_path, datum.cvOutputData)
    return normalised_center_point_1, centered_normalised_poseKeypoints.reshape(-1)


def extract_keypoints_frames(video_path, image_folder, opWrapper, save_json_path, save_keypoint_image_folder=False, generate_frames=False):
    # get video properties
    frame_count, fps, image_width, image_height = video.get_video_property(video_path)

    # create folders
    original_image_folder = os.path.join(image_folder, 'origin')
    selected_image_folder = os.path.join(image_folder, 'selected')
    os.makedirs(original_image_folder, exist_ok=True)
    os.makedirs(selected_image_folder, exist_ok=True)
    os.makedirs(save_json_path, exist_ok=True)
    os.makedirs(save_keypoint_image_folder, exist_ok=True)

    # genenerate video frames
    if generate_frames:
        video.simple_video2frame(video_path, original_image_folder)
        video.select_frames(original_image_folder, selected_image_folder, 1883, 2280)

    # start keypoints processing
    frames_path = glob.glob(os.path.join(selected_image_folder + "/*.jpg"))
    nframe = len(frames_path)
    centers = np.zeros((nframe, 2))
    keypoints = np.zeros((nframe, 18))
    for i, image_path in enumerate(frames_path):
        if save_keypoint_image_folder is not False:
            save_keypoint_image_path = os.path.join(save_keypoint_image_folder, str(i + 1) + '.jpg')
        else:
            save_keypoint_image_path = False
        normalised_center_point_1, centered_normalised_poseKeypoints = extract_keypoints(image_path, opWrapper, save_keypoint_image_path)
        centers[i] = normalised_center_point_1
        keypoints[i] = centered_normalised_poseKeypoints

    # save to json
    keypoints_data = {
        'imgWidth': image_width,
        'imgHeight': image_height,
        'centers': centers.tolist(),
        'keypoints': keypoints.tolist(),
    }
    with open(os.path.join(save_json_path, 'ellen.json'), 'w') as outfile:
        json.dump(keypoints_data, outfile)
    return keypoints_data


# OpenPose initialisation
params = {}
params['model_folder'] = '/opt/openpose/models'
# params['hand'] = True
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

root = '/home/yxz2569/'


def main():
    extract_keypoints_frames(video_path=os.path.join(root, 'ellen_show/video/2014-11-18_0000_US_KNBC_The_Ellen_DeGeneres_Show_950-1240.mp4'),
                             image_folder=os.path.join(root, 'ellen_show/frames'),
                             opWrapper=opWrapper,
                             save_json_path=os.path.join(root, 'ellen_show/keypoints'),
                             save_keypoint_image_folder=os.path.join(root, 'ellen_show/keypoints/frames'),
                             generate_frames=False
                             )


if __name__ == '__main__':
    main()
