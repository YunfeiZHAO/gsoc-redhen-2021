import cv2
import os
import glob
from tqdm import tqdm

import numpy as np
import json



from openpose import pyopenpose as op


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


def extract_keypoints_frames(image_folder, image_width, image_height, opWrapper, save_json_path, save_folder=False):
    os.makedirs(save_json_path.split('.')[0], exist_ok=True)
    frames_path = glob.glob(os.path.join(image_folder + "/*.jpg"))
    nframe = len(frames_path)
    centers = np.zeros((nframe, 2))
    keypoints = np.zeros((nframe, 18))
    for i, image_path in enumerate(frames_path):
        if save_folder:
            save_path = os.path.join(save_folder, image_path.split('/')[-1])
        else: save_path = False
        normalised_center_point_1, centered_normalised_poseKeypoints = extract_keypoints(image_path, opWrapper, save_path)
        centers[i] = normalised_center_point_1
        keypoints[i] = centered_normalised_poseKeypoints
    keypoints_data = {
        'imgWidth': image_width,
        'imgHeight': image_height,
        'centers': centers.tolist(),
        'keypoints': keypoints.tolist(),
    }
    with open(save_json_path, 'w') as outfile:
        json.dump(keypoints_data, outfile)
    return keypoints_data


def draw_normalised_keypoints_on_frame(frame_path, keypoint_path, frame_number, save_path):
    image_path = os.path.join(frame_path, str(frame_number) + '.jpg')
    im = cv2.imread(image_path)
    keypoint_data = json.load(open(keypoint_path))
    width = keypoint_data['imgWidth']
    height = keypoint_data['imgHeight']
    center = np.array(keypoint_data['centers'][frame_number - 1])
    keypoints = np.array(keypoint_data['keypoints'][frame_number - 1])
    keypoints = keypoints.reshape((-1,2))
    keypoints_pos = keypoints + center
    keypoints_pos = keypoints_pos * np.array([width, height])
    keypoints_pos = keypoints_pos.astype(int)
    for k in keypoints_pos:
        cv2.circle(im, k, radius=2, color=(255, 0, 0), thickness=-1)
    cv2.imwrite(save_path, im)
    

def generate_keypoints_dataset(root='chalearn', type='train', keypoints_root='keypoints'):
    """Generate keypoints from frames"""
    annotation_path = os.path.join(root, 'Annotations', type + '_annotations.json')
    data = json.load(open(annotation_path))
    videos = data['videos']
    for video in tqdm(videos):
        id = video['id']
        keypoints_path = os.path.join(root, keypoints_root, type, id + '.json')
        if not os.path.isfile(keypoints_path):
            print(video['frame_path'])
            extract_keypoints_frames(video['frame_path'], video['imgWidth'], video['imgHeight'], opWrapper, keypoints_path)
        # extract_keypoints_frames('/home/yxz2569/chalearn/frames/train/003/00103/', 320.0, 240.0, opWrapper, '/home/yxz2569/chalearn/keypoints/train/003/00103.json')

# OpenPose initialisqtion
params = {}
params['model_folder'] = '/opt/openpose/models'
# params['hand'] = True
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

root = '/home/yxz2569/chalearn'
# extract_keypoints('/home/yxz2569/chalearn/frames/train/001/00002/16.jpg', opWrapper, '/home/yxz2569/chalearn/keypoints/train/001/00002/16.jpg')
# extract_keypoints_frames('/home/yxz2569/chalearn/frames/train/003/00103/', 320.0, 240.0, opWrapper, '/home/yxz2569/chalearn/keypoints/train/003/00103.json', '/home/yxz2569/chalearn/keypoints/train/003/00103/')
# extract_keypoints_frames('/home/yxz2569/chalearn/frames/train/003/00103/', 320.0, 240.0, opWrapper, '/home/yxz2569/chalearn/keypoints/train/003/00103.json')
# draw_normalised_keypoints_on_frame('/home/yxz2569/chalearn/frames/train/003/00103/', '/home/yxz2569/chalearn/keypoints/train/003/00103.json', 1, 'test1.jpg')
generate_keypoints_dataset(root, type='train', keypoints_root='keypoints')