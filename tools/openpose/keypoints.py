import cv2
import os
import glob
import pdb 

from openpose import pyopenpose as op


def extract_keypoints(image_path, opWrapper, save_path):
    imageToProcess = cv2.imread(image_path)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

    # Extract first 8 points on body
    datum.poseKeypoints = datum.poseKeypoints[:, 0:9, :]

    # display
    if save_path:
        cv2.imwrite(save_path, datum.cvOutputData)
    pdb.set_trace()


def extract_keypoints_frames(image_folder, opWrapper, save_folder=False):
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
    for image_path in glob.glob(image_folder + "*.jpg"):
        if save_folder:
            save_path = os.path.join(save_folder, image_path.split('/')[-1])
        extract_keypoints(image_path, opWrapper, save_path)


# OpenPose initialisqtion
params = {}
params['model_folder'] = '/opt/openpose/models'
# params['hand'] = True
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



# 
# extract_keypoints('/home/yxz2569/chalearn/frames/train/001/00002/16.jpg', opWrapper, '/home/yxz2569/chalearn/keypoints/train/001/00002/16.jpg')
extract_keypoints_frames('/home/yxz2569/chalearn/frames/train/003/00103/', opWrapper, '/home/yxz2569/chalearn/keypoints/train/003/00103/')