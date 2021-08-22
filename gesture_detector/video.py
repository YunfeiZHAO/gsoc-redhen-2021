import cv2
import os
import shutil
import glob
import json
import numpy as np


def get_video_property(video_path):
    """Get properties from a video
        :return: frame number, fps, width, height
    """
    videocap = cv2.VideoCapture(video_path)
    frame_count = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = videocap.get(cv2.CAP_PROP_FPS)
    height = videocap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = videocap.get(cv2.CAP_PROP_FRAME_WIDTH)
    return frame_count, fps, width, height


def simple_video2frame(video_path, frame_path, silence=True):
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    while success:
        count += 1
        cv2.imwrite(os.path.join(frame_path, f'{count}.jpg'), frame)
        success, frame = vidcap.read()
        if not silence == True:
            print(f'frame {count}')
    vidcap.release()
    if not silence == True:
        print(f"total Frames:{count}, saved at {frame_path}")


def select_frames(origin_path, destination_path, start, end):
    for i, f in enumerate(range(start, end)):
        shutil.copy(os.path.join(origin_path, str(f)) + '.jpg', os.path.join(destination_path, str(i+1)) + '.jpg')


def simple_frame2video(frame_path, video_path):
    img_array = []
    img = cv2.imread(os.path.join(frame_path, '1.jpg'))
    height, width, layers = img.shape
    size = (width, height)
    for i in range(1, 398):
        filename = os.path.join(frame_path, str(i) + '.jpg')
        img = cv2.imread(filename)
        img_array.append(img)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


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
    cv2.imwrite(os.path.join(save_path, str(frame_number) + '.jpg'), im)


def main():
    root = '/home/yxz2569/'
    # simple_video2frame(os.path.join(root, 'ellen_show/video/2014-11-18_0000_US_KNBC_The_Ellen_DeGeneres_Show_950-1240.mp4'),
    #                    os.path.join(root, 'ellen_show/frames/origin'))
    # select_frames(os.path.join(root, 'ellen_show/frames/origin'), os.path.join(root, 'ellen_show/frames/selected'), 1883, 2280)
    # simple_frame2video(os.path.join(root, 'ellen_show/keypoints/output'), os.path.join(root, 'ellen_show/keypoints/output/result.avi'))

    draw_normalised_keypoints_on_frame(frame_path=os.path.join(root, 'ellen_show/frames/selected'),
                                       keypoint_path=os.path.join(root, 'ellen_show/keypoints/ellen.json'),
                                       frame_number=1,
                                       save_path=os.path.join(root, 'ellen_show/frames/labeled/'))

if __name__ == '__main__':
    main()