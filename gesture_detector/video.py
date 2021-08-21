import cv2
import os
import shutil


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




def main():
    root = '/home/yxz2569/'
    # simple_video2frame(os.path.join(root, 'ellen_show/video/2014-11-18_0000_US_KNBC_The_Ellen_DeGeneres_Show_950-1240.mp4'),
    #                    os.path.join(root, 'ellen_show/frames/origin'))
    select_frames(os.path.join(root, 'ellen_show/frames/origin'), os.path.join(root, 'ellen_show/frames/selected'), 1883, 2280)


if __name__ == '__main__':
    main()