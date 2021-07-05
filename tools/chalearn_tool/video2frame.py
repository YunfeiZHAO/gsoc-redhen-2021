import cv2
import os


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

def simple_video2frame(video_path, frame_path, silence = True):
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


def frames_dataset_generation(dataset_txt_path):
	pass

def keypoints_dataset_generation(dataset_txt_path):
	pass

def main():
	simple_video2frame('chalearn/train/001/00001.M.avi', 'test')

if __name__ == '__main__':
	main()



