import cv2

"""
Task 1: Given as input two matrices of sizes 512x512 representing grayscale images of an object in motion at
moments T - 33.36 milliseconds and T, write a general purpose algorithm that can estimate what the image will look
like at moment T+1ms, T+33.36 ms, T+1 s . Test and provide input-output example.
"""

"""
1 Extract images from a video at different time stamps. 
2 Transfer it to grayscale image with a shape of 215 times 512. 
3 Evaluate optical flow of the images at moments T - 33.36 milliseconds and T 
4 Predict the images at moment T+1ms, T+33.36 ms, T+1 s by supposing that the object is doing a linear movement with 
  constant speed.
"""


# 1 Extract images from a video at different time stamps.
def extract_image(path, time):
    """
    Extract image from a video
    :param path: The file path to the video
    :param time: the given time of the image to be captured in ms
    :return:
    """
    vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time)
    success, image = vidcap.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if success:
        # cv2.imshow(f"{time}sec", gray_image)
        # cv2.waitKey(0)
        # croped_image = gray_image[105:105+512, 384:384+512]
        cv2.imshow("croped gray image", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return gray_image
    else:
        print("unsuccessful image extraction")


def main():
    # extract_image('./long shoot.mkv', 26000)
    pass


if __name__ == '__main__':
    main()