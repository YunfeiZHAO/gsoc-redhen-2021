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


# Preprocessing for step 1 and 2
def extract_image(path, time, x, y):
    """
    Extract image from a video
    :param path: The file path to the video
    :param time: the given time of the image to be captured in ms
    :return: The cropped image in grayscale
    """
    vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time)
    success, image = vidcap.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if success:
        cropped_image = gray_image[x:x+512, y:y+512]
        cv2.imshow("croped gray image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return cropped_image
    else:
        print("unsuccessful image extraction")


def extract_four_images(path, time=253500, x=0, y=250):
    """
    Extract fours images of T-33.36 ms, T+1ms, T+33.36 ms, T+1s, from a video
    :param path: The file path to the video
    :param time: the given time of the image to be captured in ms
    :return: The cropped image in grayscale
    """
    vidcap = cv2.VideoCapture(path)
    for t in [-33.36, 0, 1, 33.36, 1000]:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, time + t)
        success, image = vidcap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if success:
            cropped_image = gray_image[x:x+512, y:y+512]
            cv2.imwrite(f'./{t}.png', cropped_image)
        else:
            print("unsuccessful image extraction")


# optical flow calculated from the images at moments T - 33.36 milliseconds and T
'''
Hypothesis
We have two images takenith a little time difference
1 The intensity of points are constant on both images 
2 There is only a little displacement of object between images
'''
def gradient


def main():
    # extract_four_images('3 Point contest 2015.mp4')

    pass


if __name__ == '__main__':
    main()