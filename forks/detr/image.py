import PIL
import cv2
import torchvision.transforms as T
import numpy as np
import torch
import random


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def piltensor_to_cvnp(pil_image):
#     """change tensor of PIL image to numpy array of oepncv image"""
#     tensor_image = T.ToTensor()(pil_image)  # it will change int8 to value between [0,1] of type float32
#
#     # change axi and recover scale
#     numpy_image = np.moveaxis(tensor_image.numpy()*255, 0, -1)
#
#     # change to opencv numpy BGR and change data type
#     numpy_cv = numpy_image[:, :, ::-1].astype(np.uint8)
#     return numpy_cv


# pil_image = PIL.Image.open('../000000112342.jpg').convert('RGB')
# """change tensor of PIL image to numpy array of oepncv image"""
# tensor_image = T.ToTensor()(pil_image)  # it will change int8 to value between [0,1] of type float32
#
# # change axi and recover scale
# numpy_image = np.moveaxis(tensor_image.numpy() * 255, 0, -1)
#
# # change to opencv numpy BGR and change data type
# numpy_cv = numpy_image[:, :, ::-1].astype(np.uint8)
# show(numpy_cv)


class_name = {1: 'vehicle fallback', 2: 'rider', 3: 'bus', 4: 'car', 5: 'autorickshaw', 6: 'truck',
              7: 'motorcycle', 8: 'person', 9: 'traffic sign', 10: 'animal', 11: 'bicycle',
              12: 'traffic light', 13: 'caravan', 14: 'train', 15: 'trailer'}


def piltensor_to_cvnp(image):
    """change tensor of PIL image to numpy array of oepncv image"""
    tensor_image = image

    # change axi and recover scale
    numpy_image = np.moveaxis((tensor_image * 255).clamp(0, 255).numpy(), 0, -1)

    # change to opencv numpy BGR and change data type
    numpy_cv = numpy_image[:, :, ::-1].astype(np.uint8)
    return numpy_cv


def test(image, target):
    _, im_h, im_w = image.size()
    labels_num = target['labels']
    # rescale = torch.tensor([[im_w, im_h, im_w, im_h]])
    # bboxs = target['boxes'] * rescale
    img = piltensor_to_cvnp(image)
    bboxs = target['boxes']
    for i, bbox in enumerate(bboxs):
        x, y, w, h = bbox
        label = class_name[int(labels_num[i])]
        # plot_one_box((int(x), int(y), int(xm), int(ym)), img, label=label, line_thickness=3)
        color = [random.randint(0, 255) for _ in range(3)]
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (int(x), int(y))

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (int(x+w), int(y+h))

        # Blue color in BGR
        # color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = np.array(img).copy()
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        # cv2.rectangle(img, (x, y), (xm, ym), color, thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_im_rec(pil_image, target):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    for i in target['annotations']:
        x, y, w, h = i['bbox']
        color = [random.randint(0, 255) for _ in range(3)]
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (int(x), int(y))
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (int(x+w), int(y+h))
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(open_cv_image, start_point, end_point, color, thickness)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

