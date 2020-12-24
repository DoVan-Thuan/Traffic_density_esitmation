import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val)  # 0-1
    new_img *= 255
    return new_img


def my_canny(img, min_val, max_val, sobel_size=3, is_L2_gradient=False):
    """
    Try to implement Canny algorithm in OpenCV tutorial @ https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
    """

    # 2. Noise Reduction
    smooth_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)

    # 3. Finding Intensity Gradient of the Image
    Gx = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    Gy = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    if is_L2_gradient:
        edge_gradient = np.sqrt(Gx * Gx + Gy * Gy)
    else:
        edge_gradient = np.abs(Gx) + np.abs(Gy)

    angle = np.arctan2(Gy, Gx) * 180 / np.pi

    # round angle to 4 directions
    angle = np.abs(angle)
    angle[angle <= 22.5] = 0
    angle[angle >= 157.5] = 0
    angle[(angle > 22.5) * (angle < 67.5)] = 45
    angle[(angle >= 67.5) * (angle <= 112.5)] = 90
    angle[(angle > 112.5) * (angle <= 157.5)] = 135

    # 4. Non-maximum Suppression
    keep_mask = np.zeros(smooth_img.shape, np.uint8)
    for y in range(1, edge_gradient.shape[0] - 1):
        for x in range(1, edge_gradient.shape[1] - 1):
            area_grad_intensity = edge_gradient[y - 1:y + 2, x - 1:x + 2]  # 3x3 area
            area_angle = angle[y - 1:y + 2, x - 1:x + 2]  # 3x3 area
            current_angle = area_angle[1, 1]
            current_grad_intensity = area_grad_intensity[1, 1]

            if current_angle == 0:
                if current_grad_intensity > max(area_grad_intensity[1, 0], area_grad_intensity[1, 2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 45:
                if current_grad_intensity > max(area_grad_intensity[2, 0], area_grad_intensity[0, 2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 90:
                if current_grad_intensity > max(area_grad_intensity[0, 1], area_grad_intensity[2, 1]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 135:
                if current_grad_intensity > max(area_grad_intensity[0, 0], area_grad_intensity[2, 2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0

    # 5. Hysteresis Thresholding
    canny_mask = np.zeros(smooth_img.shape, np.uint8)
    canny_mask[(keep_mask > 0) * (edge_gradient > min_val)] = 255

    return scale_to_0_255(canny_mask)

def roi(frame):
    pts = np.array([[790, 716], [485, 218], [837, 235], [1277, 603], [1280, 720]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # polygon = Polygon([(195, 327), (127, 889), (1881, 919), (1873, 333)])
    polygon = Polygon([(790, 716), (485, 218), (837, 235), (1277, 603), (1280, 720)])
    mask = np.zeros((720, 1280, 3), np.uint8)
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255), 2)
    mask2 = cv2.fillPoly(mask.copy(), [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, frame)  # frame and with white region = region of frame (ROI)
    cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
    return ROI

def image_dilation(img):
    #dialte nó làm với ảnh đ trắng thôi hay sao ấy
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def number_edge(img):
    roi_image = roi(img)
    roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    canny = my_canny(roi_image, min_val=100, max_val=200)
    none_zero = cv2.countNonZero(canny)
    return none_zero

def predict_light(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    light = []
    a = int(h / 3)
    b = 2 * a
    light.append(img[0:a, 0:w])
    light.append(img[a:b, 0:w])
    light.append(img[b:h, 0:w])
    # for i in range(0, h, int(h / 3)):
    #     light.append(img[i:i + int(h / 3), 0:w])
    # cal_his(light)
    max_value = -99999999
    index = 0
    for i in range(0, len(light)):
        if (np.mean(light[i]) > max_value):
            max_value = np.mean(light[0])
            index = i
    return index

def distance_between(a, b):
    return int(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    a = int(max([distance_between(top_left, bottom_left), distance_between(bottom_right, top_right)]))
    b = int(max([distance_between(top_left, top_right), distance_between(bottom_right, bottom_left)]))
    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)
    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (b, a))