import numpy as np
from typing import List, Tuple
import cv2


def calc_iou(bboxes1: np.ndarray, bboxes2: np.ndarray) -> float:
    """Calculates the intersection of union of given two bounding box arrays in a vectorized way.

    Args:
        bboxes1 (np.ndarray): The first array of bounding boxes.
        bboxes2 (np.ndarray): The second array of bounding boxes.

    Returns:
        float: The intersection over union.
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    if not (boxAArea + np.transpose(boxBArea) - interArea).all():
        return np.zeros((len(bboxes1),len(bboxes2)),dtype=np.float32)
    
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    
    return iou

def draw_boxes(img: np.ndarray, box: List[int], thickness: int, class_names: Tuple[str], label: int, 
               score: float, colors: List[Tuple[int]], font: int=1) -> np.ndarray:
    """Draws bounding boxes on the image.

    Args:
        img (np.ndarray): The image.
        box (List[int]): The bounding box of a detected object in the image.
        thickness (int): Thickness of the bounding box rectangle.
        class_names (Tuple[str]): Class names of the dataset.
        label (int): The label of the detected object.
        score (float): The confidence score of the detected object.
        colors (List[Tuple[int]]): Colors of each class in the dataset.
        font (int, optional): Font of class names which will print on the image. Defaults to 1.

    Returns:
        np.ndarray: The image with bounding boxes.
    """
    color = colors[label]
    pos = np.array([box[1], box[0]]) - thickness


    label_text =f'{class_names[label]}'
    
    label_text += f'| {score:.02f}'
    
    pt1 = (int(box[1]), int(box[0]))
    pt2 = (int(box[3]), int(box[2]))
    cv2.rectangle(img,pt1,pt2,color,2)
    text_size, _ = cv2.getTextSize(label_text, font, 0.8, 1)

    text_w, text_h = text_size
    text_w += 2
    text_h += 2

    x, y = pos

    cv2.putText(img, label_text, (x, y + text_h ), font, 
            0.8, color, 1, cv2.LINE_AA)
    
    return img