from mmdet2trt.apis import create_wrap_detector
from mmdet.apis import inference_detector
import numpy as np
from typing import Tuple, List

class Detector:
    
    def __init__(self, checkpoint_path: str='/workspaces/detection_and_tracking/yolox_xl_epoch_329_32-2map_trt.pth',
                 model_config_path: str='/workspaces/detection_and_tracking/yolox_x_8x8_300e_coco.py',
                 device: str='cuda:0',
                 score_thr: float=0.4,
                 class_names: List[str]=None) -> None:
        """The class in whic the object detection phase occurs. The TensorRT module runs in this class.

        Args:
            checkpoint_path (str, optional): The checkpoint path of TensorRT module. Defaults to '/workspaces/detection_and_tracking/yolox_xl_epoch_329_32-2map_trt.pth'.
            model_config_path (str, optional): The config path of the model. Defaults to '/workspaces/detection_and_tracking/yolox_x_8x8_300e_coco.py'.
            device (_type_, optional): The device where object detection occurs. Defaults to 'cuda:0'.
            score_thr (float, optional): The confidence threshold of the TensorRT module. Defaults to 0.4.
            class_names (List[str]): The class names of the dataset.
        """
        
        self.detector = create_wrap_detector(checkpoint_path, model_config_path, device)
        if class_names is not None:
            self.detector.CLASSES = class_names
        self.score_thr = score_thr

    def detect_image(self, image: np.ndarray) -> Tuple:
        """The function which detects the objects in an image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            Tuple: Bounding boxes, labels and confidence scores.
        """
        result = inference_detector(self.detector, image)
        
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
                
        else:
            bbox_result, segm_result = result, None
            
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        
        labels = np.concatenate(labels)
        
        if self.score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            

        bboxes = bboxes[:,:4]
        new_boxes = np.empty_like(bboxes)
        new_boxes[:, 0], new_boxes[:, 1], new_boxes[:, 2], new_boxes[:, 3] = bboxes[:, 1], bboxes[:, 0], bboxes[:, 3], bboxes[:, 2]

        z_box = list(new_boxes)
        
        return z_box, labels, scores