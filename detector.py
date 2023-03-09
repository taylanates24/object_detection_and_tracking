from mmdet2trt.apis import create_wrap_detector
from mmdet.apis import inference_detector
import numpy as np


class Detector:
    
    
    def __init__(self, checkpoint_path='/workspaces/detection_and_tracking/yolox_xl_epoch_329_32-2map_trt.pth',
                 model_config_path='/workspaces/detection_and_tracking/yolox_x_8x8_300e_coco.py',
                 device='cuda:0',
                 score_thr= 0.4) -> None:
        
        self.detector = create_wrap_detector(checkpoint_path, model_config_path, device)
        self.score_thr = score_thr

    def detect_image(self, image):

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
            
        #img = mmcv.bgr2rgb(img)
        bboxes = bboxes[:,:4]
        new_boxes = np.empty_like(bboxes)
        new_boxes[:, 0], new_boxes[:, 1], new_boxes[:, 2], new_boxes[:, 3] = bboxes[:, 1], bboxes[:, 0], bboxes[:, 3], bboxes[:, 2]
        #new_boxes = new_boxes.astype('uint8')
        z_box = list(new_boxes)
        
        return z_box, labels, scores