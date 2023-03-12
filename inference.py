import cv2
from collections import deque
import argparse
from detector import Detector
from process import *
from tkinter import *
import time
import numpy as np
from typing import List, Tuple
from utils import *
import yaml

def process_frame(img: np.ndarray, detector: Callable, frame_count: int, max_age: int, min_hits: int, tracker_list: List, 
                  track_id_list: deque, colors: List[Tuple[int]], num_skip_frame: int, iou_thr: float=0.3) -> np.ndarray:
    """The function which processes the frame, detects the objects and tracks them.

    Args:
        img (np.ndarray): Input image.
        detector (Callable): TensorRT detection module.
        frame_count (int): The current frame number.
        max_age (int): A tracker will be deleted if there is no detection until the max_age.
        min_hits (int): If a detection show up min_hits time, a tracker will be assignet to that detection.
        tracker_list (List): The list of tracker objects.
        track_id_list (deque): The list of tracker IDs.
        colors (List[Tuple[int]]): The colors of each classes in the dataset.
        num_skip_frame (int): Number of frames which passes the detection, only goes to tracker.
        iou_thr (float, optional): The threshold in which SORT algorithm uses to assign two object in two consecutive 
                frames. Defaults to 0.3.

    Returns:
        np.ndarray: The image with tracker bounding boxes.
    """

    if frame_count % (num_skip_frame+1) == 0:
        z_box, labels, scores = detector.detect_image(img)
        
    else:
        z_box, labels, scores = [], [], []

    x_box =[]

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thr = iou_thr)      

    if matched.size > 0:
        process_matched_detections(matched, z_box, x_box, tracker_list, labels, scores)
        
    if len(unmatched_dets) > 0:
        process_unmatched_detections(z_box, unmatched_dets, x_box, tracker_list, track_id_list, labels, scores)          
    
    if len(unmatched_trks) > 0:
        process_unmatched_trackers(unmatched_trks, x_box, tracker_list)

    for trk in tracker_list:
        
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):

             x_cv2 = trk.box
             label = trk.label
             score = trk.score
             img = draw_boxes(img, x_cv2, 2, detector.detector.CLASSES, label, score, colors)

    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)  

    for trk in deleted_tracks:
        
            track_id_list.append(trk.id)
            if len(track_id_list) > 50:
                del track_id_list[0]

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]    

    return img

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_cfg', type=str, default='inference.yaml', help='inference config file')

    args = parser.parse_args()
    
    opt = args.infer_cfg
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)
        
    frame_count = 0
    tracker_list = []

    track_id_list = deque(list(range(30)))

    colors = [(255, 255, 0), (0, 255, 255), (241,101,72), (128, 128, 0), (128, 0, 128), (0, 0, 255), (128, 0, 128), (128, 0, 0),
              (128, 0, 128), (255, 0, 255)]
   
    class_names = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign')
    
    detector = Detector(checkpoint_path=opt['detector']['checkpoint_path'], 
                        model_config_path=opt['detector']['config'],
                        score_thr=opt['detector']['score_thr'],
                        class_names=class_names)
    
    save_video = opt['save_video']
    save_video_path = opt['save_video_path']
    cap = cv2.VideoCapture(opt['input_video'])
    
    if save_video:
   
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(save_video_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30, size)
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        st = time.time()
        image = process_frame(img=frame, detector=detector, 
                                  frame_count=frame_count, max_age=opt['tracker']['max_age'], 
                                  min_hits=opt['tracker']['min_hits'], tracker_list=tracker_list, 
                                  track_id_list=track_id_list, colors=colors, 
                                  num_skip_frame=opt['tracker']['skip_frame'], iou_thr=opt['tracker']['iou_thr'])
        end = time.time()
        print('process time: ', 1000*(end-st))
        frame_count += 1
        
        if save_video:
            result.write(image)

    cap.release()
    
    if save_video:
        result.release()

