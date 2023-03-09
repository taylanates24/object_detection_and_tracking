import matplotlib.pyplot as plt
import cv2
from collections import deque
import argparse
from detector import Detector
from process import *
from tkinter import *
import time
import numpy as np
from typing import List, Tuple


def process_frame(img: np.ndarray, detector: Callable, frame_count: int, max_age: int, min_hits: int, 
                  tracker_list: List, track_id_list: deque, colors: List[Tuple[int]], num_skip_frame: int):


    if frame_count % (num_skip_frame+1) == 0:
        z_box, labels, scores = detector.detect_image(img)
    else:
        z_box, labels, scores = [], [], []

    x_box =[]

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)      

    if matched.size >0:
        process_matched_detections(matched, z_box, x_box, tracker_list, labels, scores)
        
    if len(unmatched_dets)>0:
        process_unmatched_detections(z_box, unmatched_dets, x_box, tracker_list, track_id_list, labels, scores)          
    
    if len(unmatched_trks)>0:
        process_unmatched_trackers(unmatched_trks, x_box, tracker_list)

    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):

             x_cv2 = trk.box
             label = trk.label
             score = trk.score
             img = draw_boxes(img, x_cv2, 2, detector.detector.CLASSES, label, score, colors)

    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  

    for trk in deleted_tracks:
            track_id_list.append(trk.id)
            if len(track_id_list) > 50:
                del track_id_list[0]

    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]    

    return img

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_age', type=int, default=3, help='Number of consecutive unmatched detection before a track is deleted.')
    parser.add_argument('--checkpoint', type=str, default='/workspaces/detection_and_tracking/yolox-xl-best.pth', 
                        help='The TensorRT checkpoint path.')
    parser.add_argument('--config', type=str, default='/workspaces/detection_and_tracking/yolox_x_8x8_300e_coco.py', 
                        help='The config path of mmdetection model.')
    parser.add_argument('--score_thr', type=int, default=0.4, help='Score threshold.')
    parser.add_argument('--min_hits', type=int, default=2, help='Number of consecutive matches needed to establish a track.')
    parser.add_argument('--input_video', type=str, default='/workspaces/detection_and_tracking/tokyo_1.mp4', 
                        help='The path of input video.')
    parser.add_argument('--skip_frame', type=int, default=0, help='Number of frames that only tracked, not detected by model.')
    parser.add_argument('--save_video', type=bool, default=True, help='Saving output video.')
    parser.add_argument('--save_video_path', type=str, default='out_best2.mp4', help='Output video path.')
    args = parser.parse_args()
    frame_count = 0
    tracker_list = []

    track_id_list = deque(list(range(30)))

    detector = Detector(checkpoint_path=args.checkpoint, 
                        model_config_path=args.config,
                        score_thr=args.score_thr)
    
    colors = [(255, 255, 0), (0, 255, 255), (241,101,72), (128, 128, 0), (128, 0, 128), (0, 0, 255), (128, 0, 128), (128, 0, 0),
              (128, 0, 128), (255, 0, 255)]

    cap = cv2.VideoCapture(args.input_video)
    
    if args.save_video:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(args.save_video_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30, size)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        st = time.time()
        image_box = process_frame(frame, detector, frame_count, args.max_age, args.min_hits, tracker_list, track_id_list, colors, args.skip_frame)
        end = time.time()
        print('process time: ', 1000*(end-st))
        frame_count += 1
        
        if args.save_video:
            result.write(image_box)
        #cv2.imshow('frame', image_box)   
        #cv2.waitKey(20)
        
    cap.release()
    
    if args.save_video:
        result.release()

