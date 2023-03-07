import matplotlib.pyplot as plt
import cv2
from collections import deque
import argparse
from detector import Detector
from process import *
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')

def draw_boxes(img, box, thickness, class_names, label, score, colors, font=1, eps=0.01, font_size=13):
    #alpha = 0.7

    color = colors[label]
    pos = np.array([box[1], box[0]]) - thickness

    #label_text = str(id)
    label_text =f'{class_names[label]}'
    
    label_text += f'| {score:.02f}'
    #left, top, right, bottom = int(bbox_cv2[1]), int(bbox_cv2[0]), int(bbox_cv2[3]), int(bbox_cv2[2])
    pt1 = (int(box[1]), int(box[0]))
    pt2 = (int(box[3]), int(box[2]))
    cv2.rectangle(img,pt1,pt2,color,2)
    #overlay = img.copy()        #(241,101,72)
    text_size, _ = cv2.getTextSize(label_text, font, 0.8, 1)
    #text_scale = int((int(box[1]) - int(box[0])) / 200)
    text_w, text_h = text_size
    text_w += 2
    text_h += 2

    x, y = pos
    #cv2.rectangle(overlay, pos, (x + text_w, y + text_h), (0,0,0), -1)
    #img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.putText(img, label_text, (x, y + text_h ), font, 
            0.8, color, 1, cv2.LINE_AA)
    return img

def pipeline(img, detector, frame_count, max_age, min_hits, tracker_list, track_id_list, colors, num_skip_frame):
    '''
    Pipeline function for detection and tracking
    '''

    if frame_count % (num_skip_frame+1) == 0:
        z_box, labels, scores = detector.detect_image(img)
    else:
        z_box, labels, scores = [], [], []

    x_box =[]

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, labels, scores, iou_thrd = 0.3)      

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
    parser.add_argument('--max_age', type=int, default=2, help='Number of consecutive unmatched detection before a track is deleted.')
    parser.add_argument('--checkpoint', type=str, default='/workspaces/detection_and_tracking/yolox_xl_epoch_329_32-2map_trt.pth', 
                        help='The TensorRT checkpoint path.')
    parser.add_argument('--config', type=str, default='/workspaces/detection_and_tracking/yolox_x_8x8_300e_coco.py', 
                        help='The config path of mmdetection model.')
    parser.add_argument('--score_thr', type=int, default=0.4, help='Score threshold.')
    parser.add_argument('--min_hits', type=int, default=1, help='Number of consecutive matches needed to establish a track.')
    parser.add_argument('--input_video', type=str, default='/workspaces/detection_and_tracking/tokyo_1.mp4', 
                        help='The path of input video.')
    parser.add_argument('--skip_frame', type=int, default=0, help='Number of frames that only tracked, not detected by model.')
    parser.add_argument('--save_video', type=bool, default=False, help='Saving output video.')
    parser.add_argument('--save_video_path', type=str, default='out.mp4', help='Output video path.')
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
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, size)
    
    while(cap.isOpened()):
        ret, frame = cap.read()

        image_box = pipeline(frame, detector, frame_count, args.max_age, args.min_hits, tracker_list, track_id_list, colors, args.skip_frame)
        frame_count += 1
        
        if args.save_video:
            result.write(image_box)
            
        cv2.imshow('frame', image_box)   
        cv2.waitKey(20)
        
    cap.release()
    
    if args.save_vide:
        result.release()

