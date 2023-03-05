import matplotlib.pyplot as plt
import cv2
from collections import deque
import helpers
import argparse
from detector import Detector
from process import *
import glob
from tkinter import *
import matplotlib
#plt.style.use('ggplot')
matplotlib.use('TkAgg')

def draw_boxes(img, box, thickness, class_names, label, score, font=1):
    alpha = 0.9
    pos = np.array([box[1], box[0]]) - thickness
    
    #for box, label, pos, score in zip(bboxes, labels, positions, scores):#, ids):
        
    #label_text = str(id)
    label_text =f'{class_names[label]}'
    
    label_text += f'| {score:.02f}'
    #left, top, right, bottom = int(bbox_cv2[1]), int(bbox_cv2[0]), int(bbox_cv2[3]), int(bbox_cv2[2])
    pt1 = (int(box[1]), int(box[0]))
    pt2 = (int(box[3]), int(box[2]))
    cv2.rectangle(img,pt1,pt2,(255,0,0),2)
    overlay = img.copy()
    text_size, _ = cv2.getTextSize(label_text, font, 0.7, 1)
    text_scale = int((int(box[1]) - int(box[0])) / 200)
    text_w, text_h = text_size
    
    x, y = pos
    cv2.rectangle(overlay, pos, (x + text_w, y + text_h), (127,127,127), -1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.putText(img, label_text, (x, y + text_h ), font, 
            0.7, (0,0,255), 1, cv2.LINE_AA)
    return img

def pipeline(img, detector, frame_count, max_age, min_hits, tracker_list, track_id_list):
    '''
    Pipeline function for detection and tracking
    '''
    

    frame_count+=1

    z_box, labels, scores = detector.detect_image(img)
       
    x_box =[]
    
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
    
    
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, labels, scores, iou_thrd = 0.3)      
         
    # Deal with matched detections     
    if matched.size >0:
        process_matched_detections(matched, z_box, x_box, tracker_list, labels, scores)
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        process_unmatched_detections(z_box, unmatched_dets, x_box, tracker_list, track_id_list, labels, scores)          
    
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        process_unmatched_trackers(unmatched_trks, x_box, tracker_list)

    # The list of tracks to be annotated  
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             label = trk.label
             score = trk.score
             img = draw_boxes(img, x_cv2, 2, detector.detector.CLASSES, label, score)
             

             #img= helpers.draw_box_label(img, x_cv2) # Draw the bounding boxes on the 
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]    
       
    return img


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_age', type=int, default=1, help='Number of consecutive unmatched detection before a track is deleted.')
    parser.add_argument('--min_hits', type=int, default=1, help='Number of consecutive matches needed to establish a track.')
    frame_count = 0 # frame counter
    args = parser.parse_args()
    
    tracker_list =[] # list for trackers
    # list for track ID
    track_id_list= deque(list(range(10000)))
    debug = True
    detector = Detector()
    
    if debug: # test on a sequence of images
        #images = [plt.imread(file) for file in sorted(glob.glob('./2/*.jpg'))]
        cap = cv2.VideoCapture('/workspaces/detection_and_tracking/tokyo.mp4')
        while(cap.isOpened()):
            ret, frame = cap.read()
            #image = images[i]
            image_box = pipeline(frame, detector, frame_count, args.max_age, args.min_hits, tracker_list, track_id_list)
            #cv2.imwrite('frame' + str(i) + '.jpg', image_box)
            cv2.imshow('frame', image_box)   
            cv2.waitKey(30)


