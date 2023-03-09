import numpy as np
from tracker import Tracker
from sklearn.utils.linear_assignment_ import linear_assignment
from typing import List, Callable, Tuple
from collections import deque
import cv2

def process_unmatched_detections(z_box: List[np.ndarray], unmatched_dets: np.ndarray, 
                                 x_box: List[int], tracker_list: List[Callable], track_id_list: deque, 
                                 labels: List[int], scores: np.ndarray) -> None:
    
    for idx in unmatched_dets:
        
        z = z_box[idx]
        label = labels[idx]
        score = scores[idx]
        
        z = np.expand_dims(z, axis=0).T
        tmp_trk = Tracker() # Create a new tracker
        
        x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]], dtype=object).T
        tmp_trk.x_state = x
        tmp_trk.predict_only()
        
        x_state = tmp_trk.x_state
        x_state = x_state.T[0].tolist()
        x_state = [x_state[0], x_state[2], x_state[4], x_state[6]]
        tmp_trk.box = x_state
        tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
        tracker_list.append(tmp_trk)
        x_box.append(x_state)
        tmp_trk.label = label
        tmp_trk.score = score
        

def process_matched_detections(matched: np.ndarray, z_box: List, x_box: List[List[int]], 
                               tracker_list: List[Callable], labels: np.ndarray, scores: np.ndarray) -> None:

    for trk_idx, det_idx in matched:
        
        z = z_box[det_idx]
        label = labels[det_idx]
        score = scores[det_idx]
        
        z = np.expand_dims(z, axis=0).T
        
        tmp_trk = tracker_list[trk_idx]
        
        tmp_trk.kalman_filter(z)
        
        x_state = tmp_trk.x_state.T[0].tolist()
        x_state = [x_state[0], x_state[2], x_state[4], x_state[6]]
        x_box[trk_idx] = x_state
        
        tmp_trk.box = x_state
        tmp_trk.hits += 1
        tmp_trk.no_losses = 0
        tmp_trk.label = label
        tmp_trk.score = score


def process_unmatched_trackers(unmatched_trks: np.ndarray, x_box: List[List[int]], tracker_list: List[Callable]) -> None:
    
    for trk_idx in unmatched_trks:
        
        tmp_trk = tracker_list[trk_idx]
        tmp_trk.no_losses += 1
        tmp_trk.predict_only()
        x_state = tmp_trk.x_state
        x_state = x_state.T[0]
        x_state = [x_state[0], x_state[2], x_state[4], x_state[6]]
        tmp_trk.box = x_state
        x_box[trk_idx] = x_state

def calc_iou(bboxes1: np.ndarray, bboxes2: np.ndarray) -> float:
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

def assign_detections_to_trackers(trackers: List[Callable], detections: List[np.ndarray], iou_thr: float=0.3) -> Tuple:

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    
    if len(trackers) and len(detections):
        IOU_mat = calc_iou(np.array(trackers), np.array(detections))
    
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thr to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thr):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers) 

