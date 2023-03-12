import numpy as np
from tracker import Tracker
from sklearn.utils.linear_assignment_ import linear_assignment
from typing import List, Callable, Tuple
from collections import deque
from utils import calc_iou

def process_unmatched_detections(z_box: List[np.ndarray], unmatched_dets: np.ndarray, 
                                 x_box: List[int], tracker_list: List[Callable], track_id_list: deque, 
                                 labels: List[int], scores: np.ndarray) -> None:
    """Creates new tracker for new detections and updates the trackers.

    Args:
        z_box (List[np.ndarray]): Detection bounding boxes.
        unmatched_dets (np.ndarray): The array of unmatched (new) detections.
        x_box (List[int]): The bounding boxes coming from trackers.
        tracker_list (List[Callable]): The list of tracker objects.
        track_id_list (deque): The list of tracker IDs.
        labels (List[int]): The labels of detected object in the image.
        scores (np.ndarray): The confidence scores of detected object in the image.
    """
    
    for idx in unmatched_dets:
        
        z = z_box[idx]
        label = labels[idx]
        score = scores[idx]
        
        z = np.expand_dims(z, axis=0).T
        tmp_trk = Tracker() 
        
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
    """Kalman Filter's prediction and update functions take place in this function. Each tracker comes with a detection.

    Args:
        matched (np.ndarray): The array of matched trackers and detections.
        z_box (List): Detection bounding boxes.
        x_box (List[List[int]]): Tracker bounding boxes
        tracker_list (List[Callable]): The list of tracker objects.
        labels (np.ndarray): The labels of detected object in the image.
        scores (np.ndarray): The confidence scores of detected object in the image.
    """
    for trk_idx, det_idx in matched:
        
        z = z_box[det_idx]
        label = labels[det_idx]
        score = scores[det_idx]
        
        z = np.expand_dims(z, axis=0).T
        
        tmp_trk = tracker_list[trk_idx]
        
        tmp_trk.prediction_and_update(z)
        
        x_state = tmp_trk.x_state.T[0].tolist()
        x_state = [x_state[0], x_state[2], x_state[4], x_state[6]]
        x_box[trk_idx] = x_state
        
        tmp_trk.box = x_state
        tmp_trk.hits += 1
        tmp_trk.no_losses = 0
        tmp_trk.label = label
        tmp_trk.score = score


def process_unmatched_trackers(unmatched_trks: np.ndarray, x_box: List[List[int]], tracker_list: List[Callable]) -> None:
    """Checks the unmatched trackers state, predicts new location of objects and deletes the trackers if they exceed max_age.

    Args:
        unmatched_trks (np.ndarray): _description_
        x_box (List[List[int]]): _description_
        tracker_list (List[Callable]): _description_
    """
    for trk_idx in unmatched_trks:
        
        tmp_trk = tracker_list[trk_idx]
        tmp_trk.no_losses += 1
        tmp_trk.predict_only()
        x_state = tmp_trk.x_state
        x_state = x_state.T[0]
        x_state = [x_state[0], x_state[2], x_state[4], x_state[6]]
        tmp_trk.box = x_state
        x_box[trk_idx] = x_state


def assign_detections_to_trackers(trackers: List[Callable], detections: List[np.ndarray], iou_thr: float=0.3) -> Tuple:
    """Matches the detections and the trackers using the Hungarian algorithm.

    Args:
        trackers (List[Callable]): List of tracker objects.
        detections (List[np.ndarray]): The bounding boxes of detected objects in the frame.
        iou_thr (float, optional): The IOU threshold. This is used to match two objects in consecutive frames. Defaults to 0.3.

    Returns:
        Tuple: The list of matched detection and trackers, unmatched detections and unmathced trackers.
    """
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    
    if len(trackers) and len(detections):
        IOU_mat = calc_iou(np.array(trackers), np.array(detections))
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    
    for t in range(len(trackers)):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d in range(len(detections)):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    for m in matched_idx:
        
        if(IOU_mat[m[0],m[1]] < iou_thr):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if (len(matches) == 0):
        matches = np.empty((0,2),dtype=int)
        
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers) 

