import numpy as np
from tracker import Tracker
from helpers import box_iou2
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
def process_unmatched_detections(z_box, unmatched_dets, x_box, tracker_list, track_id_list, labels, scores):
    
    for idx in unmatched_dets:
        z = z_box[idx]
        label = labels[idx]
        score = scores[idx]
        z = np.expand_dims(z, axis=0).T
        tmp_trk = Tracker() # Create a new tracker
        x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]], dtype=object).T
        tmp_trk.x_state = x
        tmp_trk.predict_only()
        xx = tmp_trk.x_state
        xx = xx.T[0].tolist()
        xx =[xx[0], xx[2], xx[4], xx[6]]
        tmp_trk.box = xx
        tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
        tracker_list.append(tmp_trk)
        x_box.append(xx)
        tmp_trk.label = label
        tmp_trk.score = score
        

def process_matched_detections(matched, z_box, x_box, tracker_list, labels, scores):

    for trk_idx, det_idx in matched:
        z = z_box[det_idx]
        label = labels[det_idx]
        score = scores[det_idx]
        z = np.expand_dims(z, axis=0).T
        tmp_trk= tracker_list[trk_idx]
        tmp_trk.kalman_filter(z)
        xx = tmp_trk.x_state.T[0].tolist()
        xx =[xx[0], xx[2], xx[4], xx[6]]
        x_box[trk_idx] = xx
        tmp_trk.box =xx
        tmp_trk.hits += 1
        tmp_trk.no_losses = 0
        tmp_trk.label = label
        tmp_trk.score = score


def process_unmatched_trackers(unmatched_trks, x_box, tracker_list):
    
    for trk_idx in unmatched_trks:
        tmp_trk = tracker_list[trk_idx]
        tmp_trk.no_losses += 1
        tmp_trk.predict_only()
        xx = tmp_trk.x_state
        xx = xx.T[0]
        xx =[xx[0], xx[2], xx[4], xx[6]]
        tmp_trk.box =xx
        x_box[trk_idx] = xx

def calc_iou(bboxes1, bboxes2):
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
        return 0
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def assign_detections_to_trackers(trackers, detections, labels, scores, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    



    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    
    if len(trackers) and len(detections):
        IOU_mat = calc_iou(np.array(trackers), np.array(detections))
    
    '''  
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = box_iou2(trk,det)'''
    
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
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers) 

