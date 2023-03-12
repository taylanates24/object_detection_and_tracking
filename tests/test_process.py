import unittest
import numpy as np
from collections import deque
import sys
sys.path.append('/workspaces/detection_and_tracking')
from process import *
from tracker import Tracker

class TestProcessUnmatchedDetections(unittest.TestCase):

    def setUp(self):
        
        self.z_box = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        self.unmatched_dets = np.array([0, 1])
        self.x_box = []
        self.tracker_list = []
        self.track_id_list = deque([0, 1, 2])
        self.labels = [1, 2]
        self.scores = np.array([0.6, 0.8])
    
    def test_process_unmatched_detections(self):
        process_unmatched_detections(self.z_box, self.unmatched_dets, self.x_box, self.tracker_list,
                                     self.track_id_list, self.labels, self.scores)
        
        self.assertEqual(len(self.x_box), 2)
        self.assertEqual(len(self.tracker_list), 2)
        self.assertEqual(len(self.track_id_list), 1)
        
        self.assertEqual(self.tracker_list[0].id, 0)
        self.assertEqual(self.tracker_list[1].id, 1)
        self.assertEqual(self.tracker_list[0].label, 1)
        self.assertEqual(self.tracker_list[1].label, 2)
        self.assertEqual(self.tracker_list[0].score, 0.6)
        self.assertEqual(self.tracker_list[1].score, 0.8)
        
        self.assertTrue(np.array_equal(self.x_box[0], [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(self.x_box[1], [5, 6, 7, 8]))
    
    
class TestProcessMatchedDetections(unittest.TestCase):
    
    def test_process_matched_detections(self):
        
        track_id_list = deque([0, 1, 2])
        z_box = [[0, 1, 2, 3], [10, 11, 12, 13]]
        x_box = [[0, 1, 2, 3], [10, 11, 12, 13]]
        unmatched_dets = [0, 1]
        tracker_list = []
        labels = np.array([0, 1])
        scores = np.array([0.9, 0.8])

        if len(unmatched_dets) > 0:
            process_unmatched_detections(z_box, unmatched_dets, x_box, tracker_list, track_id_list, labels, scores) 
            
        matched = np.array([[0, 0], [1, 1]])
        process_matched_detections(matched, z_box, x_box, tracker_list, labels, scores)

        self.assertEqual(tracker_list[0].box, [0, 1, 2, 3])
        self.assertEqual(tracker_list[0].hits, 1)
        self.assertEqual(tracker_list[0].no_losses, 0)
        self.assertEqual(tracker_list[0].label, 0)
        self.assertEqual(tracker_list[0].score, 0.9)

        self.assertEqual(tracker_list[1].box, [10, 11, 12, 13])
        self.assertEqual(tracker_list[1].hits, 1)
        self.assertEqual(tracker_list[1].no_losses, 0)
        self.assertEqual(tracker_list[1].label, 1)
        self.assertEqual(tracker_list[1].score, 0.8)

class TestProcessUnmatchedTrackers(unittest.TestCase):
    
    def test_process_unmatched_trackers(self):
        # create some dummy data
        unmatched_trks = np.array([0, 1, 2]) # unmatched tracker indices
        x_box = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]] # bounding boxes
        tracker_list = [Tracker(), Tracker(), Tracker()] # tracker instances

        # set initial state of trackers
        tracker_list[0].x_state = np.array([10, 0, 20, 0, 30, 0, 40, 0]).reshape((-1, 1))
        tracker_list[1].x_state = np.array([50, 0, 60, 0, 70, 0, 80, 0]).reshape((-1, 1))
        tracker_list[2].x_state = np.array([90, 0, 100, 0, 110, 0, 120, 0]).reshape((-1, 1))

        # call the function
        process_unmatched_trackers(unmatched_trks, x_box, tracker_list)

        # assert that the tracker boxes were updated correctly
        self.assertEqual(tracker_list[0].box, [10, 20, 30, 40])
        self.assertEqual(tracker_list[1].box, [50, 60, 70, 80])
        self.assertEqual(tracker_list[2].box, [90, 100, 110, 120])

        # assert that the x_box was updated correctly
        expected_x_box = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        self.assertEqual(x_box, expected_x_box)
        
if __name__ == '__main__':
    unittest.main()