import unittest
import numpy as np
from typing import List, Callable, Deque
from collections import deque
import sys
sys.path.append('/workspaces/detection_and_tracking')
from process import process_unmatched_detections

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
        
if __name__ == '__main__':
    unittest.main()