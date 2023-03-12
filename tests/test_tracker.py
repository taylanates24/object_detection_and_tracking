import numpy as np
import unittest
import sys
sys.path.append('/workspaces/detection_and_tracking')

from tracker import Tracker


class TestTracker(unittest.TestCase):
    
    def setUp(self):
        
        self.tracker = Tracker()
        
    def test_initial_values(self):
        
        self.assertEqual(self.tracker.id, 0)
        self.assertEqual(self.tracker.box, [])
        self.assertEqual(self.tracker.hits, 0)
        self.assertEqual(self.tracker.no_losses, 0)
        self.assertIsNone(self.tracker.label)
        self.assertIsNone(self.tracker.score)
        self.assertEqual(len(self.tracker.x_state), 0)
        self.assertEqual(self.tracker.dt, 1.)
        
    def test_prediction_and_update(self):

        z = self.init_state()
        
        self.tracker.prediction_and_update(z)
        self.assertEqual(len(self.tracker.x_state), 8)
        self.assertEqual(len(self.tracker.P), 8)
        self.assertEqual(self.tracker.x_state.dtype, int)
    
    def test_predict_only(self):
        
        z = self.init_state()
        
        self.tracker.prediction_and_update(z)
        
        self.tracker.predict_only()
        
        self.assertEqual(len(self.tracker.x_state), 8)
        self.assertEqual(len(self.tracker.P), 8)
        self.assertEqual(self.tracker.x_state.dtype, int)
        
        
    def init_state(self):
        
        z = [10, 20, 30, 40]
        z = np.expand_dims(z, axis=0).T
        x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]], dtype=object).T
        
        self.tracker.x_state = x
        self.tracker.predict_only()
        
        x_state = self.tracker.x_state
        x_state = x_state.T[0].tolist()
        x_state = [x_state[0], x_state[2], x_state[4], x_state[6]]
        self.tracker.box = x_state
        
        return z

if __name__ == '__main__':
    unittest.main()