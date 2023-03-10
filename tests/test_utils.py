import unittest
import numpy as np
import sys
sys.path.append('/workspaces/detection_and_tracking')
from utils import calc_iou

class TestCalcIOU(unittest.TestCase):
    
    def test_calc_iou(self):
        # Test case 1
        bboxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
        bboxes2 = np.array([[5, 5, 15, 15], [25, 25, 35, 35]])
        expected_iou = np.array([[0.17475728, 0.        ],
                                 [0.        , 0.17475728],
                                 [0.        , 0.        ]])
        iou = calc_iou(bboxes1, bboxes2)
        np.testing.assert_array_almost_equal(iou, expected_iou)
        
        # Test case 2
        bboxes1 = np.array([[0, 0, 20, 20], [30, 30, 50, 50]])
        bboxes2 = np.array([[10, 10, 30, 30], [40, 40, 60, 60]])
        expected_iou = np.array([[0.15900131, 0.        ],
                                 [0.00113507, 0.15900131]])
        iou = calc_iou(bboxes1, bboxes2)
        np.testing.assert_array_almost_equal(iou, expected_iou)
        
        # Test case 3
        bboxes1 = np.array([[0, 0, 10, 10]])
        bboxes2 = np.array([[5, 5, 15, 15]])
        expected_iou = np.array([[0.17475728]])
        iou = calc_iou(bboxes1, bboxes2)
        np.testing.assert_array_almost_equal(iou, expected_iou)
        
        # Test case 4
        bboxes1 = np.array([[0, 0, 10, 10]])
        bboxes2 = np.array([[15, 15, 20, 20]])
        expected_iou = np.array([[0]])
        iou = calc_iou(bboxes1, bboxes2)
        np.testing.assert_array_almost_equal(iou, expected_iou)
        
        # Test case 5
        bboxes1 = np.array([[0, 0, 10, 10]])
        bboxes2 = np.array([[0, 0, 10, 10]])
        expected_iou = np.array([[1]])
        iou = calc_iou(bboxes1, bboxes2)
        np.testing.assert_array_almost_equal(iou, expected_iou)
        
        # Test case 6
        bboxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
        bboxes2 = np.array([[5, 5, 15, 15], [25, 25, 35, 35]])
        expected_iou = np.array([[0.17475728, 0.        ],
                                 [0.        , 0.17475728],
                                 [0.        , 0.        ]])
        iou = calc_iou(bboxes1, bboxes2)
        np.testing.assert_array_almost_equal(iou, expected_iou)
        
if __name__ == '__main__':
    unittest.main()