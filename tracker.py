import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
from typing import List

class Tracker():
    def __init__(self):
        """
        Kalman Filter class.
        """
        self.id = 0 
        self.box = [] 
        self.hits = 0 
        self.no_losses = 0 
        self.label = None
        self.score = None

        self.x_state=[] 
        self.dt = 1.
        
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])
        

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        

        self.L = 10.0
        self.P = np.diag(self.L*np.ones(8))
        
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)
        
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)
        
        
    def update_R(self):
        """Updates the measurement covariance, R.
        """
           
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)
        
        
    def kalman_filter(self, z: List[int]): 
        """Implements prediction and update phases of the Kalman Filter.

        Args:
            z (List[int]): The bounding box comes from the detector.
        """
        
        self.x_state = dot(self.F, self.x_state)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S))
        y = z - dot(self.H, self.x_state)
        self.x_state += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = self.x_state.astype(int)
        
        
    def predict_only(self):  
        """Implements the prediction phase of the Kalman Filter.
        """
        
        self.x_state = dot(self.F, self.x_state)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = self.x_state.astype(int)