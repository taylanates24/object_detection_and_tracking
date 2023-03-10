import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
from typing import List

class Tracker():
    def __init__(self, id: int=0, box: List[int]=[], hits: int=0, no_losses: int=0, label: int=None, score: float=None,
                 x_state: List=[], dt: float=1.0, F: np.ndarray=None, H: np.ndarray=None, L: float=10.0, 
                 Q_comp_mat: np.ndarray=None, R_scalar: float=1.0):
        """The Kalman Filter class.

        Args:
            id (int, optional): ID of tracker object. Defaults to 0.
            box (List[int], optional): Bounding box of tracker object. Defaults to [].
            hits (int, optional): The number of showing up of a tracker. Defaults to 0.
            no_losses (int, optional): The number of show up of a tracker without a matched detection box. Defaults to 0.
            label (int, optional): Label of the tracked object. Defaults to None.
            score (float, optional): Confidence score of tracker object. Defaults to None.
            x_state (List, optional): The x state. Defaults to None.
            dt (float, optional): Time step. Defaults to 1.0.
            F (np.ndarray, optional): The process matrix. Defaults to None.
            H (np.ndarray, optional): The measurement matrix. Defaults to None.
            L (float, optional): The state covariance value. Defaults to 10.0.
            Q_comp_mat (np.ndarray, optional): Process covariance unit. Defaults to None.
            R_scalar (float, optional): The measurement covariance unit. Defaults to 1.0.
        """
        self.id = id
        self.box = box
        self.hits = hits
        self.no_losses = no_losses 
        self.label = label
        self.score = score

        self.x_state = x_state
        self.dt = dt
        
        if F is None:
            self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                            [0, 1,  0,  0,  0,  0,  0, 0],
                            [0, 0,  1,  self.dt, 0,  0,  0, 0],
                            [0, 0,  0,  1,  0,  0,  0, 0],
                            [0, 0,  0,  0,  1,  self.dt, 0, 0],
                            [0, 0,  0,  0,  0,  1,  0, 0],
                            [0, 0,  0,  0,  0,  0,  1, self.dt],
                            [0, 0,  0,  0,  0,  0,  0,  1]])
        
        if H is None:
            self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 1, 0]])
        

        self.L = L
        self.P = np.diag(self.L*np.ones(8))
        
        if Q_comp_mat is None:
            self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                        [self.dt**3/2., self.dt**2]])
        
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)
        
        self.R_scaler = R_scalar
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)
        
        
    def update_R(self):
        """Updates the measurement covariance, R.
        """
           
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)
        
        
    def prediction_and_update(self, z: List[int]): 
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