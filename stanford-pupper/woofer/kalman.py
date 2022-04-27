import time
import numpy as np


class Kalman:

    def __init__(self, cov=0.2, measurement_cov=0.5, command=np.array( [[0],[0],[0]])):
        #Transition matrix
        self.F_t = np.array([[1,0,0] , [0,1,0] , [0,0,1]])

        #Initial State cov
        self.prev_P_t = np.identity(3) * cov

        # Process cov
        self.Q_t = np.identity(3)

        # Control matrix
        self.B_t = np.identity(3)

        # Control vector velocity (Policy)
        self.U_t = command # [3 X 1]

        # Measurment Matrix
        self.H_t = np.array([ [1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]])

        # Measurment cov
        self.R_t = np.identity(3) * measurement_cov
        self.prev_X_hat_t = np.array( [[0],[0],[0]])


    def _prediction(self, X_hat_t_1, P_t_1):

        X_hat_t = self.F_t.dot(X_hat_t_1) + self.B_t.dot(self.U_t).reshape(self.B_t.shape[0], -1)
        P_t = np.diag(np.diag(self.F_t.dot(P_t_1).dot(self.F_t.transpose())))+self.Q_t
        
        return X_hat_t,P_t      
    

    def _update(self, X_hat_t, P_t, Z_t):

        K_prime = P_t.dot(self.H_t.transpose()).dot( np.linalg.inv ( self.H_t.dot(P_t).dot(self.H_t.transpose()) + self.R_t ) )  
        
        X_t = X_hat_t + K_prime.dot(Z_t - self.H_t.dot(X_hat_t))
        P_t = P_t - K_prime.dot(self.H_t).dot(P_t)
        
        return X_t,P_t

    def compute_kalman(self, data):
        # Prediction model
        X_hat_t, P_hat_t = self._prediction(self.prev_X_hat_t, self.prev_P_t)
        
        # Measurement
        Z_t = np.array([[data[0]], [data[1]], [data[2]]]) + self.prev_X_hat_t
        
        # Update estimator
        X_t, P_t = self._update(X_hat_t, P_hat_t, Z_t)
        
        self.prev_X_hat_t = X_t
        self.prev_P_t = P_t

        return X_t
        

    
