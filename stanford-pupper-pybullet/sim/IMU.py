
import time
import numpy as np
import pybullet

class IMU:

    def __init__(self, simulator=True):

        self.simulator = simulator
        self.last_pos = [None, None, None]
        self.last_ang = [None, None, None]

    def _simulator_observation(self, urdf_id=1, dt=1/60):
        # Get the current position and orientation of the robot
        base_position, q = pybullet.getBasePositionAndOrientation(urdf_id)
        base_orientation = np.array([q[3], q[0], q[1], q[2]])
        base_position = np.array(base_position)

        # Compute the rotation matrix and orientation angles
        R = self._compute_rotation_matrix(base_orientation)
        projected_gravity = np.array([0, 0, -9.81]) @ R
        yaw, pitch, roll = self._get_angles_from_quat(base_orientation)
        angular_orientation = np.array([yaw, pitch, roll])

        # Compute the velocity
        if None in self.last_pos:
            velocity = np.array([0, 0, 0])
        else:
            # velocity = (base_position - self.last_pos) / dt # in global frame
            velocity = - np.dot(self.last_pos - base_position, R) / dt # in robot frame
        self.last_pos = base_position

        # Compute the angular velocity
        if None in self.last_ang:
            ang_velocity = np.array([0, 0, 0])
        else:
            # ang_velocity = (angular_orientation - self.last_ang) / dt # in global frame
            ang_velocity = - np.dot(self.last_ang - angular_orientation, R) # in robot frame
        self.last_ang = angular_orientation

        return velocity, ang_velocity, projected_gravity


    def read_orientation(self):
        return pybullet.getBasePositionAndOrientation(1)[1]

    def _compute_rotation_matrix(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0] # W
        q1 = Q[1] # X
        q2 = Q[2] # Y
        q3 = Q[3] # Z

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

        return rot_matrix


    def _get_angles_from_quat(self, q):
        w, x, y, z = q
        yaw = np.arctan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z)
        pitch = np.arcsin(-2.0*(x*z - w*y))
        roll = np.arctan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z)
        return yaw, pitch, roll
