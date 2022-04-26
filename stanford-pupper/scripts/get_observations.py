
import time
import numpy as np
from woofer.imu import Imu
from woofer.encoders import Encoders

imu = Imu()
encoders = Encoders()


def main():
    vel_x = []
    vel_y = []
    vel_z = []
    ang_vel_x = []
    ang_vel_y = []
    ang_vel_z = []
    joint_vel = []
    joint_pos = []
    time_list = []
    start_time = time.time()
    finished = False
    while not finished:
        velocity = imu.read_imu_vel()
        ang_vel = imu.read_ang_vel()
        joint_pos.append(encoders.read_encoder_position(0,1))
        joint_vel.append(encoders.read_encoder_vel(0,1))
        vel_x.append(velocity[0])
        vel_y.append(velocity[1])
        vel_z.append(velocity[2])
        ang_vel_x.append(ang_vel[0])
        ang_vel_y.append(ang_vel[1])
        ang_vel_z.append(ang_vel[2])
        time_elapsed = time.time() - start_time
        finished = time_elapsed > 20
        time_list.append(time_elapsed)
        time.sleep(0.3)

    np.savez("data/observation_sample.npz", t=np.array(time_list),
                                       vel_x=np.array(vel_x), 
                                       vel_y=np.array(vel_y), 
                                       vel_z=np.array(vel_z),
                                       ang_vel_x=np.array(ang_vel_z),
                                       ang_vel_y=np.array(ang_vel_y),
                                       ang_vel_z=np.array(ang_vel_z),
                                       joint_pos=np.array(joint_pos),
                                       joint_vel=np.array(joint_vel))


if __name__ == "__main__":
    main()    


