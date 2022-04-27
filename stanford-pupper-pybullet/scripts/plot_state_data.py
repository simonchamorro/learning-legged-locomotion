import numpy as np
import matplotlib.pyplot as plt


DT = 0.02
TARGET = [0.1, 0.0]
ACTION_SCALE = 0.25
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CLIP_OBS = 100.0
CLIP_ACT = 100.0
DEFAULT_JOINT_POS = np.array([-0.15, 0.5, -1.0, 0.15, 0.5, -1.0,
                              -0.15, 0.7, -1.0, 0.15, 0.7, -1.0])


def plot(log):
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols)

    for key, value in log.items():
        time = np.linspace(0, len(value)*DT, len(value))
        break
    
    # plot joint targets and measured positions
    a = axs[1, 0]
    a.plot(time, log["dof_pos"], label='measured')
    # a.plot(time, log["dof_pos_target"], label='target')
    a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
    a.legend()

    # plot joint velocity
    a = axs[1, 1]
    a.plot(time, log["dof_vel"], label='measured')
    # a.plot(time, log["dof_vel_target"], label='target')
    a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
    a.legend()

    # plot base vel x
    a = axs[0, 0]
    a.plot(time, log["base_vel_x"], label='measured')
    a.plot(time, log["command_x"], label='commanded')
    a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
    a.legend()

    # plot base vel y
    a = axs[0, 1]
    a.plot(time, log["base_vel_y"], label='measured')
    a.plot(time, log["command_y"], label='commanded')
    a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
    a.legend()

    # plot base vel yaw
    a = axs[0, 2]
    a.plot(time, log["base_vel_yaw"], label='measured')
    a.plot(time, log["command_yaw"], label='commanded')
    a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
    a.legend()

    # plot base vel z
    a = axs[1, 2]
    a.plot(time, log["base_vel_z"], label='measured')
    a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
    a.legend()

    # # plot contact forces
    # a = axs[2, 0]
    # if log["contact_forces_z"]:
    #     forces = np.array(log["contact_forces_z"])
    #     for i in range(forces.shape[1]):
    #         a.plot(time, forces[:, i], label=f'force {i}')
    # a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
    # a.legend()

    # # plot torque/vel curves
    # a = axs[2, 1]
    # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
    # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
    # a.legend()

    # # plot torques
    # a = axs[2, 2]
    # if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
    # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
    # a.legend()

    plt.show()



if __name__ == "__main__":
    data = np.load("data/pybullet_states.npz")["data"]

    log = {}
    time = np.arange(0, data.shape[0]) * 0.02
    
    # Lin vel
    log["base_vel_x"] = data[:,0] / LIN_VEL_SCALE
    log["base_vel_y"] = data[:,1] / LIN_VEL_SCALE
    log["base_vel_z"] = data[:,2] / LIN_VEL_SCALE

    # Ang vel
    log["base_vel_yaw"] = data[:,3] / ANG_VEL_SCALE

    # Projected gravity
    log["projected_gravity"] = data[:,6:9]

    # Command
    log["command_x"] = data[:,9] / LIN_VEL_SCALE
    log["command_y"] = data[:,10] / LIN_VEL_SCALE
    log["command_yaw"] = data[:,11] / LIN_VEL_SCALE

    # DOF pos
    log["dof_pos"] = data[:,13] / DOF_POS_SCALE # 12:24
    log["dof_vel"] = data[:,25] / DOF_VEL_SCALE # 24:36
    
    # Action
    log["last_action"] = data[:,37] # 36:48
    
    plt.figure()
    plot(log)
