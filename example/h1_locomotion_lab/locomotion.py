import numpy as np
import torch
from pathlib import Path
import time
import sys
import select
import warnings
from threading import Lock
import copy
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread, RecurrentThread
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
import unitree_legged_const as h1

@dataclass
class MotorCmdData:
    motor_cmd: list
    crc: int

class DataBuffer:
    def __init__(self):
        self._lock = Lock()
        self._data = None

    def set_data(self, data):
        with self._lock:
            self._data = data

    def get_data(self):
        with self._lock:
            return self._data


def quat_rotate_inverse(q, v):
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * (2.0 * q_w)[:, np.newaxis]
    c = q_vec * np.sum(q_vec * v, axis=-1)[:, np.newaxis] * 2.0
    return a - b + c

class Locomotion:
    def __init__(self, model_path: Path):
        # init model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.obs = np.zeros((1, 66))
        self.actions_scale = 0.25

        # init thread
        self.init_pose_mode = True
        self.command_thread = RecurrentThread(interval=0.02, target=self.Control)  # 50Hz
        self.publish_thread = RecurrentThread(interval=0.002, target=self.LowCommandWriter)  # 500Hz

        # init buffer
        self.low_state_buffer = DataBuffer()
        self.cmd_buffer = DataBuffer()
        self.gravity_vec = np.array([0.0, 0.0, -1.0])
        self.commands = np.zeros((1, 3))
        self.projected_gravity = np.zeros((1, 3))
        self.dof_pos = np.zeros((1, 20))
        self.dof_vel = np.zeros((1, 20))
        self.default_dof_pos = np.array([[
            0.0000, # left_hip_yaw
            0.0000, # right_hip_yaw
            0.0000, # torso
            0.0000, # left_hip_roll
            0.0000, # right_hip_roll
            0.0000, # left_shoulder_pitch
            0.0000, # right_shoulder_pitch
            -0.4000, # left_hip_pitch
            -0.4000, # right_hip_pitch
            0.0000, # left_shoulder_roll
            0.0000, # right_shoulder_roll
            0.8000, # left_knee
            0.8000, # right_knee
            0.0000, # left_shoulder_yaw
            0.0000, # right_shoulder_yaw
            -0.4000, # left_ankle
            -0.4000, # right_ankle
            0.0000, # left_elbow
            0.0000, # right_elbow
            ]])
        self.dof_names = list(h1.ID.keys())
        self.actions = np.zeros((1, 19))

        # init channel
        ChannelFactoryInitialize(0, "eno1")
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.LowStateHandler, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        # init lowcmd
        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0]=0xFE
        self.cmd.head[1]=0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_hip_roll_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_hip_pitch_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_knee_joint"]], 0x0A, kp=300.0, kd=6.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_hip_roll_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_knee_joint"]], 0x0A, kp=300.0, kd=6.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["torso_joint"]], 0x0A, kp=300.0, kd=6.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_hip_yaw_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_hip_yaw_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_ankle_joint"]], 0x01, kp=40.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_ankle_joint"]], 0x01, kp=40.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_shoulder_pitch_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_shoulder_roll_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_shoulder_yaw_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_elbow_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_shoulder_pitch_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_shoulder_roll_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_shoulder_yaw_joint"]], 0x01, kp=100.0, kd=2.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_elbow_joint"]], 0x01, kp=100.0, kd=2.0)
        self.cmd.crc = self.crc.Crc(self.cmd)

        self.pub_cmd = copy.deepcopy(self.cmd)

        self.last_write_time = None
        self.last_control_time = None

    def set_motor_cmd(self, motor, mode, q=0.0, kp=0.0, dq=0.0, kd=0.0, tau=0.0):
        motor.mode = mode
        motor.q = q
        motor.kp = kp
        motor.dq = dq
        motor.kd = kd
        motor.tau = tau

    def load_model(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def LowStateHandler(self, msg: LowState_):
        self.low_state_buffer.set_data(msg)

    def dof_reindex_to_model(self, input_dofs):  # input[1:20] output[1:19]
        indices = np.array([7, 8, 6, 3, 0, 16, 12, 4, 1, 17, 13, 5, 2, 18, 14, 10, 11, 19, 15])
        dof = input_dofs[:, indices]
        return dof

    def dof_reindex_to_lowcmd(self, input_dofs):  # input[1:19] output[1:20]
        indices = np.array([4, 8, 12, 3, 7, 11, 2, 0, 1, 15, 16, 6, 10, 14, 18, 5, 9, 13, 17])
        dof = np.zeros((input_dofs.shape[0], 20))
        dof[:, :9] = input_dofs[:, indices[:9]]
        dof[:, 10:] = input_dofs[:, indices[9:]]
        return dof

    def prepare_commands(self):
        low_state = copy.deepcopy(self.low_state_buffer.get_data())

        gyro_data = np.empty(3)
        quaternion = np.empty(4)
        motor_data = np.empty((20, 2))

        gyro_data[:] = low_state.imu_state.gyroscope
        quaternion[:] = low_state.imu_state.quaternion

        for i in range(20):
            motor_data[i, 0] = low_state.motor_state[i].q
            motor_data[i, 1] = low_state.motor_state[i].dq

        # ang_vel [0:3]
        self.obs[0, 0:3] = gyro_data

        # projected_gravity [3:6]
        q = np.concatenate([quaternion[1:], quaternion[:1]]).reshape(1, -1)
        self.projected_gravity = quat_rotate_inverse(q, self.gravity_vec)
        self.obs[0, 3:6] = self.projected_gravity

        # commands [6:9]
        self.obs[0, 6:9] = self.commands

        # dof pos [9:28] 和 dof vel [28:47]
        self.dof_pos[0] = motor_data[:, 0]
        self.dof_vel[0] = motor_data[:, 1]

        dof_pos = self.dof_reindex_to_model(self.dof_pos)
        dof_vel = self.dof_reindex_to_model(self.dof_vel)

        self.obs[0, 9:28] = dof_pos - self.default_dof_pos
        self.obs[0, 28:47] = dof_vel

        # last actions [47:66]
        self.obs[0, 47:66] = self.actions

        # predict actions
        with torch.no_grad():
            obs_tensor = torch.from_numpy(self.obs).float().to(self.device)
            obs_tensor = torch.clamp(obs_tensor, -100.0, 100.0)
            actions = self.model(obs_tensor).cpu().numpy()
            actions = np.clip(actions, -10.0, 10.0)
        self.actions = actions

        # add bias and convert to lowcmd
        actions_add_bias = actions * self.actions_scale + self.default_dof_pos
        actions_to_commands = self.dof_reindex_to_lowcmd(actions_add_bias)

        # update cmd
        for i in range(20):
            self.cmd.motor_cmd[i].q = float(actions_to_commands[0, i])

        self.cmd.crc = self.crc.Crc(self.cmd)

        self.cmd_buffer.set_data(copy.deepcopy(self.cmd))

    def prepare_init_pose_commands(self):
        low_state = copy.deepcopy(self.low_state_buffer.get_data())
        motor_q = np.array([motor.q for motor in low_state.motor_state])
        self.dof_pos[0, :] = motor_q

        actions_to_commands = self.dof_reindex_to_lowcmd(self.default_dof_pos)

        angle_diff = actions_to_commands[0, :] - self.dof_pos[0, :]

        weights = np.full((20,), 0.1)
        weights[:9] = 0.3

        new_angle = self.dof_pos[0, :] + weights * angle_diff
        # new_angle = self.dof_pos[0, :]

        for i in range(20):
            self.cmd.motor_cmd[i].q = float(new_angle[i])

        self.cmd.crc = self.crc.Crc(self.cmd)


        self.cmd_buffer.set_data(copy.deepcopy(self.cmd))
        self.actions = self.default_dof_pos

    def Control(self):
        # current_time = time.time()
        # if self.last_control_time is not None:
        #     interval = current_time - self.last_control_time
        #     print(f"Control 调用间隔: {interval:.4f} 秒")
        # self.last_control_time = current_time

        if self.init_pose_mode:
            # start_time = time.time()
            self.prepare_init_pose_commands()
            # print(f"prepare_init_pose_commands 耗时: {time.time() - start_time:.4f} 秒")
        else:
            # start_time = time.time()
            self.prepare_commands()
            # print(f"prepare_commands 耗时: {time.time() - start_time:.4f} 秒")
            # pass

    def LowCommandWriter(self):
        # current_time = time.time()
        # if self.last_write_time is not None:
        #     interval = current_time - self.last_write_time
        #     print(f"LowCommandWriter 调用间隔: {interval:.4f} 秒")
        # self.last_write_time = current_time

        cmd = self.cmd_buffer.get_data()
        # for i in range(20):
        #     print(cmd.motor_cmd[i].q)
        self.pub.Write(cmd)

    def run(self):
        self.command_thread.Start()
        time.sleep(0.5)
        self.publish_thread.Start()


if __name__ == '__main__':

    model_path = Path(__file__).parent / 'policy.pt'
    locomotion = Locomotion(model_path)
    
    print("Switch to init pose mode")
    locomotion.run()

    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 's':
                locomotion.init_pose_mode = False
                locomotion.commands[:,0] = 0.0
                print("Switch to locomotion mode : Stand !")
            if key == 'f':
                locomotion.init_pose_mode = False
                locomotion.commands[:,0] = 0.6
                print("Switch to locomotion mode : Forward !")
            if key == 'b':
                locomotion.init_pose_mode = False
                locomotion.commands[:,0] = -0.6
                print("Switch to locomotion mode : Backward !")
            time.sleep(0.1)

