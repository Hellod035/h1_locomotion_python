import torch
from pathlib import Path
import time
import sys
import select
import warnings
from threading import Lock

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread, RecurrentThread
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
import unitree_legged_const as h1


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class Locomotion:
    def __init__(self, model_path: Path):
        # init model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.obs = torch.zeros(1, 66, device=self.device)
        self.actions_scale = 0.25

        # init thread
        self.lock = Lock()
        self.init_pose_mode = True
        self.command_thread = Thread(target=self.command_loop)
        self.publish_thread = RecurrentThread(interval=0.02, target=self.publish_cmd)  # 50Hz

        # init buffer
        self.low_state = None
        self.gravity_vec = torch.tensor([0.0, 0.0, -1], device=self.device, requires_grad=False).view(1, 3)
        self.commands = torch.zeros(1, 3, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.projected_gravity = torch.zeros(1, 3, device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(1, 20, device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(1, 20, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.tensor([[ 
            0.0000, # left_hip_yaw_joint
            0.0000, # left_hip_roll_joint
            -0.4000, # left_hip_pitch_joint
            0.8000, # left_knee_joint
            -0.4000, # left_ankle_joint
            0.0000, # right_hip_yaw_joint
            0.0000, # right_hip_roll_joint
            -0.4000, # right_hip_pitch_joint
            0.8000, # right_knee_joint
            -0.4000, # right_ankle_joint
            0.0000, # torso_joint
            0.0000, # left_shoulder_pitch_joint
            0.0000, # left_shoulder_roll_joint
            0.0000, # left_shoulder_yaw_joint
            0.0000, # left_elbow_joint
            0.0000, # right_shoulder_pitch_joint
            0.0000, # right_shoulder_roll_joint
            0.0000, # right_shoulder_yaw_joint
            0.0000 # right_elbow_joint
            ]], device=self.device, requires_grad=False)
        self.dof_names = list(h1.ID.keys())
        self.actions = torch.zeros(1, 19, device=self.device, requires_grad=False)


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


        self.last_publish_time = None

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05

    def set_motor_cmd(self,motor, mode, q=0.0, kp=0.0, dq=0.0, kd=0.0, tau=0.0):
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
        with self.lock:
            self.low_state = msg

    def dof_reindex_to_model(self,input_dofs): # input[1:20] output[1:19]
        indices = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15]
        dof = torch.cat([input_dofs[:, i:i+1] for i in indices], dim=1)
        return dof
    
    def dof_reindex_to_lowcmd(self,input_dofs): # input[1:19] output[1:20]
        indices = [6, 7, 8, 1, 2, 3, 10, 0, 5, 
                   0, # not used
                   4, 9, 15, 16, 17, 18, 11, 12, 13, 14]
        dof = torch.cat([input_dofs[:, i:i+1] for i in indices], dim=1)
        dof[:, 9] = 0
        return dof

    def prepare_commands(self):
        with self.lock:
            # ang_vel [0:3]
            gyro_data = torch.tensor(self.low_state.imu_state.gyroscope, device=self.device)
            self.obs[0,0:3] = gyro_data * self.obs_scales.ang_vel

            # projected_gravity [3:6]
            quaternion = torch.tensor(self.low_state.imu_state.quaternion, device=self.device).view(1, 4)
            quaternion = torch.cat([quaternion[:, 1:4], quaternion[:, 0:1]], dim=1)
            self.projected_gravity = quat_rotate_inverse(quaternion, self.gravity_vec)
            self.obs[0,3:6] = self.projected_gravity

            # commands [6:9]
            self.obs[0,6:9] = self.commands * self.commands_scale

            # dof pos [9:28]
            for i in range(20):
                self.dof_pos[0, i] = self.low_state.motor_state[i].q
                
            dof_pos = self.dof_reindex_to_model(self.dof_pos)
            self.obs[0, 9:28] = (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos

            # dof vel [28:47]
            for i in range(20):
                self.dof_vel[0, i] = self.low_state.motor_state[i].dq

            dof_vel = self.dof_reindex_to_model(self.dof_vel)
            self.obs[0, 28:47] = dof_vel * self.obs_scales.dof_vel

            # last actions [47:66]
            self.obs[0, 47:66] = self.actions

            # predict actions
            self.obs = torch.clip(self.obs, -100.0, 100.0)
            actions = self.model(self.obs.detach())
            actions = torch.clip(actions, -10.0, 10.0)
            self.actions = actions

            # add bias and convert to lowcmd
            actions_add_bias = actions * self.actions_scale + self.default_dof_pos
            actions_to_commands = self.dof_reindex_to_lowcmd(actions_add_bias)
            for i in range(20):
                self.cmd.motor_cmd[i].q = actions_to_commands[0, i].item()

            self.cmd.crc = self.crc.Crc(self.cmd)

    def prepare_init_pose_commands(self):
        with self.lock:
            motor_q = torch.tensor([motor.q for motor in self.low_state.motor_state], device=self.device)
            self.dof_pos[0, :] = motor_q
            
            actions_to_commands = self.dof_reindex_to_lowcmd(self.default_dof_pos)
            
            angle_diff = actions_to_commands[0, :] - self.dof_pos[0, :]
            
            weights = torch.full((20,), 0.1, device=self.device)
            weights[:9] = 0.3
            
            new_angle = self.dof_pos[0, :] + weights * angle_diff
            
            for i in range(20):
                self.cmd.motor_cmd[i].q = new_angle[i].item()
            
            self.cmd.crc = self.crc.Crc(self.cmd)
            
            self.actions = self.default_dof_pos

    def command_loop(self):
        while True:
            # current_time = time.time()
            # if self.last_publish_time is not None:
            #     interval = current_time - self.last_publish_time
            #     print(f"command_loop 调用间隔: {interval:.4f} 秒")
            # self.last_publish_time = current_time
            if self.init_pose_mode:
                self.prepare_init_pose_commands()
            else:
                self.prepare_commands()
                # pass

    def publish_cmd(self):
        # current_time = time.time()
        # if self.last_publish_time is not None:
        #     interval = current_time - self.last_publish_time
        #     print(f"publish_cmd 调用间隔: {interval:.4f} 秒")
        # self.last_publish_time = current_time

        with self.lock:
            # for i in range(20):
            #     print(self.cmd.motor_cmd[i].q)
            # self.pub.Write(self.cmd)
            pass

    def run(self):
        self.command_thread.Start()
        time.sleep(0.1)
        self.publish_thread.Start()


if __name__ == '__main__':

    model_path = Path(__file__).parent / 'locomotion.pt'
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
            time.sleep(0.1)

        # with locomotion.lock:
        #     for i in range(20):
        #         print(locomotion.cmd.motor_cmd[i].q)



