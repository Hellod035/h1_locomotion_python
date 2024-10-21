import torch
from pathlib import Path
import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
import unitree_legged_const as h1
import warnings
DEBUG_MODE = True

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

        # init buffer
        self.low_state = None
        self.gravity_vec = torch.tensor([0.0, 0.0, -1], device=self.device, requires_grad=False).view(1, 3)
        self.commands = torch.zeros(1, 3, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False) # TODO change this
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
            -0.0000, # left_elbow_joint
            0.0000, # right_shoulder_pitch_joint
            0.0000, # right_shoulder_roll_joint
            0.0000, # right_shoulder_yaw_joint
            -0.0000 # right_elbow_joint
            ]], device=self.device, requires_grad=False)
        self.dof_names = list(h1.ID.keys())
        self.actions = torch.zeros(1, 19, device=self.device, requires_grad=False)


        # init channel
        if DEBUG_MODE:
            ChannelFactoryInitialize(0)
        else:
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
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["right_knee_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_hip_roll_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]], 0x0A, kp=200.0, kd=5.0)
        self.set_motor_cmd(self.cmd.motor_cmd[h1.ID["left_knee_joint"]], 0x0A, kp=200.0, kd=5.0)
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

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05

    def set_motor_cmd(self,motor, mode, q=0, kp=0, dq=0, kd=0, tau=0):
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
        self.low_state = msg

    def publish_cmd(self):
        self.pub.Write(self.cmd)

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

    def prepare_init_pose_commands(self):
        for i in range(20):
            self.dof_pos[0, i] = self.low_state.motor_state[i].q

        actions_to_commands = self.dof_reindex_to_lowcmd(self.default_dof_pos)

        for i in range(20):
            current_angle = self.dof_pos[0, i].item()
            target_angle = actions_to_commands[0, i].item()
            angle_diff = target_angle - current_angle
            
            if i < 9:
                new_angle = current_angle + 0.3 * angle_diff
            else:
                new_angle = current_angle + 0.1 * angle_diff
            
            self.cmd.motor_cmd[i].q = new_angle

        self.cmd.crc = self.crc.Crc(self.cmd)

if __name__ == '__main__':
    # load model
    model_path = Path(__file__).parent / 'locomotion.pt'
    locomotion = Locomotion(model_path)

    while True:
        start_time = time.time()
        locomotion.prepare_init_pose_commands()
        for i in range(20):
            print(i, locomotion.cmd.motor_cmd[i].q)
        locomotion.publish_cmd()

        time.sleep(0.02)
        

