import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread
import unitree_legged_const as h1

crc = CRC()

def set_motor_cmd(motor, mode, q=0, kp=0, dq=0, kd=0, tau=0):
    motor.mode = mode
    motor.q = q
    motor.kp = kp
    motor.dq = dq
    motor.kd = kd
    motor.tau = tau


if __name__ == '__main__':

    ChannelFactoryInitialize(0, "eno1")
    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0]=0xFE
    cmd.head[1]=0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    [set_motor_cmd(cmd.motor_cmd[i], 0x0A) for i in range(9)]
    [set_motor_cmd(cmd.motor_cmd[i], 0x01) for i in range(10, 20)]

    while True:  
        id = 10
        # Toque controle, set left_elbow_joint toque
        cmd.motor_cmd[id].q = -0.0 # Set to stop position(rad)
        cmd.motor_cmd[id].kp = 20
        cmd.motor_cmd[id].dq = 0.0 # Set to stop angular velocity(rad/s)
        cmd.motor_cmd[id].kd = 4
        cmd.motor_cmd[id].tau = 0.0 # target toque is set to 1N.m

        # # Poinstion(rad) control, set right_elbow_joint rad
        # cmd.motor_cmd[h1.ID["right_elbow_joint"]].q = 0.0  # Taregt angular(rad)
        # cmd.motor_cmd[h1.ID["right_elbow_joint"]].kp = 10.0 # Poinstion(rad) control kp gain
        # cmd.motor_cmd[h1.ID["right_elbow_joint"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        # cmd.motor_cmd[h1.ID["right_elbow_joint"]].kd = 1.0  # Poinstion(rad) control kd gain
        # cmd.motor_cmd[h1.ID["right_elbow_joint"]].tau = 0.0 # Feedforward toque 1N.m

        # cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]].q = 0  # Taregt angular(rad)
        # cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]].kp = 10.0 # Poinstion(rad) control kp gain
        # cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        # cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]].kd = 1.0  # Poinstion(rad) control kd gain
        # cmd.motor_cmd[h1.ID["left_hip_pitch_joint"]].tau = 0.0 # Feedforward toque 1N.m
        
        cmd.crc = crc.Crc(cmd)

        #Publish message
        if pub.Write(cmd):
            print("Publish success. msg:", cmd.crc)
        else:
            print("Waitting for subscriber.")

        time.sleep(0.02) # 50Hz
