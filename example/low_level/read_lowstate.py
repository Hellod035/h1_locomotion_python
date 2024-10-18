import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

import unitree_legged_const as h1


def LowStateHandler(msg: LowState_):
    
    # print front right hip motor states
    # print("left_elbow_joint motor state: ", msg.motor_state[h1.ID["left_elbow_joint"]])
    # print("right_elbow_joint motor state: ", msg.motor_state[h1.ID["right_elbow_joint"]])
    for i in range(20):
        print("motor state : ", i, msg.motor_state[i].q)
    # print("IMU state: ", msg.imu_state)
    # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)


if __name__ == "__main__":
    ChannelFactoryInitialize(0, "eno1")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)

    while True:
        time.sleep(0.01)
