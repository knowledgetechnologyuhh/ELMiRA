#!/usr/bin/env python3
from evo_ik import EvoIK
from nico_demo.srv import InverseKinematics, InverseKinematicsResponse
from os.path import dirname, abspath, join, pardir
import rospy
import torch


class KinematicsServer:
    def __init__(self):
        rospy.init_node("kinematics_server")
        urdf_dir = join(dirname(abspath(__file__)), pardir, "urdf")
        self.left_arm = EvoIK(
            join(urdf_dir, "nico_left_arm.urdf"), "left_tcp", device="cuda:0"
        )
        self.right_arm = EvoIK(
            join(urdf_dir, "nico_right_arm.urdf"), "right_tcp", device="cuda:0"
        )
        rospy.Service(
            "inverse_kinematics", InverseKinematics, self.get_inverse_kinematics
        )
        rospy.loginfo("EvoIK started successfully")
        rospy.spin()

    def get_inverse_kinematics(self, request):
        if request.planning_group == "l_arm":
            solver = self.left_arm
        elif request.planning_group == "r_arm":
            solver = self.right_arm
        else:
            rospy.logerr(f"Unknown planning group {request.planning_group}")
            return
        response = InverseKinematicsResponse()
        response.joint_name = solver.joint_names[:6]
        pos = request.pose.position
        quat = request.pose.orientation
        response.position = solver.inverse_kinematics(
            torch.tensor([pos.x, pos.y, pos.z]).to("cuda"),
            torch.tensor([quat.w, quat.x, quat.y, quat.z]).to("cuda"),
            max_steps=100,
        )
        return response

    # def get_forward_kinematic(self, solver):
    #     rad_angles = torch.tensor(
    #         self.get_motor_angles(solver.joint_names[:6])
    #     ).deg2rad()
    #     return solver.forward_kinematics(rad_angles)


if __name__ == "__main__":
    KinematicsServer()
