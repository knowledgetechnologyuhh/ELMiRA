#!/usr/bin/env python3
from evo_ik import EvoIK
import numpy as np
from elmira.srv import InverseKinematics, InverseKinematicsResponse
from elmira.msg import JointPosition
from os.path import dirname, abspath, join, pardir
import rospy
import torch


class KinematicsServer:
    def __init__(self):
        rospy.init_node("kinematics_server")
        urdf_dir = join(dirname(abspath(__file__)), pardir, "urdf")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.left_arm = EvoIK(
            join(urdf_dir, "nico_left_arm.urdf"), "left_tcp", device=device
        )
        self.right_arm = EvoIK(
            join(urdf_dir, "nico_right_arm.urdf"), "right_tcp", device=device
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
        # ensure that initial position is in the right order
        joint_ids = np.argsort(request.initial_position.joint_name)
        initial_joints = torch.tensor(request.initial_position.position)[
            joint_ids[
                np.searchsorted(
                    request.initial_position.joint_name,
                    solver.joint_names[:6],
                    sorter=joint_ids,
                )
            ]
        ]
        # solve trajectory
        results = []
        for pose in request.poses:
            ik_result = JointPosition()
            ik_result.joint_name = solver.joint_names[:6]
            pos = pose.position
            quat = pose.orientation
            ik_result.position = solver.inverse_kinematics(
                torch.tensor([pos.x, pos.y, pos.z]).to("cuda"),
                torch.tensor([quat.w, quat.x, quat.y, quat.z]).to("cuda"),
                initial_joints=initial_joints,
                max_steps=100,
            )
            results.append(ik_result)
            # use solution as starting point for the next one
            initial_joints = ik_result.position
        return InverseKinematicsResponse(results)

    # def get_forward_kinematic(self, solver):
    #     rad_angles = torch.tensor(
    #         self.get_motor_angles(solver.joint_names[:6])
    #     ).deg2rad()
    #     return solver.forward_kinematics(rad_angles)


if __name__ == "__main__":
    KinematicsServer()
