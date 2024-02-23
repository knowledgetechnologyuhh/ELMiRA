from os.path import dirname, abspath, join, pardir
import signal
import time
import torch
from nicomotion.Motion import Motion
from evo_ik import EvoIK


class RobotController:
    def __init__(self):
        urdf_dir = join(dirname(abspath(__file__)), pardir, "urdf")
        self.left_arm = EvoIK(
            join(urdf_dir, "nico_left_arm.urdf"), "left_tcp", device="cuda:0"
        )
        self.right_arm = EvoIK(
            join(urdf_dir, "nico_right_arm.urdf"), "right_tcp", device="cuda:0"
        )
        json_config = join(
            dirname(abspath(__file__)), pardir, "json/nico_humanoid_upper.json"
        )
        self.robot = Motion(json_config)

        # Emergency sigint handler to stop the robot with ctrl-c
        def sigint_handler(sig, frame):
            print("Execution aborted")
            self.robot.disableTorqueAll()
            exit(130)

        signal.signal(signal.SIGINT, sigint_handler)
        self.initial_position()

    def __del__(self):
        if self.robot is not None:
            self.close()

    def close(self):
        self.robot.disableTorqueAll()
        self.robot.cleanup()
        self.robot = None

    def initial_position(self):
        self.look_up()
        self.move_motors_to_angles(
            self.left_arm.joint_names[:6] + self.right_arm.joint_names[:6],
            [45.0, -45.0, 0.0, 50.0, 0.0, 0.0, -45.0, 45.0, 0.0, -50.0, 0.0, 0.0],
        )

    def look_up(self, speed=0.05):
        self.move_motors_to_angles(["head_z", "head_y"], [0.0, 0.0], speed)

    def look_down(self, speed=0.05):
        self.move_motors_to_angles(["head_z", "head_y"], [0.0, 47.0], speed)
        time.sleep(1.0)

    def move_arm_to_pose(self, solver, position, euler_angles, convention="ZYX"):
        joint_names = solver.joint_names[:6]
        angles = solver.inverse_kinematics_from_euler(
            position, euler_angles.deg2rad(), max_steps=100
        )
        self.safety_check(solver, angles, position)
        self.move_motors_to_angles(joint_names, angles.rad2deg())

    def move_motors_to_angles(self, motors, angles, speed=0.03):
        for i, motor in enumerate(motors):
            self.robot.setAngle(motor, angles[i], speed)
        time.sleep(0.2)
        moving = True
        while moving:
            time.sleep(0.1)
            max_speed = max([abs(self.robot.getSpeed(motor)) for motor in motors])
            moving = max_speed != 0

    def get_motor_angles(self, motors):
        return [self.robot.getAngle(motor) for motor in motors]

    def get_forward_kinematic(self, solver):
        rad_angles = torch.tensor(
            self.get_motor_angles(solver.joint_names[:6])
        ).deg2rad()
        return solver.forward_kinematics(rad_angles)

    def safety_check(self, solver, angles, target_position):
        ik_position = solver.forward_kinematics(angles).get_matrix()[:, :3, 3].squeeze()
        distance = torch.linalg.norm(ik_position - target_position)
        if ik_position[2] < 0.66:
            raise UnsafeKinematicsException(
                "Inverse kinematics result below safe height"
            )
        if distance > 0.02:
            raise UnsafeKinematicsException(
                f"Inverse kinematics result too imprecise (Error: {distance.item()}cm)"
            )

    def touch(self, x, y, z=0.68):
        if y >= 0:
            solver = self.left_arm
            euler_angles = torch.tensor([0.0, 0.0, 90.0])
        else:
            solver = self.right_arm
            euler_angles = torch.tensor([0.0, 0.0, -90.0])
        position = torch.cat(
            (
                torch.unsqueeze(x - 0.03, dim=0),
                torch.unsqueeze(y, dim=0),
                torch.tensor([z]).to("cuda"),
            )
        )
        self.move_arm_to_pose(solver, position, euler_angles)

    def show(self, x, y, z=0.68):
        print("Execute show action")

    def push(self, x, y, z=0.68):
        print("Execute push action")


class UnsafeKinematicsException(Exception):
    pass
