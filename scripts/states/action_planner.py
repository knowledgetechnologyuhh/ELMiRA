import numpy as np
import rospy
import smach
import smach_ros

from geometry_msgs.msg import Pose, Point, Quaternion
from nico_demo.srv import (
    DetectObjects,
    CoordinateTransfer,
    InverseKinematics,
    CheckLLMObjectVisibility,
)


class ObjectSelector(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["succeeded", "object_not_found", "object_out_of_reach"],
            input_keys=["objects", "target_object"],
            output_keys=["image_x", "image_y", "system_message"],
        )
        self.workspace = np.array(
            [
                [0.0396, 0.7160],
                [0.2021, 0.3444],
                [0.7646, 0.3278],
                [0.9448, 0.7313],
                [0.8162, 0.8069],
                [0.6391, 0.8632],
                [0.4380, 0.8757],
                [0.2599, 0.8375],
                [0.1328, 0.7771],
            ]
        )

    def within_workspace(self, x, y):
        cross_products = np.array(
            [
                (x - self.workspace[i - 1][0])
                * (self.workspace[i][1] - self.workspace[i - 1][1])
                - (self.workspace[i][0] - self.workspace[i - 1][0])
                * (y - self.workspace[i - 1][1])
                for i in range(len(self.workspace))
            ],
        )
        return np.logical_or(np.all(cross_products <= 0), np.all(cross_products >= 0))

    def execute(self, userdata):
        if len(userdata.objects) == 0:
            rospy.logwarn(
                f"SYSTEM: Could not find {userdata.target_object} in the image."
            )
            userdata.system_message = (
                f"SYSTEM: Could not find {userdata.target_object} in the image."
            )
            return "object_not_found"
        high_to_low = np.argsort([-obj.score for obj in userdata.objects])
        for obj in np.array(userdata.objects)[high_to_low]:
            bottom_x = obj.center_x
            bottom_y = obj.center_y + obj.height / 2
            if self.within_workspace(bottom_x, bottom_y):
                userdata.image_x = bottom_x
                userdata.image_y = bottom_y
                return "succeeded"
        rospy.logwarn(
            f"SYSTEM: All {len(userdata.objects)} detected candidates for {userdata.target_object} in the image are out of reach."
        )
        userdata.system_message = f"SYSTEM: All {len(userdata.objects)} detected candidates for {userdata.target_object} in the image or out of reach."
        return "object_out_of_reach"


class ActionTrajectory(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["succeeded", "unknown_action"],
            input_keys=["action_type", "target_x", "target_y", "target_z"],
            output_keys=[
                "planning_group",
                "poses",
            ],  # TODO: change from one pose to list
        )

    def execute(self, userdata):
        is_right = userdata.target_y < 0
        # calculate target for action
        target_poses = []
        if userdata.action_type == "touch":
            target_pose = Pose()
            target_pose.position = Point(
                userdata.target_x - 0.0, userdata.target_y, userdata.target_z
            )
            if is_right:  # x, y, z, w
                target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
            target_poses.append(target_pose)
        elif userdata.action_type == "show":
            target_pose = Pose()
            target_pose.position = Point(
                userdata.target_x - 0.08, userdata.target_y, userdata.target_z + 0.03
            )
            if is_right:  # x, y, z, w
                target_pose.orientation = Quaternion(-1.0, 0, 0, 0.0)
            else:
                target_pose.orientation = Quaternion(1.0, 0, 0, 0.0)
            target_poses.append(target_pose)
        elif userdata.action_type == "push":
            for offset in [
                (-0.04, 0.0, 0.0),
                (0.03, 0.0, 0.0),
                (0.06, 0.0, 0.0),
                (0.06, 0.0, 0.10),
            ]:
                target_pose = Pose()
                target_pose.position = Point(
                    userdata.target_x + offset[0],
                    userdata.target_y + offset[1],
                    userdata.target_z + offset[2],
                )
                if is_right:  # x, y, z, w
                    target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
                else:
                    target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
                target_poses.append(target_pose)
        elif userdata.action_type == "push_left":
            # 08, 06
            for offset in [
                (0.04, -0.10, 0.10),
                (0.04, -0.10, 0.0),
                (0.04, 0.04, 0.0),
                (0.04, 0.04, 0.10),
            ]:
                target_pose = Pose()
                target_pose.position = Point(
                    userdata.target_x + offset[0],
                    userdata.target_y + offset[1],
                    userdata.target_z + offset[2],
                )
                if is_right:  # x, y, z, w
                    target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
                else:
                    target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
                target_poses.append(target_pose)
        elif userdata.action_type == "push_right":
            for offset in [
                (0.04, 0.10, 0.10),
                (0.04, 0.10, 0.0),
                (0.04, -0.04, 0.0),
                (0.04, -0.04, 0.10),
            ]:
                target_pose = Pose()
                target_pose.position = Point(
                    userdata.target_x + offset[0],
                    userdata.target_y + offset[1],
                    userdata.target_z + offset[2],
                )
                if is_right:  # x, y, z, w
                    target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
                else:
                    target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
                target_poses.append(target_pose)
        else:
            rospy.loginfo(f"Action '{userdata.action_type}' not defined")
            userdata.system_message(
                f"SYSTEM: Action '{userdata.action_type}' not defined"
            )
            return "unknown_action"
        # set output userdata
        userdata.planning_group = "r_arm" if is_right else "l_arm"
        userdata.poses = target_poses
        return "succeeded"


class ActionPlanner(smach.StateMachine):
    """Locates target object and plans motion trajectory for the requested action."""

    def __init__(
        self,
    ):
        super(ActionPlanner, self).__init__(
            input_keys=["action_type", "target_object", "table_z", "motion_init_pose"],
            output_keys=["joint_trajectory", "system_message", "real_x", "real_y"],
            outcomes=[
                "succeeded",
                "preempted",
                "aborted",
                "system_out",
            ],
        )
        # Open the container
        with self:
            # detect object in image space
            smach.StateMachine.add(
                "OBJECT_DETECTION",
                smach_ros.ServiceState(
                    "object_detector",
                    DetectObjects,
                    request_slots=["texts"],
                    response_slots=["objects"],
                ),
                transitions={
                    "succeeded": "CHOOSE_TARGET_OBJECT",
                },
                remapping={
                    "texts": "target_object",
                },
            )
            # process dected objects
            smach.StateMachine.add(
                "CHOOSE_TARGET_OBJECT",
                ObjectSelector(),
                transitions={
                    "succeeded": "COORDINATE_TRANSFER",
                    "object_not_found": "system_out",  # TODO combine not found and out of reach?
                    "object_out_of_reach": "system_out",
                },
            )
            # image to real
            smach.StateMachine.add(
                "COORDINATE_TRANSFER",
                smach_ros.ServiceState(
                    "image_to_real",
                    CoordinateTransfer,
                    request_slots=["image_x", "image_y"],
                    response_slots=["real_x", "real_y"],
                ),
                transitions={"succeeded": "PLAN_ACTION_TARGETS"},
            )
            # set action target
            # TODO one state per action? maybe in sub-statemachine?
            smach.StateMachine.add(
                "PLAN_ACTION_TARGETS",
                ActionTrajectory(),
                transitions={
                    "succeeded": "SOLVE_IK",
                    "unknown_action": "system_out",
                },
                remapping={
                    "target_x": "real_x",
                    "target_y": "real_y",
                    "target_z": "table_z",
                },
            )

            # callback to create initial pose
            def ik_request_callback(userdata, request):
                initial_position = userdata.motion_init_pose[userdata.planning_group]
                request.initial_position.joint_name = initial_position["names"]
                request.initial_position.position = initial_position["positions"]
                return request

            # callback to post-process detected objects
            def ik_response_callback(userdata, response):
                userdata.joint_trajectory = [
                    {
                        userdata.planning_group: {
                            "names": position.joint_name,
                            "positions": position.position,
                        }
                    }
                    for position in response.positions
                ] + [userdata.motion_init_pose]
                return "succeeded"

            # IK Solver
            smach.StateMachine.add(
                "SOLVE_IK",
                smach_ros.ServiceState(
                    "inverse_kinematics",
                    InverseKinematics,
                    input_keys=["planning_group", "motion_init_pose"],
                    request_slots=["planning_group", "poses"],
                    output_keys=["joint_trajectory"],
                    request_cb=ik_request_callback,
                    response_cb=ik_response_callback,
                ),
                transitions={
                    "succeeded": "succeeded",
                },
            )


class ConcurrentPlanAndVerify(smach.Concurrence):
    """Starts Action Planning and LLM Vision Verification in parallel to safe time."""

    def __init__(
        self,
    ):
        super(ConcurrentPlanAndVerify, self).__init__(
            input_keys=[
                "action_type",
                "target_object",
                "llm_input",
                "table_z",
                "motion_init_pose",
            ],
            output_keys=["joint_trajectory", "system_message", "real_x", "real_y"],
            outcomes=[
                "succeeded",
                "preempted",
                "aborted",
                "system_out",
            ],
            default_outcome="system_out",
            child_termination_cb=self.child_termination_cb,
            outcome_cb=self.outcome_cb,
        )
        # Open the container
        with self:
            smach.Concurrence.add(
                "ACTION_PLANNER",
                ActionPlanner(),
            )

            def object_visible_response_cb(userdata, response):
                if not response.object_visible:
                    rospy.logwarn(response.system_message)
                    return "object_not_visible"
                return "succeeded"

            smach.Concurrence.add(
                "CHECK_OBJECT_VISIBILITY",
                smach_ros.ServiceState(
                    "llm_object_visibility",
                    CheckLLMObjectVisibility,
                    request_slots=["prompt"],
                    response_slots=["system_message"],
                    response_cb=object_visible_response_cb,
                    outcomes=["object_not_visible"],
                ),
                remapping={
                    "prompt": "llm_input",
                },
            )

    def child_termination_cb(self, outcome_map):
        # Terminate early when either system returns a failure message
        if (
            outcome_map["ACTION_PLANNER"] == "system_out"
            or outcome_map["CHECK_OBJECT_VISIBILITY"] == "object_not_visible"
        ):
            return True
        return False

    def outcome_cb(self, outcome_map):
        # Terminate early when either system returns a failure message
        if outcome_map["ACTION_PLANNER"] == "system_out":
            self.get_children()["CHECK_OBJECT_VISIBILITY"].recall_preempt()
            return "system_out"
        elif outcome_map["CHECK_OBJECT_VISIBILITY"] == "object_not_visible":
            self.get_children()["ACTION_PLANNER"].recall_preempt()
            return "system_out"
        elif (
            outcome_map["ACTION_PLANNER"] == "succeeded"
            and outcome_map["CHECK_OBJECT_VISIBILITY"] == "succeeded"
        ):
            return "succeeded"


class PushActionSuccess(smach.StateMachine):
    def __init__(
        self,
    ):
        super(PushActionSuccess, self).__init__(
            input_keys=[
                "action",
                "action_type",
                "target_object",
                "target_origin_x",
                "target_origin_y",
            ],
            output_keys=["system_message"],
            outcomes=["succeeded", "preempted", "aborted", "system_out"],
        )
        # Open the container
        with self:
            # check if push action
            smach.StateMachine.add(
                "IS_PUSH_ACTION",
                smach_ros.ConditionState(
                    cond_cb=lambda ud: ud.action == "act" and "push" in ud.action_type,
                    input_keys=["action", "action_type"],
                ),
                {"true": "OBJECT_DETECTION", "false": "succeeded"},
            )
            # detect object in image space
            smach.StateMachine.add(
                "OBJECT_DETECTION",
                smach_ros.ServiceState(
                    "object_detector",
                    DetectObjects,
                    request_slots=["texts"],
                    response_slots=["objects"],
                ),
                transitions={
                    "succeeded": "CHOOSE_TARGET_OBJECT",
                },
                remapping={
                    "texts": "target_object",
                },
            )
            # process dected objects
            smach.StateMachine.add(
                "CHOOSE_TARGET_OBJECT",
                ObjectSelector(),
                transitions={
                    "succeeded": "COORDINATE_TRANSFER",
                    "object_not_found": "system_out",  # TODO combine not found and out of reach?
                    "object_out_of_reach": "system_out",
                },
            )
            # image to real
            smach.StateMachine.add(
                "COORDINATE_TRANSFER",
                smach_ros.ServiceState(
                    "image_to_real",
                    CoordinateTransfer,
                    request_slots=["image_x", "image_y"],
                    response_slots=["real_x", "real_y"],
                ),
                transitions={"succeeded": "DETERMINE_PUSH_DISTANCE"},
            )

            @smach.cb_interface(
                input_keys=[
                    "action_type",
                    "target_origin_x",
                    "target_origin_y",
                    "real_x",
                    "real_y",
                    "target_object",
                ],
                output_keys=[],
                outcomes=["succeeded"],
            )
            def push_distance_cb(userdata):
                if userdata.action_type == "push":
                    direction = "forward"
                    push_distance = userdata.real_x - userdata.target_origin_x
                elif userdata.action_type == "push_left":
                    direction = "left"
                    push_distance = userdata.real_y - userdata.target_origin_y
                elif userdata.action_type == "push_right":
                    direction = "right"
                    push_distance = userdata.target_origin_y - userdata.real_y
                rospy.loginfo(f"Object was pushed {direction} by {push_distance}m")
                with open("action_success_log.txt", "a") as f:
                    print(
                        f"{userdata.target_object} was pushed {direction} by {push_distance}m",
                        file=f,
                    )
                    print(f"Success: {push_distance > 0.02}", file=f)
                    print(f"Diff: {np.abs(0.02 - push_distance)}", file=f)
                return "succeeded"

            smach.StateMachine.add(
                "DETERMINE_PUSH_DISTANCE",
                smach.CBState(push_distance_cb),
                {"succeeded": "succeeded"},
            )
