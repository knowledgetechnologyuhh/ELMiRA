#!/usr/bin/env python
import numpy as np

import smach
from smach_ros import ServiceState, MonitorState
from open_manipulator_msgs.srv import SetJointPosition
from open_manipulator_msgs.msg import JointPosition
from sensor_msgs.msg import JointState


class JointTrajectoryIterator(smach.Iterator):
    """Executes a sequence of joint movements."""

    def __init__(
        self,
        srv_topic_head,
        sub_topic_head,
        srv_topic_left,
        sub_topic_left,
        srv_topic_right,
        sub_topic_right,
    ):
        super(JointTrajectoryIterator, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=[
                "joint_trajectory",
            ],
            output_keys=[],
            it=lambda: range(0, len(self.userdata.joint_trajectory)),
            it_label="trajectory_step",
            exhausted_outcome="succeeded",
        )
        with self:
            trajectory_sm = smach.StateMachine(
                outcomes=["succeeded", "preempted", "aborted", "next_pose"],
                input_keys=["joint_trajectory", "trajectory_step"],
            )
            with trajectory_sm:
                # parse next joint state
                @smach.cb_interface(
                    input_keys=["joint_trajectory", "trajectory_step"],
                    output_keys=[
                        "names_head",
                        "positions_head",
                        "names_left",
                        "positions_left",
                        "names_right",
                        "positions_right",
                    ],
                    outcomes=["succeeded"],
                )
                def trajectory_step_cb(userdata):
                    joint_states = userdata.joint_trajectory[userdata.trajectory_step]
                    if "head" in joint_states:
                        userdata.names_head = joint_states["head"]["names"]
                        userdata.positions_head = joint_states["head"]["positions"]
                    else:
                        userdata.names_head = []
                        userdata.positions_head = []
                    if "l_arm" in joint_states:
                        userdata.names_left = joint_states["l_arm"]["names"]
                        userdata.positions_left = joint_states["l_arm"]["positions"]
                    else:
                        userdata.names_left = []
                        userdata.positions_left = []
                    if "r_arm" in joint_states:
                        userdata.names_right = joint_states["r_arm"]["names"]
                        userdata.positions_right = joint_states["r_arm"]["positions"]
                    else:
                        userdata.names_right = []
                        userdata.positions_right = []
                    return "succeeded"

                smach.StateMachine.add(
                    "SET_JOINT_TARGETS",
                    smach.CBState(trajectory_step_cb),
                    {"succeeded": "MOVE_ROBOT"},
                )
                smach.StateMachine.add(
                    "MOVE_ROBOT",
                    MoveRobot(
                        srv_topic_head,
                        sub_topic_head,
                        srv_topic_left,
                        sub_topic_left,
                        srv_topic_right,
                        sub_topic_right,
                    ),
                    transitions={"movement_done": "next_pose"},
                )

            # close trajectory_sm
            smach.Iterator.set_contained_state(
                "EXECUTE_JOINT_TRAJECTORY", trajectory_sm, loop_outcomes=["next_pose"]
            )


class MoveRobot(smach.Concurrence):
    """Move robot head and arms in parallel."""

    def __init__(
        self,
        srv_topic_head,
        sub_topic_head,
        srv_topic_left,
        sub_topic_left,
        srv_topic_right,
        sub_topic_right,
    ):
        super(MoveRobot, self).__init__(
            input_keys=[
                "names_head",
                "positions_head",
                "names_left",
                "positions_left",
                "names_right",
                "positions_right",
            ],
            outcomes=["movement_done"],
            default_outcome="movement_done",
            outcome_map={
                "movement_done": {
                    "MOVE_HEAD": "succeeded",
                    "MOVE_LEFT_ARM": "succeeded",
                    "MOVE_RIGHT_ARM": "succeeded",
                }
            },
        )
        # Open the container
        with self:
            smach.Concurrence.add(
                "MOVE_HEAD",
                MoveRobotPart(srv_topic_head, sub_topic_head),
                remapping={"names": "names_head", "positions": "positions_head"},
            )
            smach.Concurrence.add(
                "MOVE_LEFT_ARM",
                MoveRobotPart(srv_topic_left, sub_topic_left),
                remapping={"names": "names_left", "positions": "positions_left"},
            )
            smach.Concurrence.add(
                "MOVE_RIGHT_ARM",
                MoveRobotPart(srv_topic_right, sub_topic_right),
                remapping={"names": "names_right", "positions": "positions_right"},
            )


class MoveRobotPart(smach.Sequence):
    """Move robot part and monitor state until success."""

    def __init__(self, srv_topic, sub_topic):
        super(MoveRobotPart, self).__init__(
            input_keys=["names", "positions"],
            outcomes=["succeeded", "aborted", "preempted"],
            connector_outcome="succeeded",
        )
        # Open the container
        with self:

            @smach.cb_interface(input_keys=["names", "positions"])
            def set_joint_position_request_cb(userdata, request):
                joint_position = JointPosition()
                joint_position.joint_name = userdata.names
                joint_position.position = userdata.positions
                request.joint_position = joint_position
                return request

            smach.Sequence.add(
                "START_MOVEMENT",
                ServiceState(
                    srv_topic,
                    SetJointPosition,
                    request_cb=set_joint_position_request_cb,
                    input_keys=["names", "positions"],
                ),
            )

            # TODO precision input parameter?
            @smach.cb_interface(input_keys=["names", "positions"])
            def target_joint_state_reached_cb(userdata, message):
                joint_ids = np.argsort(message.name)
                ordered_state = np.array(message.position)[
                    joint_ids[
                        np.searchsorted(message.name, userdata.names, sorter=joint_ids)
                    ]
                ]
                return not (
                    np.allclose(userdata.positions, ordered_state, atol=0.052)  # < ~3°
                    and np.all(np.array(message.velocity) == 0)
                )

            smach.Sequence.add(
                "WAIT_UNTIL_REACHED",
                MonitorState(
                    sub_topic,
                    JointState,
                    target_joint_state_reached_cb,
                    max_checks=300,  # publisher at 50hz
                    input_keys=["names", "positions"],
                ),
                transitions={"valid": "succeeded", "invalid": "succeeded"},
            )
