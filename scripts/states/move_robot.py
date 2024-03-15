#!/usr/bin/env python
import numpy as np

import rospy
import smach
from smach_ros import ServiceState, MonitorState
from open_manipulator_msgs.srv import SetJointPosition
from open_manipulator_msgs.msg import JointPosition
from sensor_msgs.msg import JointState


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
        super(MoveRobot, self).__init__(  # TODO define proper userdata (poses etc.)
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
        super(MoveRobotPart, self).__init__(  # TODO define proper userdata (poses etc.)
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

            @smach.cb_interface(input_keys=["names", "positions"])  # TODO precision?
            def target_joint_state_reached_cb(userdata, message):
                joint_ids = np.argsort(message.name)
                ordered_state = np.array(message.position)[
                    joint_ids[
                        np.searchsorted(message.name, userdata.names, sorter=joint_ids)
                    ]
                ]
                ordered_names = np.array(message.name)[
                    joint_ids[
                        np.searchsorted(message.name, userdata.names, sorter=joint_ids)
                    ]
                ]
                print(dict(zip(ordered_names, userdata.positions - ordered_state)))
                return not np.allclose(
                    userdata.positions, ordered_state, atol=0.05  # < ~3°
                )  # TODO check speed instead? (needs to be in state then)

            smach.Sequence.add(
                "WAIT_UNTIL_REACHED",
                MonitorState(
                    sub_topic,
                    JointState,
                    target_joint_state_reached_cb,
                    # max_checks = ..., # publisher at 50hz
                    input_keys=["names", "positions"],
                ),
                transitions={"valid": "succeeded", "invalid": "succeeded"},
            )
