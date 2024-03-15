#!/usr/bin/env python3

import rospy
import smach
import smach_ros

from nico_demo.msg import PerformASRAction
from nico_demo.srv import DetectObjects, CoordinateTransfer
from states.language_to_action import FlanT5ActionExtractor
from states.move_robot import MoveRobot, MoveRobotPart


def main():
    rospy.init_node("nico_demo_state_machine")

    # Create a state machine
    sm = smach.StateMachine(outcomes=["succeeded", "aborted", "preempted"])

    # set topic names(TODO change into proper NICO paths TODO turn into rospy param?)
    MOTION_SUB_LEFT = "/left/open_manipulator_p/joint_states"
    MOTION_SRV_LEFT = "/left/open_manipulator_p/goal_joint_space_path"
    MOTION_SUB_RIGHT = "/right/open_manipulator_p/joint_states"
    MOTION_SRV_RIGHT = "/right/open_manipulator_p/goal_joint_space_path"
    MOTION_SUB_HEAD = "/NICOL/joint_states"
    MOTION_SRV_HEAD = "/NICOL/head/goal_joint_space_path"
    # set initial userdata
    # speech recognition
    sm.userdata.asr_detect_start = rospy.get_param("~detect_start", True)
    sm.userdata.asr_detect_stop = rospy.get_param("~detect_stop", True)
    sm.userdata.asr_start_timeout = rospy.get_param("~start_timeout", 0.0)
    sm.userdata.asr_min_duration = rospy.get_param("~min_duration", 3.0)
    sm.userdata.asr_max_duration = rospy.get_param("~max_duration", 30.0)
    sm.userdata.asr_min_period = rospy.get_param("~min_period", 3.0)
    sm.userdata.asr_live_text = rospy.get_param("~live_text", True)
    # robot motion
    sm.userdata.motion_joints_left = [
        "l_shoulder_z",
        "l_shoulder_y",
        "l_arm_x",
        "l_elbow_y",
        "l_wrist_z",
        "l_wrist_x",
    ]
    sm.userdata.motion_init_left = [0.157, 0.0, 1.57, 1.57, 1.39, 0.0]
    sm.userdata.motion_joints_right = [
        "r_shoulder_z",
        "r_shoulder_y",
        "r_arm_x",
        "r_elbow_y",
        "r_wrist_z",
        "r_wrist_x",
    ]
    sm.userdata.motion_init_right = [-0.157, 0.0, -1.57, -1.57, -1.39, 0.0]
    sm.userdata.motion_joints_head = [
        "head_z",
        "head_y",
    ]
    sm.userdata.motion_init_head = [0.0, 0.8203]

    # Add states
    with sm:
        # TODO move to initial state
        smach.StateMachine.add(
            "INITIAL_ROBOT_POSE",
            MoveRobot(
                MOTION_SRV_HEAD,
                MOTION_SUB_HEAD,
                MOTION_SRV_LEFT,
                MOTION_SUB_LEFT,
                MOTION_SRV_RIGHT,
                MOTION_SUB_RIGHT,
            ),
            transitions={"movement_done": "SPEECH_ASR"},
            remapping={
                "names_head": "motion_joints_head",
                "positions_head": "motion_init_head",
                "names_left": "motion_joints_left",
                "positions_left": "motion_init_left",
                "names_right": "motion_joints_right",
                "positions_right": "motion_init_right",
            },
        )
        # listen for human command via speech asr ros action
        smach.StateMachine.add(
            "SPEECH_ASR",
            smach_ros.SimpleActionState(
                "speech_asr",
                PerformASRAction,
                goal_slots=[
                    "detect_start",
                    "detect_stop",
                    "start_timeout",
                    "min_duration",
                    "max_duration",
                    "min_period",
                    "live_text",
                ],
                result_slots=["text"],
            ),
            transitions={"succeeded": "LLM_SPEECH_PROCESSOR"},
            remapping={
                "detect_start": "asr_detect_start",
                "detect_stop": "asr_detect_stop",
                "start_timeout": "asr_start_timeout",
                "min_duration": "asr_min_duration",
                "max_duration": "asr_max_duration",
                "min_period": "asr_min_period",
                "live_text": "asr_live_text",
                "text": "asr_result_text",
            },
        )
        # extract action and target object from detected speech with LLM
        # TODO turn into ros service?
        # TODO use Vicuna or GPT
        smach.StateMachine.add(
            "LLM_SPEECH_PROCESSOR",
            FlanT5ActionExtractor(),
            transitions={
                "action_detected": "OBJECT_DETECTION"
            },  # TODO more transitions (i.e. robot action, repeat asr, demo end?)
            remapping={"text_query": "asr_result_text"},
        )

        # callback to post-process detected objects
        def object_response_callback(userdata, response):
            if len(response.objects) == 0:
                return "object_not_found"
            userdata.image_x = response.objects[0].center_x
            userdata.image_y = (
                response.objects[0].center_y + response.objects[0].height / 2
            )
            return "succeeded"

        # detect object in image space
        smach.StateMachine.add(
            "OBJECT_DETECTION",
            smach_ros.ServiceState(
                "object_detector",
                DetectObjects,
                request_slots=["texts"],
                response_cb=object_response_callback,
                output_keys=["image_x", "image_y"],
                outcomes=["object_not_found"],
            ),
            transitions={
                "succeeded": "COORDINATE_TRANSFER",
                "object_not_found": "SPEECH_ASR",
            },
            remapping={
                "texts": "target_object",
            },
        )
        # TODO process bounding boxes
        # image to real
        smach.StateMachine.add(
            "COORDINATE_TRANSFER",
            smach_ros.ServiceState(
                "image_to_real",
                CoordinateTransfer,
                request_slots=["image_x", "image_y"],
                response_slots=["real_x", "real_y"],
            ),
            transitions={"succeeded": "succeeded"},
        )

        # TODO ros service?
        # TODO Perform action
        # TODO one state per action? maybe in sub-statemachine?

    # Create and start the introspection server for visualization
    sis = smach_ros.IntrospectionServer("nico_demo_introspection", sm, "/NICO_DEMO")
    sis.start()
    # Execute state machine
    outcome = sm.execute()
    if outcome == "succeeded":
        rospy.loginfo(f"real x: {sm.userdata.real_x}")
        rospy.loginfo(f"real y: {sm.userdata.real_y}")
    sis.stop()


if __name__ == "__main__":
    main()
