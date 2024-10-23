#!/usr/bin/env python3

import json
import rospy
import smach
import smach_ros

from actionlib_msgs.msg import GoalStatus
from elmira.msg import PerformASRAction
from elmira.srv import PromptTextLLM, PromptVisionLLM
from nicomsg.srv import SayText
from nicomsg.msg import empty

from states.move_robot import JointTrajectoryIterator, MoveRobotPart, MoveRobot
from states.action_planner import ConcurrentPlanAndVerify
from states.action_parser import ActionParser


def main():
    rospy.init_node("elmira_state_machine")

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
    sm.userdata.system_message = ""
    sm.userdata.llm_input = ""
    # speech recognition
    sm.userdata.asr_detect_start = rospy.get_param("~detect_start", True)
    sm.userdata.asr_detect_stop = rospy.get_param("~detect_stop", True)
    sm.userdata.asr_start_timeout = rospy.get_param("~start_timeout", 0.0)
    sm.userdata.asr_min_duration = rospy.get_param("~min_duration", 3.0)
    sm.userdata.asr_max_duration = rospy.get_param("~max_duration", 30.0)
    sm.userdata.asr_min_period = rospy.get_param("~min_period", 3.0)
    sm.userdata.asr_live_text = rospy.get_param("~live_text", True)
    # robot motion
    sm.userdata.motion_init_pose = {
        "l_arm": {
            "names": [
                "l_shoulder_z",
                "l_shoulder_y",
                "l_arm_x",
                "l_elbow_y",
                "l_wrist_z",
                "l_wrist_x",
            ],
            "positions": [0.157, 0.0, 1.57, 1.57, 1.39, 0.0],
        },
        "r_arm": {
            "names": [
                "r_shoulder_z",
                "r_shoulder_y",
                "r_arm_x",
                "r_elbow_y",
                "r_wrist_z",
                "r_wrist_x",
            ],
            "positions": [-0.157, 0.0, -1.57, -1.57, -1.39, 0.0],
        },
        "head": {
            "names": [
                "head_z",
                "head_y",
            ],
            "positions": [0.0, 0.0],  # [0.0, 0.8203],
        },
    }
    sm.userdata.motion_safe_names_left = [
        "l_shoulder_z",
        "l_shoulder_y",
        "l_arm_x",
        "l_elbow_y",
        "l_wrist_z",
        "l_wrist_x",
    ]
    sm.userdata.motion_safe_pose_left = [0.157, 0.0, 0.8203, 1.57, 1.39, 0.0]
    sm.userdata.motion_safe_names_right = [
        "r_shoulder_z",
        "r_shoulder_y",
        "r_arm_x",
        "r_elbow_y",
        "r_wrist_z",
        "r_wrist_x",
    ]
    sm.userdata.motion_safe_pose_right = [-0.157, 0.0, -0.8203, -1.57, -1.39, 0.0]
    sm.userdata.motion_look_down_names = ["head_z", "head_y"]
    sm.userdata.motion_look_down_positions = [0.0, 0.8203]
    sm.userdata.table_z = 0.68
    # TTS
    sm.userdata.tts_language = "en"
    sm.userdata.tts_pitch = 0.0
    sm.userdata.tts_speed = 1.0
    sm.userdata.tts_blocking = True

    # Add states
    with sm:
        # move to initial state
        @smach.cb_interface(output_keys=["llm_actions"], outcomes=["initial_pose"])
        def initial_pose_callback(userdata):
            userdata.llm_actions = [{"action": "initial_pose"}]
            return "initial_pose"

        smach.StateMachine.add(
            "INIT",
            smach.CBState(initial_pose_callback),
            {"initial_pose": "LLM_RESPONSE_ITERATOR"},
        )

        # callback to post-process asr result
        def asr_result_callback(userdata, status, result):
            if status == GoalStatus.SUCCEEDED:
                if len(result.text) == 0:
                    rospy.logwarn("Empty speech result")
                    return "empty"
                else:
                    rospy.loginfo(f"USER: {result.text}")
                    userdata.llm_input = f"USER: {result.text}"
                    return "succeeded"

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
                result_cb=asr_result_callback,
                output_keys=["llm_input"],
                outcomes=["empty"],
            ),
            transitions={"succeeded": "LLM_SPEECH_PROCESSOR", "empty": "SPEECH_ASR"},
            remapping={
                "detect_start": "asr_detect_start",
                "detect_stop": "asr_detect_stop",
                "start_timeout": "asr_start_timeout",
                "min_duration": "asr_min_duration",
                "max_duration": "asr_max_duration",
                "min_period": "asr_min_period",
                "live_text": "asr_live_text",
            },
        )

        # callback to post-process detected objects
        def llm_response_callback(userdata, response):
            rospy.loginfo(f"LLM output:\n{response.response}")
            userdata.llm_actions = json.loads(response.response)["actions"]
            return "succeeded"

        smach.StateMachine.add(
            "LLM_SPEECH_PROCESSOR",
            smach_ros.ServiceState(
                "llm_chat",
                PromptTextLLM,
                request_slots=["prompt"],
                response_cb=llm_response_callback,
                output_keys=["llm_actions"],
            ),
            remapping={"prompt": "llm_input"},
            transitions={
                "succeeded": "LLM_RESPONSE_ITERATOR",
            },
        )

        # iterate through llm response actions
        llm_response_it = smach.Iterator(
            outcomes=[
                "succeeded",
                "preempted",
                "aborted",
                "update_actions",
                "system_out",
                "quit",
            ],
            input_keys=[
                "motion_init_pose",
                "motion_look_down_names",
                "motion_look_down_positions",
                "llm_input",
                "llm_actions",
                "table_z",
                "tts_language",
                "tts_pitch",
                "tts_speed",
                "tts_blocking",
                "system_message",
            ],
            it=lambda: range(0, len(sm.userdata.llm_actions)),
            output_keys=["llm_actions", "system_message"],
            it_label="action_index",
            exhausted_outcome="succeeded",
        )
        with llm_response_it:
            execute_actions_sm = smach.StateMachine(
                outcomes=[
                    "succeeded",
                    "preempted",
                    "aborted",
                    "next_action",
                    "update_actions",
                    "system_out",
                    "quit",
                ],
                input_keys=[
                    "action_index",
                    "motion_init_pose",
                    "motion_look_down_names",
                    "motion_look_down_positions",
                    "llm_input",
                    "llm_actions",
                    "table_z",
                    "tts_language",
                    "tts_pitch",
                    "tts_speed",
                    "tts_blocking",
                    "system_message",
                ],
                output_keys=["llm_actions", "system_message"],
            )
            with execute_actions_sm:

                # parse next action
                smach.StateMachine.add(
                    "ACTION_PARSER",
                    ActionParser(),
                    transitions={
                        "speak": "TEXT_TO_SPEECH",
                        "act": "LOOK_DOWN_ACT",
                        "describe": "LOOK_DOWN_DESCRIBE",  # "LLM_SCENE_DESCRIPTION",
                        "quit": "quit",
                        "initial_pose": "JOINT_TRAJECTORY_ITERATOR",
                    },
                )

                # SPEAK ACTION
                smach.StateMachine.add(
                    "TEXT_TO_SPEECH",
                    smach_ros.ServiceState(
                        "nico/text_to_speech/say",
                        SayText,
                        request_slots=[
                            "text",
                            "language",
                            "pitch",
                            "speed",
                            "blocking",
                        ],
                        # response_slots=["duration"],
                    ),
                    remapping={
                        "text": "tts_text",
                        "language": "tts_language",
                        "pitch": "tts_pitch",
                        "speed": "tts_speed",
                        "blocking": "tts_blocking",
                    },
                    transitions={"succeeded": "next_action"},
                )

                # DESCRIBE ACTION
                smach.StateMachine.add(
                    "LOOK_DOWN_DESCRIBE",
                    MoveRobotPart(MOTION_SRV_HEAD, MOTION_SUB_HEAD),
                    remapping={
                        "names": "motion_look_down_names",
                        "positions": "motion_look_down_positions",
                    },
                    transitions={"succeeded": "LLM_SCENE_DESCRIPTION"},
                )

                def llm_scene_description_callback(userdata, response):
                    rospy.loginfo(f"LLM output:\n{response.response}")
                    userdata.llm_actions = json.loads(response.response)["actions"]
                    return "succeeded"

                smach.StateMachine.add(
                    "LLM_SCENE_DESCRIPTION",
                    smach_ros.ServiceState(
                        "llm_vision",
                        PromptVisionLLM,
                        response_cb=llm_scene_description_callback,
                        output_keys=["llm_actions"],
                    ),
                    transitions={
                        "succeeded": "update_actions",
                    },
                )

                # ACT ACTION
                smach.StateMachine.add(
                    "LOOK_DOWN_ACT",
                    MoveRobotPart(MOTION_SRV_HEAD, MOTION_SUB_HEAD),
                    remapping={
                        "names": "motion_look_down_names",
                        "positions": "motion_look_down_positions",
                    },
                    transitions={"succeeded": "PLAN_ACTION_TRAJECTORY"},
                )
                # plan action and verify if object is actually on the table
                smach.StateMachine.add(
                    "PLAN_ACTION_TRAJECTORY",
                    ConcurrentPlanAndVerify(),
                    {
                        "succeeded": "JOINT_TRAJECTORY_ITERATOR",
                        "system_out": "system_out",
                    },
                )
                # execute movement # TODO movement seperately/concurrently?
                smach.StateMachine.add(
                    "JOINT_TRAJECTORY_ITERATOR",
                    JointTrajectoryIterator(
                        MOTION_SRV_HEAD,
                        MOTION_SUB_HEAD,
                        MOTION_SRV_LEFT,
                        MOTION_SUB_LEFT,
                        MOTION_SRV_RIGHT,
                        MOTION_SUB_RIGHT,
                    ),
                    {"succeeded": "next_action"},
                )

            # close execute_actions_sm
            smach.Iterator.set_contained_state(
                "EXECUTE_ACTIONS", execute_actions_sm, loop_outcomes=["next_action"]
            )
        # close the llm_response_it
        smach.StateMachine.add(
            "LLM_RESPONSE_ITERATOR",
            llm_response_it,
            {
                "succeeded": "SPEECH_ASR",
                "aborted": "aborted",
                "update_actions": "LLM_RESPONSE_ITERATOR",
                "system_out": "LLM_SPEECH_PROCESSOR",
                "quit": "MOVE_TO_SHUTDOWN_POSITION",
            },
            remapping={"system_message": "llm_input"},
        )
        # move robot to safe position and shut down
        smach.StateMachine.add(
            "MOVE_TO_SHUTDOWN_POSITION",
            MoveRobot(
                MOTION_SRV_HEAD,
                MOTION_SUB_HEAD,
                MOTION_SRV_LEFT,
                MOTION_SUB_LEFT,
                MOTION_SRV_RIGHT,
                MOTION_SUB_RIGHT,
            ),
            {"movement_done": "DISABLE_TORQUE"},
            remapping={
                "names_head": "motion_look_down_names",
                "positions_head": "motion_look_down_positions",
                "names_left": "motion_safe_names_left",
                "positions_left": "motion_safe_pose_left",
                "names_right": "motion_safe_names_right",
                "positions_right": "motion_safe_pose_right",
            },
        )

        # disable torque
        @smach.cb_interface(outcomes=["quit"])
        def shutdown_callback(userdata):
            pub = rospy.Publisher("/nico/motion/disableTorqueAll", empty, queue_size=1)
            # wait until subscribers are connected to publish
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                connections = pub.get_num_connections()
                if connections > 0:
                    pub.publish(empty())
                    break
                rate.sleep()
            return "quit"

        smach.StateMachine.add(
            "DISABLE_TORQUE",
            smach.CBState(shutdown_callback),
            {"quit": "succeeded"},
        )

    # Create and start the introspection server for visualization
    sis = smach_ros.IntrospectionServer("elmira_introspection", sm, "/ELMiRA")
    sis.start()
    # Execute state machine
    sm.execute()
    sis.stop()


if __name__ == "__main__":
    main()
