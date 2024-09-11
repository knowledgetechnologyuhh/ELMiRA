import rospy
import smach


class ActionParser(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["initial_pose", "speak", "act", "describe", "quit"],
            input_keys=["llm_actions", "action_index", "motion_init_pose"],
            output_keys=[
                "action",
                "tts_text",
                "action_type",
                "target_object",
                "llm_input",
                "joint_trajectory",
            ],
        )

    def execute(self, userdata):
        next_action = userdata.llm_actions[userdata.action_index]
        rospy.loginfo(f"Action: {next_action['action']}")
        userdata.action = next_action["action"]
        if next_action["action"] == "speak":
            rospy.loginfo(f"Text: {next_action['text']}")
            userdata.tts_text = next_action["text"]
            return "speak"
        elif next_action["action"] == "act":
            rospy.loginfo(
                f"Type: {next_action['type']}, Object: {next_action['object']}"
            )
            userdata.llm_input = f"{next_action['object']}"
            userdata.action_type = next_action["type"]
            userdata.target_object = [next_action["object"]]
            return "act"
        elif next_action["action"] == "describe":
            return "describe"
        elif next_action["action"] == "quit":
            return "quit"
        elif next_action["action"] == "initial_pose":
            userdata.joint_trajectory = [userdata.motion_init_pose]
            return "initial_pose"
