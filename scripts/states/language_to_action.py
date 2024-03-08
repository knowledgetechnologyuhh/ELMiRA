import rospy
import smach

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class FlanT5ActionExtractor(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=[
                "action_detected"
            ],  # TODO alternative outcomes (i.e. robot action, repeat asr, end demo)
            input_keys=["text_query"],
            output_keys=["action", "target_object"],
        )
        # Get the LLM for action and target object word distinction
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.llm_tokeniser = AutoTokenizer.from_pretrained("google/flan-t5-large")

    def execute(self, userdata):
        rospy.loginfo(f"Text query: {userdata.text_query}")
        # Input the speech text to the LLM to extract action and object words
        inputs_action = self.llm_tokeniser(
            # "Find the action in the following statement: " + text_query,
            "Categorize the action in the following statement into 'touch', 'push' or 'show': "
            + userdata.text_query,
            return_tensors="pt",
        )
        inputs_object = self.llm_tokeniser(
            "Find the target object in the following statement: " + userdata.text_query,
            return_tensors="pt",
        )
        outputs_action = self.llm.generate(**inputs_action)
        outputs_object = self.llm.generate(**inputs_object)

        action = self.llm_tokeniser.batch_decode(
            outputs_action, skip_special_tokens=True
        )[0]
        target_object = self.llm_tokeniser.batch_decode(
            outputs_object, skip_special_tokens=True
        )
        rospy.loginfo(f"Action: {action}")
        rospy.loginfo(f"Target: {target_object}")
        userdata.action = action
        userdata.target_object = target_object
        return "action_detected"
