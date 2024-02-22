from os.path import dirname, abspath, join, pardir
import os.path
from image2real import ImageToRealWorldMLP
from coordinate_transfer_net import ImplicitCoordinateTransfer
from image_recorder import ImageRecorder
from robot_controller import RobotController
from object_localiser import OWLv2
import torch
import rospy
from speech_asr_client import SpeechASRClient


def main():
    # Initialize Camera
    camera = ImageRecorder()
    # Initialize Robot
    robot = RobotController()
    # Create the OWLv2 instance
    owlv2 = OWLv2("owlv2")
    # Text and Image input
    # text_query = [
    #    "Touch the sponge"
    # ]  # ["human face", "rocket", "nasa badge", "star-spangled banner"]#

    # Get the ASR ready and listen to human commands
    rospy.init_node("speech_asr_client")

    verbose = rospy.get_param("~verbose", True)
    detect_start = rospy.get_param("~detect_start", True)
    detect_stop = rospy.get_param("~detect_stop", True)
    if (detect := rospy.get_param("~detect", None)) is not None:
        detect_start = detect
        detect_stop = detect
    start_timeout = rospy.get_param("~start_timeout", 0.0)
    min_duration = rospy.get_param("~min_duration", 3.0)
    max_duration = rospy.get_param("~max_duration", 30.0)
    min_period = rospy.get_param("~min_period", 3.0)
    live_text = rospy.get_param("~live_text", True)
    done_timeout = rospy.get_param("~done_timeout", 0.0)

    client = SpeechASRClient(verbose=verbose)
    rospy.loginfo(f"Waiting for action server to exist: {client.ns}")
    client.wait_for_server()

    while True:
        robot.look_down()
        rospy.loginfo("Performing ASR using action server...")
        text_query = client.perform_asr(
            detect_start=detect_start,
            detect_stop=detect_stop,
            start_timeout=start_timeout,
            min_duration=min_duration,
            max_duration=max_duration,
            min_period=min_period,
            live_text=live_text,
            done_timeout=done_timeout,
        )
        rospy.loginfo(f'ASR text: "{text_query}"')
        print("User speech input: ", text_query)

        image_path = camera.record_image()

        # Get the X, Y pixel space target points
        x, y = owlv2(image_path, [text_query], False)
        print("X and Y position of the target object: ", x, y)

        # Use GPU if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        print(
            "The currently selected GPU is number:",
            torch.cuda.current_device(),
            ", it's a ",
            torch.cuda.get_device_name(device=None),
        )
        # # Create an instance of the image-to-real-world MLP
        # image2real_mlp = ImageToRealWorldMLP().to(device)
        # # Load the trained model
        # checkpoint = torch.load(
        #     join(
        #         dirname(abspath(__file__)),
        #         pardir,
        #         "model_checkpoints/image2real_model_22000epochs.tar",
        #     )
        # )  # get the checkpoint
        # image2real_mlp.load_state_dict(
        #     checkpoint["model_state_dict"]
        # )  # load the model state
        # image2real_mlp.eval()
        image2real_implicit = ImplicitCoordinateTransfer(
            join(
                dirname(abspath(__file__)),
                pardir,
                "model_checkpoints/implicit_model_weights.pth",
            ),
            device,
        )
        # Get the target positions in the real world
        # target_positions = image2real_mlp(
        #     torch.unsqueeze(torch.FloatTensor([x, y]), 0).to(device)
        # )
        target_positions = image2real_implicit.derivative_free_optimizer(
            torch.unsqueeze(torch.FloatTensor([x, y]), 0)
        )
        print(
            "Real-world coordinates of the target object: ",
            target_positions[0][0].item(),
            target_positions[0][1].item(),
        )
        robot.touch(target_positions[0][0], target_positions[0][1])
        in_str = input("Press enter to continue, q to quit")
        robot.initial_position()
        if in_str == "q":
            break
    # Cleanup
    robot.close()
    camera.close()


if __name__ == "__main__":
    main()
