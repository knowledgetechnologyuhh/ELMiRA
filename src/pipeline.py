from os.path import dirname, abspath, join, pardir

from image2real import ImageToRealWorldMLP
from image_recorder import ImageRecorder
from robot_controller import RobotController
from object_localiser import OWLv2
import torch


def main():
    # Initialize Camera
    camera = ImageRecorder()
    # Initialize Robot
    robot = RobotController()
    # Create the OWLv2 instance
    owlv2 = OWLv2("owlv2")
    # Text and Image input
    text_query = [
        "Touch the sponge"
    ]  # ["human face", "rocket", "nasa badge", "star-spangled banner"]#
    robot.look_down()
    image_path = camera.record_image()
    # Get the X, Y pixel space target points
    x, y = owlv2(image_path, text_query, True)
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
    # Create an instance of the image-to-real-world MLP
    image2real_mlp = ImageToRealWorldMLP().to(device)
    # Load the trained model
    checkpoint = torch.load(
        join(
            dirname(abspath(__file__)),
            pardir,
            "model_checkpoints/image2real_model_22000epochs.tar",
        )
    )  # get the checkpoint
    image2real_mlp.load_state_dict(
        checkpoint["model_state_dict"]
    )  # load the model state
    image2real_mlp.eval()
    # Get the target positions in the real world
    target_positions = image2real_mlp(
        torch.unsqueeze(torch.FloatTensor([x, y]), 0).to(device)
    )
    print(
        "Real-world coordinates of the target object: ",
        target_positions[0][0].item(),
        target_positions[0][1].item(),
    )
    robot.touch(target_positions[0][0], target_positions[0][1])
    input("Press enter to continue")
    # Cleanup
    robot.initial_position()
    robot.close()
    camera.close()


if __name__ == "__main__":
    main()
