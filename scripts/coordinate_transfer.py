#!/usr/bin/env python3

from os.path import dirname, abspath, join, pardir
import rospy
import torch

from coordinate_transfer_net import ImplicitCoordinateTransfer
from elmira.srv import CoordinateTransfer


class ImplicitCoordinateTransferServer:
    def __init__(
        self,
    ):
        rospy.init_node("implicit_transfer_server")
        # use GPU if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using {device} device")
        # init implicit model
        self.implicit_model = ImplicitCoordinateTransfer(
            join(
                dirname(abspath(__file__)),
                pardir,
                "model_checkpoints/implicit_model_weights.pth",
            ),
            device,
        )
        # launch service
        rospy.Service(
            "image_to_real", CoordinateTransfer, self.image_to_real_request_handler
        )
        rospy.loginfo("ImplicitCoordinateTransfer started successfully")
        rospy.spin()

    def image_to_real_request_handler(self, request):
        # get image
        real_x, real_y = (
            self.implicit_model.derivative_free_optimizer(
                torch.unsqueeze(
                    torch.FloatTensor([request.image_x, request.image_y]), 0
                )
            )
            .squeeze()
            .cpu()
        )
        rospy.loginfo(f"Real coordinates: x={real_x}, y={real_y}")
        return real_x, real_y


if __name__ == "__main__":
    ImplicitCoordinateTransferServer()
