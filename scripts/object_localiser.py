#!/usr/bin/env python3

from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from transformers.image_utils import ImageFeatureExtractionMixin

import cv_bridge
import rospy
import sensor_msgs.msg

from elmira.msg import DetectedObject
from elmira.srv import DetectObjects, DetectObjectsResponse


class OWLv2(nn.Module):
    def __init__(self, owl_version="owlv2", score_threshold=0.05):
        super(OWLv2, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if owl_version == "owl-vit":
            self.model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32"
            )
            self.processor = OwlViTProcessor.from_pretrained(
                "google/owlvit-base-patch32"
            )
        else:
            self.processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
            self.model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
        self.mixin = ImageFeatureExtractionMixin()
        self.score_threshold = score_threshold

    def forward(self, image, text_query):
        # set the model in evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # open and prepare the image
        if image.height != image.width:
            image = self.mixin.resize(image, min(image.height, image.width))
        inputs = self.processor(text=text_query, images=image, return_tensors="pt").to(
            self.device
        )

        # get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values)
        valid_indices = torch.where(scores >= self.score_threshold)

        # get prediction labels and boundary boxes
        scores = scores[valid_indices].cpu().detach().numpy()
        labels = logits.indices[valid_indices].cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0][valid_indices].cpu().detach().numpy()
        # TODO only best for each label?
        # cx, cy, w, h = boxes[np.argmax(scores)]
        # if bounding_box:
        #     image_size = self.model.config.vision_config.image_size
        #     image = self.mixin.resize(image, image_size)
        #     input_image = np.asarray(image).astype(np.float32) / 255.0
        #     self.plot_predictions(input_image, text_query, scores, boxes, labels)
        return scores, boxes, labels

    def plot_predictions(self, image, text_queries, scores, boxes, labels):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial", 24)
        label_offset = 5
        label_padding = 5

        for score, box, label in zip(scores, boxes, labels):
            cx, cy, w, h = box
            cx *= image.width
            cy *= image.height
            w *= image.width
            h *= image.height
            # draw object bounding box
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], width=3)
            # create text with score and label
            label_text = f"{text_queries[label]}: {score:1.2f}"
            # determine bbox of label text
            textbbox = draw.textbbox(
                [cx - w / 2 + label_padding, cy + h / 2 + label_offset + label_padding],
                label_text,
                font=font,
                anchor="lt",
            )
            # draw textbbox
            draw.rectangle(
                [
                    textbbox[0] - label_padding,
                    textbbox[1] - label_padding,
                    textbbox[2] + label_padding,
                    textbbox[3] + label_padding,
                ],
                fill="white",
                outline="red",
                width=3,
            )
            # add label text
            draw.text(
                [cx - w / 2 + label_padding, cy + h / 2 + label_offset + label_padding],
                label_text,
                fill="red",
                font=font,
                anchor="lt",
            )
        return image


class OWLv2Server:
    def __init__(self):
        rospy.init_node("owlv2_server")
        self.owlv2 = OWLv2("owlv2")
        self.bridge = cv_bridge.CvBridge()
        rospy.Service("object_detector", DetectObjects, self.detection_request_handler)
        # TODO make optional?
        self.debug_pub = rospy.Publisher(
            "owlv2_server/result_image",
            sensor_msgs.msg.Image,
            latch=True,
            queue_size=1,
        )
        rospy.loginfo("OWLv2 started successfully")
        rospy.spin()

    def detection_request_handler(self, request):
        # get latest camera image
        img_msg = rospy.wait_for_message(
            "/nico/vision/right",
            sensor_msgs.msg.Image,
        )
        # convert message to PIL image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        image = Image.fromarray(cv_image)
        # detect objects
        scores, boxes, labels = self.owlv2(image, request.texts)
        # visualize detection
        # TODO optional?
        image = self.owlv2.plot_predictions(image, request.texts, scores, boxes, labels)
        debug_img_msg = self.bridge.cv2_to_imgmsg(np.array(image), "rgb8")
        debug_img_msg.header.stamp = rospy.Time.now()
        self.debug_pub.publish(debug_img_msg)
        # return response with detected objects
        return DetectObjectsResponse(
            [
                DetectedObject(request.texts[labels[i]], scores[i], *boxes[i])
                for i in range(len(labels))
            ]
        )


if __name__ == "__main__":
    OWLv2Server()
