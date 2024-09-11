from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)
import cv2
import skimage
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin


class OWLv2(nn.Module):
    def __init__(self, owl_version="owlv2", score_threshold=0.1):
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

    def forward(self, image_path, text_query, bounding_box=False):
        # set the model in evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # open and prepare the image
        image = Image.open(image_path)
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
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
        cx, cy, w, h = boxes[np.argmax(scores)]
        if bounding_box:
            image_size = self.model.config.vision_config.image_size
            image = self.mixin.resize(image, image_size)
            input_image = np.asarray(image).astype(np.float32) / 255.0
            self.plot_predictions(input_image, text_query, scores, boxes, labels)
        return cx, cy + h / 2

    def plot_predictions(self, input_image, text_queries, scores, boxes, labels):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

        for score, box, label in zip(scores, boxes, labels):
            if score < self.score_threshold:
                continue

            cx, cy, w, h = box
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                "r",
            )
            ax.text(
                cx - w / 2,
                cy + h / 2 + 0.015,
                f"{text_queries[label]}: {score:1.2f}",
                ha="left",
                va="top",
                color="red",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "red",
                    "boxstyle": "square,pad=.3",
                },
            )
        plt.show()


if __name__ == "__main__":
    owlv2 = OWLv2("owlv2")
    text_query = [
        "cube",
        "cup",
    ]  # ["human face", "rocket", "nasa badge", "star-spangled banner"]#
    image_path = "/informatik3/wtm/home/oezdemir/Downloads/nico_examples_higher/picture-2024-01-31T14-51-26.738369.png"  # target_RLBench_three_buttons_120_vars/image_train/230510/target008008/0.png'
    x, y = owlv2(image_path, text_query, True)
    print("X and Y position of the target object: ", x, y)
