from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
import cv2
import skimage
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin

def plot_predictions(input_image, text_queries, scores, boxes, labels, score_threshold):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue

        cx, cy, w, h = box
        ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], "r")
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
                "boxstyle": "square,pad=.3"
            })
    plt.show()

def owlvit_example(images_path, text_queries, owl_version='owl-vit'):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if owl_version == 'owl-vit':
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    else:
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    mixin = ImageFeatureExtractionMixin()
    images = []
    # Download sample image
    #image = skimage.data.astronaut()
    #image = Image.fromarray(np.uint8(image)).convert("RGB")
    for (root, dirs, files) in os.walk(images_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(root+'/'+file)
                if image.height != image.width:
                    image = mixin.resize(image, min(image.height, image.width))
                images.append(image)
    #image = Image.open(image_path)

    #if image.height != image.width:
    #   image = mixin.resize(image, min(image.height, image.width))
    # Process image and text inputs
    inputs = processor(text=text_queries, images=images[10], return_tensors="pt").to(device)

    # Print input names and shapes
    print("\nInput names and shapes")
    for key, val in inputs.items():
        print(f"{key}: {val.shape}")


    # Set model in evaluation mode
    model = model.to(device)
    model.eval()

    # Get predictions
    with torch.no_grad():
      outputs = model(**inputs)
    print("\nMiscellaneous model outputs")
    for k, val in outputs.items():
        if k not in {"text_model_output", "vision_model_output"}:
            print(f"{k}: shape of {val.shape}")

    print("\nText model outputs")
    for k, val in outputs.text_model_output.items():
        print(f"{k}: shape of {val.shape}")

    print("\nVision model outputs")
    for k, val in outputs.vision_model_output.items():
        print(f"{k}: shape of {val.shape}")


    #mixin = ImageFeatureExtractionMixin()

    # Load example image
    image_size = model.config.vision_config.image_size
    image = mixin.resize(images[10], image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    score_threshold = 0.1

    # Get prediction logits
    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
    plot_predictions(input_image, text_queries, scores, boxes, labels, score_threshold)

def visualbert_example(image_path, text_queries):
    from transformers import BertTokenizer, VisualBertModel

    model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(text_queries, return_tensors="pt")
    # this is a custom function that returns the visual embeddings given the image path
    visual_embeds = get_visual_embeddings(image_path)

    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    inputs.update(
        {
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }
    )
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state


# Text queries to search the image for
text_queries = ["can you show me the cup?"]#["human face", "rocket", "nasa badge", "star-spangled banner"]#
images_path = '/informatik3/wtm/home/oezdemir/Downloads/nico_examples'#target_RLBench_three_buttons_120_vars/image_train/230510/target008008/0.png'
owlvit_example(images_path, text_queries, 'owlv2')
#visualbert_example(images_path, text_queries)
#https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb