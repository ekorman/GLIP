import requests
from io import BytesIO
from PIL import Image
import numpy as np
from glip.config import cfg
from glip.engine.predictor_glip import GLIPDemo

# top_predictions.bbox: tensor([[337.5107, 205.8404, 600.7050, 364.8850],
#         [317.8773, 289.4856, 639.3996, 477.6625],
#         [367.7430, 306.4413, 398.3886, 316.4792],
#         [188.3125,  19.4669, 200.0233,  47.5475]])
# top_predictions.get_field('scores'): tensor([0.7998, 0.7174, 0.5777, 0.5554])
# top_predictions.get_field('labels'): tensor([2, 2, 3, 1])


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


# Use this command for evaluate the GLPT-T model
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=0.7)
glip_demo.color = 255
image = load("http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg")
class_labels = ["person", "sofa", "remote"]
result, _ = glip_demo.run_on_web_image(image, class_labels, 0.5)
Image.fromarray(result[:, :, [2, 1, 0]]).save("result.png")

response = requests.get(
    "http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg"
)
img = Image.open(BytesIO(response.content)).convert("RGB")
top_predictions = glip_demo(img)
print(f"call top_predictions.bbox: {top_predictions.bbox}")
print(
    f"call top_predictions.get_field('scores'): {top_predictions.get_field('scores')}"
)
print(
    f"call top_predictions.get_field('labels'): {top_predictions.get_field('labels')}"
)
