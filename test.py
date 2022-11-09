import requests
from io import BytesIO
from PIL import Image
from glip.engine.predictor_glip import GLIP, make_glip_t_cfg

import torch


# top_predictions.bbox: tensor([[337.5107, 205.8404, 600.7050, 364.8850],
#         [317.8773, 289.4856, 639.3996, 477.6625],
#         [367.7430, 306.4413, 398.3886, 316.4792],
#         [188.3125,  19.4669, 200.0233,  47.5475]])
# top_predictions.get_field('scores'): tensor([0.7998, 0.7174, 0.5777, 0.5554])
# top_predictions.get_field('labels'): tensor([2, 2, 3, 1])


# Use this command for evaluate the GLPT-T model
weight_file = "/MODEL/glip_tiny_model_o365_goldg_cc_sbu_model_only.pth"  # "MODEL/glip_large_model_model_only.pth"


cfg = make_glip_t_cfg()  # make_glip_l_cfg

glip_demo = GLIP(
    cfg,
    device=torch.device("cuda"),
    model_weight_path=weight_file,
    min_image_size=800,
)

response = requests.get(
    "http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg"
)
img = Image.open(BytesIO(response.content)).convert("RGB")
class_labels = ["person", "sofa", "remote"]
top_predictions = glip_demo([img], class_labels, 0.5)

for i, top_prediction in enumerate(top_predictions):
    print(f"call top_predictions[{i}].bbox: {top_prediction.bbox}")
    print(
        f"call top_predictions[{i}].get_field('scores'): {top_prediction.get_field('scores')}"
    )
    print(
        f"call top_predictions[{i}].get_field('labels'): {top_prediction.get_field('labels')}"
    )
