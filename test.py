import requests
from io import BytesIO
from PIL import Image
from glip.config import cfg
from glip.engine.predictor_glip import GLIP

# top_predictions.bbox: tensor([[337.5107, 205.8404, 600.7050, 364.8850],
#         [317.8773, 289.4856, 639.3996, 477.6625],
#         [367.7430, 306.4413, 398.3886, 316.4792],
#         [188.3125,  19.4669, 200.0233,  47.5475]])
# top_predictions.get_field('scores'): tensor([0.7998, 0.7174, 0.5777, 0.5554])
# top_predictions.get_field('labels'): tensor([2, 2, 3, 1])


# Use this command for evaluate the GLPT-T model
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
from yacs.config import CfgNode as CN


cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

import pdb

pdb.set_trace()

cfg = CN.load_cfg(cfg.dump())

glip_demo = GLIP(cfg, min_image_size=800, confidence_threshold=0.7)
glip_demo.color = 255

response = requests.get(
    "http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg"
)
img = Image.open(BytesIO(response.content)).convert("RGB")
class_labels = ["person", "sofa", "remote"]
top_predictions = glip_demo(img, class_labels, 0.5)
print(f"call top_predictions.bbox: {top_predictions.bbox}")
print(
    f"call top_predictions.get_field('scores'): {top_predictions.get_field('scores')}"
)
print(
    f"call top_predictions.get_field('labels'): {top_predictions.get_field('labels')}"
)
