import requests
from io import BytesIO
from PIL import Image
from glip.engine.predictor_glip import GLIP

import torch
from glip.config import cfg

cfg.MODEL.META_ARCHITECTURE = "GeneralizedVLRCNN"
cfg.MODEL.RPN_ONLY = True
cfg.MODEL.RPN_ARCHITECTURE = "VLDYHEAD"

cfg.MODEL.BACKBONE.CONV_BODY = "SWINT-FPN-RETINANET"
cfg.MODEL.BACKBONE.OUT_CHANNELS = 256
cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = -1

cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"  # "roberta-base", "clip"
cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False

cfg.MODEL.RPN.USE_FPN = True
cfg.MODEL.RPN.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
cfg.MODEL.RPN.ANCHOR_STRIDE = (8, 16, 32, 64, 128)
cfg.MODEL.RPN.ASPECT_RATIOS = (1.0,)
cfg.MODEL.RPN.SCALES_PER_OCTAVE = 1

cfg.MODEL.DYHEAD.CHANNELS = 256
cfg.MODEL.DYHEAD.NUM_CONVS = 6
cfg.MODEL.DYHEAD.USE_GN = True
cfg.MODEL.DYHEAD.USE_DYRELU = True
cfg.MODEL.DYHEAD.USE_DFCONV = True
cfg.MODEL.DYHEAD.USE_DYFUSE = True
cfg.MODEL.DYHEAD.TOPK = (
    9  # topk for selecting candidate positive samples from each level
)
cfg.MODEL.DYHEAD.SCORE_AGG = "MEAN"
cfg.MODEL.DYHEAD.LOG_SCALE = 0.0

cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE = "MHA-B"
cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS = False
cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS = False
cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS = False
cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM = 64
cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = True
cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = True

cfg.MODEL.DYHEAD.USE_CHECKPOINT = True

cfg.INPUT.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.INPUT.PIXEL_STD = [57.375, 57.120, 58.395]

cfg.DATALOADER.SIZE_DIVISIBILITY = 32


# top_predictions.bbox: tensor([[337.5107, 205.8404, 600.7050, 364.8850],
#         [317.8773, 289.4856, 639.3996, 477.6625],
#         [367.7430, 306.4413, 398.3886, 316.4792],
#         [188.3125,  19.4669, 200.0233,  47.5475]])
# top_predictions.get_field('scores'): tensor([0.7998, 0.7174, 0.5777, 0.5554])
# top_predictions.get_field('labels'): tensor([2, 2, 3, 1])


# Use this command for evaluate the GLPT-T model
config_file = "/code/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"


glip_demo = GLIP(
    cfg,
    device=torch.device("cuda"),
    model_weight_path=weight_file,
    min_image_size=800,
    confidence_threshold=0.7,
)

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
