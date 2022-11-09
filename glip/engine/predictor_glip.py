from typing import List

from PIL import Image
from transformers import AutoTokenizer
import torch
from torchvision import transforms as T
from yacs.config import CfgNode

from glip.config import cfg
from glip.modeling.detector import build_detection_model
from glip.structures.bounding_box import BoxList
from glip.structures.image_list import to_image_list
from glip.modeling.roi_heads.mask_head.inference import Masker
from glip.utils.model_serialization import load_state_dict


class GLIP(object):
    def __init__(
        self,
        cfg,
        device: torch.device,
        model_weight_path: str,
        min_image_size=None,
    ):
        self.cfg = cfg
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        self.min_image_size = min_image_size

        load_state_dict(self.model, torch.load(model_weight_path))

        self.transforms = self.build_transform()

        # used to make colors for each tokens
        mask_threshold = 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
        self.cpu_device = torch.device("cpu")

        self.tokenizer = self.build_tokenizer()

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """

        cfg = self.cfg

        to_bgr_transform = T.Lambda(lambda x: (x * 255)[[2, 1, 0], :, :])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.Resize(self.min_image_size)
                if self.min_image_size is not None
                else lambda x: x,
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast

            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    from_slow=True,
                    mask_token="ðŁĴĳ</w>",
                )
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained(
                    "openai/clip-vit-base-patch32", from_slow=True
                )
        return tokenizer

    def __call__(
        self, imgs: List[Image.Image], class_labels, thresh=0.5
    ) -> List[BoxList]:
        predictions = self.compute_predictions(imgs, class_labels)
        top_predictions = [
            self._post_process(prediction, thresh) for prediction in predictions
        ]
        return top_predictions

    def compute_predictions(self, imgs: List[Image.Image], class_labels):
        # image
        img_tensors = [self.transforms(img) for img in imgs]
        image_list = to_image_list(img_tensors, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        self.entities = class_labels
        # we directly provided a list of category names
        caption_string = ""
        tokens_positive = []
        seperation_tokens = " . "
        for word in class_labels:

            tokens_positive.append(
                [[len(caption_string), len(caption_string) + len(word)]]
            )
            caption_string += word
            caption_string += seperation_tokens

        tokenized = self.tokenizer([caption_string], return_tensors="pt")

        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = (
            create_positive_map_label_to_token_from_positive_map(
                positive_map, plus=plus
            )
        )
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token

        # compute predictions
        with torch.no_grad():
            predictions = self.model(
                image_list,
                captions=[caption_string] * len(imgs),
                positive_map=positive_map_label_to_token,
            )
            predictions = [o.to(self.cpu_device) for o in predictions]

        predictions = [
            prediction.resize((img.width, img.height))
            for prediction, img in zip(predictions, imgs)
        ]

        return predictions

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i in range(len(labels)):
            thresh[i] = threshold
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True
        )[0].tolist()
    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def make_glip_t_cfg() -> CfgNode:
    glip_t_cfg = cfg.clone()

    glip_t_cfg.MODEL.META_ARCHITECTURE = "GeneralizedVLRCNN"
    glip_t_cfg.MODEL.RPN_ONLY = True
    glip_t_cfg.MODEL.RPN_ARCHITECTURE = "VLDYHEAD"

    glip_t_cfg.MODEL.BACKBONE.CONV_BODY = "SWINT-FPN-RETINANET"
    glip_t_cfg.MODEL.BACKBONE.OUT_CHANNELS = 256
    glip_t_cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = -1

    glip_t_cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    glip_t_cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    glip_t_cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False

    glip_t_cfg.MODEL.RPN.USE_FPN = True
    glip_t_cfg.MODEL.RPN.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
    glip_t_cfg.MODEL.RPN.ANCHOR_STRIDE = (8, 16, 32, 64, 128)
    glip_t_cfg.MODEL.RPN.ASPECT_RATIOS = (1.0,)
    glip_t_cfg.MODEL.RPN.SCALES_PER_OCTAVE = 1

    glip_t_cfg.MODEL.DYHEAD.CHANNELS = 256
    glip_t_cfg.MODEL.DYHEAD.NUM_CONVS = 6
    glip_t_cfg.MODEL.DYHEAD.USE_GN = True
    glip_t_cfg.MODEL.DYHEAD.USE_DYRELU = True
    glip_t_cfg.MODEL.DYHEAD.USE_DFCONV = True
    glip_t_cfg.MODEL.DYHEAD.USE_DYFUSE = True
    glip_t_cfg.MODEL.DYHEAD.TOPK = (
        9  # topk for selecting candidate positive samples from each level
    )
    glip_t_cfg.MODEL.DYHEAD.SCORE_AGG = "MEAN"
    glip_t_cfg.MODEL.DYHEAD.LOG_SCALE = 0.0

    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE = "MHA-B"
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS = False
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS = False
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS = False
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM = 64
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = True
    glip_t_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = True

    glip_t_cfg.MODEL.DYHEAD.USE_CHECKPOINT = True

    glip_t_cfg.INPUT.PIXEL_MEAN = [103.530, 116.280, 123.675]
    glip_t_cfg.INPUT.PIXEL_STD = [57.375, 57.120, 58.395]

    glip_t_cfg.DATALOADER.SIZE_DIVISIBILITY = 32

    return glip_t_cfg


def make_glip_l_cfg() -> CfgNode:
    glip_l_cfg = cfg.clone()

    glip_l_cfg.MODEL.META_ARCHITECTURE = "GeneralizedVLRCNN"
    # glip_l_cfg.MODEL.WEIGHT = "swin_large_patch4_window12_384_22k.pth"
    glip_l_cfg.MODEL.RPN_ONLY = True
    glip_l_cfg.MODEL.RPN_ARCHITECTURE = "VLDYHEAD"

    glip_l_cfg.MODEL.BACKBONE.CONV_BODY = "SWINT-FPN-RETINANET"
    glip_l_cfg.MODEL.BACKBONE.OUT_CHANNELS = 256

    glip_l_cfg.MODEL.SWINT.EMBED_DIM = 192
    glip_l_cfg.MODEL.SWINT.DEPTHS = (2, 2, 18, 2)
    glip_l_cfg.MODEL.SWINT.NUM_HEADS = (6, 12, 24, 48)
    glip_l_cfg.MODEL.SWINT.WINDOW_SIZE = 12
    glip_l_cfg.MODEL.SWINT.OUT_CHANNELS = (192, 384, 768, 1536)
    glip_l_cfg.MODEL.SWINT.DROP_PATH_RATE = 0.4

    glip_l_cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    glip_l_cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    glip_l_cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False

    glip_l_cfg.MODEL.RPN.USE_FPN = True
    glip_l_cfg.MODEL.RPN.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
    glip_l_cfg.MODEL.RPN.ANCHOR_STRIDE = (8, 16, 32, 64, 128)
    glip_l_cfg.MODEL.RPN.ASPECT_RATIOS = (1.0,)
    glip_l_cfg.MODEL.RPN.SCALES_PER_OCTAVE = 1

    glip_l_cfg.MODEL.DYHEAD.CHANNELS = 256
    glip_l_cfg.MODEL.DYHEAD.NUM_CONVS = 8
    glip_l_cfg.MODEL.DYHEAD.USE_GN = True
    glip_l_cfg.MODEL.DYHEAD.USE_DYRELU = True
    glip_l_cfg.MODEL.DYHEAD.USE_DFCONV = True
    glip_l_cfg.MODEL.DYHEAD.USE_DYFUSE = True
    glip_l_cfg.MODEL.DYHEAD.TOPK = (
        9  # topk for selecting candidate positive samples from each level
    )
    glip_l_cfg.MODEL.DYHEAD.SCORE_AGG = "MEAN"
    glip_l_cfg.MODEL.DYHEAD.LOG_SCALE = 0.0

    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE = "MHA-B"
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS = False
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS = False
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS = False
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM = 64
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = True
    glip_l_cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = True

    glip_l_cfg.INPUT.PIXEL_MEAN = [103.530, 116.280, 123.675]
    glip_l_cfg.INPUT.PIXEL_STD = [57.375, 57.120, 58.395]

    glip_l_cfg.DATALOADER.SIZE_DIVISIBILITY = 32

    return glip_l_cfg


def make_glip_t(
    device: torch.device, model_weight_path: str, min_image_size: int = None
) -> GLIP:
    return GLIP(
        cfg=make_glip_t_cfg(),
        model_weight_path=model_weight_path,
        device=device,
        min_image_size=min_image_size,
    )


def make_glip_l(
    device: torch.device, model_weight_path: str, min_image_size: int = None
) -> GLIP:
    return GLIP(
        cfg=make_glip_l_cfg(),
        model_weight_path=model_weight_path,
        device=device,
        min_image_size=min_image_size,
    )
