import numpy as np
from PIL import Image
from transformers import AutoTokenizer
import torch
from torchvision import transforms as T

from glip.modeling.detector import build_detection_model
from glip.structures.image_list import to_image_list
from glip.modeling.roi_heads.mask_head.inference import Masker
from glip.utils.model_serialization import load_state_dict


class GLIP(object):
    def __init__(
        self,
        cfg,
        device: torch.device,
        model_weight_path: str,
        confidence_threshold=0.7,
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
        self.confidence_threshold = confidence_threshold

        self.tokenizer = self.build_tokenizer()

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """

        cfg = self.cfg

        to_bgr_transform = T.Lambda(lambda x: x * 255)

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
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

    def __call__(self, img: Image.Image, class_labels, thresh=0.5):
        img = np.array(img)[:, :, [2, 1, 0]]
        predictions = self.compute_prediction(img, class_labels)
        top_predictions = self._post_process(predictions, thresh)

        return top_predictions

    def compute_prediction(self, original_image, class_labels):
        # image
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
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
                captions=[caption_string],
                positive_map=positive_map_label_to_token,
            )
            predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
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
