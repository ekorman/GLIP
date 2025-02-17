# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import torch.nn.functional as F
from torch import nn

from glip.modeling import registry
from glip.modeling.box_coder import BoxCoder
from .anchor_generator import make_anchor_generator_complex
from .inference import make_retina_postprocessor


@registry.RPN_HEADS.register("RetinaNetHead")
class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        if cfg.MODEL.RPN.USE_FPN:
            num_anchors = (
                len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
            )
        else:
            num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * len(
                cfg.MODEL.RPN.ANCHOR_SIZES
            )

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_tower.append(nn.ReLU())

        self.add_module("cls_tower", nn.Sequential(*cls_tower))
        self.add_module("bbox_tower", nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.cls_logits,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_complex(cfg)
        head = RetinaNetHead(cfg)

        box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        box_selector_test = make_retina_postprocessor(cfg, box_coder, is_train=False)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        return self._forward_test(anchors, box_cls, box_regression)

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}
