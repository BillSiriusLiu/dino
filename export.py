import argparse
import os
import sys
import collections
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from groundingdino.util.misc import NestedTensor
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from transformers import AutoTokenizer

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
    checkpoint_path = 'weights/groundingdino_swint_ogc.pth'  # change the path of the model
    onnx_path = 'groundingdinoswinb.onnx'

    device = "cuda" if not cpu_only else "cpu"

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized = tokenizer(["something to say"], padding="longest", return_tensors="pt")

    samples = torch.tensor(torch.rand(1, 3, 1, 1), dtype = torch.long).to(device) # tensor [batch_size x 3 x H x W], mask [batch_size x H x W]
    input_ids = torch.tensor(tokenized['input_ids'], dtype = torch.long).to(device)
    attention_mask = torch.tensor(tokenized['attention_mask'], dtype = torch.long).to(device)
    token_type_ids = torch.tensor(tokenized['token_type_ids'], dtype = torch.long).to(device)

    torch.onnx.export(
            model = model,
            args = (samples, input_ids, attention_mask, token_type_ids),
            f = onnx_path, 
            export_params = True,
            opset_version = 13,
            do_constant_folding = True,
            input_names = ['samples', 'input_ids', 'attention_mask', 'token_type_ids'],
            output_names = ['pred_logits', 'pred_boxes'],
            dynamic_axes = {'samples': {0: 'batch_size', 2: 'H', 3: 'W'},
                            'input_ids': {0:'batch_size', 1:'caption_length'},
                            'attention_mask': {0:'batch_size', 1:'caption_length'},
                            'token_type_ids': {0:'batch_size', 1:'caption_length'},
                            'pred_logits': {0:'batch_size', 1:'objects_cnt'}, # logits [objects x 256]
                            'pred_boxes': {0:'batch_size', 1:'objects_cnt'}} # boxes [objects x 4]
        )