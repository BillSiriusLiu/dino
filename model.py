import os
from typing import Dict, List

import numpy as np
import json
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, TensorType

import torch
from bertwarper import generate_masks_with_special_tokens_and_transfer_map


class GroundingDINO:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get input tensors
            
            img = [s[0].decode("UTF-8") for s in pb_utils.get_input_tensor_by_name(request, "img").as_numpy()]
            caption = [s[0].decode("UTF-8") for s in pb_utils.get_input_tensor_by_name(request, "text").as_numpy()]
            # Preprocessing input data.
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."

            # onnx model input formulation
            captions = [caption]
            # encoder texts
            tokenized = AutoTokenizer(captions, padding="longest", return_tensors="pt")
            specical_tokens = AutoTokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
            
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, specical_tokens, AutoTokenizer)

            if text_self_attention_masks.shape[1] > 256:
                text_self_attention_masks = text_self_attention_masks[
                    :, : 256, : 256]
                
                position_ids = position_ids[:, : 256]
                tokenized["input_ids"] = tokenized["input_ids"][:, : 256]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : 256]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : 256]

            outputs = []
            input_img = np.expand_dims(img, 0)
            outputs.append(pb_utils.Tensor("img", input_img))
            outputs.append(pb_utils.Tensor("input_ids", to_numpy(tokenized["input_ids"])))
            outputs.append(pb_utils.Tensor("attention_mask", to_numpy(tokenized["attention_mask"]).astype(bool)))
            outputs.append(pb_utils.Tensor("token_type_ids", to_numpy(tokenized["token_type_ids"])))
            outputs.append(pb_utils.Tensor("position_ids", to_numpy(position_ids)))
            outputs.append(pb_utils.Tensor("text_token_mask", to_numpy(text_self_attention_masks)))
            
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses