# Adopted from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava.py. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from itertools import count
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BatchEncoding,
    CLIPModel,
    CLIPProcessor,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)


class ClimaConfig(LlamaConfig):
    model_type = "clima"


class ClimaModel(LlamaModel):
    config_class = ClimaConfig

    def __init__(self, config: LlamaConfig):
        super(ClimaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # HACK: wrap in list to not make vision model count as a parameter
            self.vision_tower = [CLIPModel.from_pretrained(config.mm_vision_tower)]

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, vision_tower, mask_token_id, pretrain_mm_mlp_adapter=None):
        self.config.mm_vision_tower = vision_tower

        processor = CLIPProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPModel.from_pretrained(vision_tower, torch_dtype=self.dtype).to(self.device)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.projection_dim
        vision_config.im_patch_token = mask_token_id
        vision_config.num_patches = 1 # since we use the pooled, projected output

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.projection_dim, self.config.hidden_size, dtype=self.dtype, device=self.device)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location=self.device)
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            processor=processor,
            vision_config=vision_config
        )

    # https://stackoverflow.com/a/57208704
    def _apply(self, fn):
        super()._apply(fn)
        self.vision_tower = [vis._apply(fn) for vis in self.vision_tower]
        return self

    def get_vision_features(self, text_or_image, *args, **kwargs):
        if isinstance(text_or_image, (BatchEncoding, dict)):
            if "pixel_values" in text_or_image:
                embeds = self.vision_tower[0].get_image_features(*args, **text_or_image, **kwargs)
            else:
                embeds = self.vision_tower[0].get_text_features(*args, **text_or_image, **kwargs)
        else:
            if text_or_image.dim() == 4:
                embeds = self.vision_tower[0].get_image_features(text_or_image, *args, **kwargs)
            else:
                embeds = self.vision_tower[0].get_text_features(text_or_image, *args, **kwargs)

        return embeds.unsqueeze(1)

    def is_tensor(self, thing):
        if isinstance(thing, (BatchEncoding, dict)):
            return all(isinstance(v, torch.Tensor) for v in thing.values())
        return isinstance(thing, torch.Tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Union[BatchEncoding, torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            vision_tower = vision_tower[0]
            with torch.no_grad():
                if self.is_tensor(images):
                    image_features = self.get_vision_features(images)
                # variable length images or texts / multimodal inputs with both image and texts
                elif isinstance(images, (BatchEncoding, dict)): # BatchEncoding of lists of tensor
                    image_features = []
                    try:
                        for idx in count():
                            image_feature = self.get_vision_features({k: v[idx].unsqueeze(0) for k, v in images.items()})
                            image_features.append(image_feature)
                    except IndexError:
                        pass
                elif isinstance(images[0], (BatchEncoding, dict)): # list of BatchEncoding of tensor
                    image_features = []
                    for image in images:
                        image_feature = self.get_vision_features({k: v.unsqueeze(0) for k, v in image.items()})
                        image_features.append(image_feature)
                else: # same, but only the list
                    for image in images:
                        image_feature = self.get_vision_features(image.unsqueeze(0))
                        image_features.append(image_feature)
            if self.is_tensor(images):
                image_features = self.mm_projector(image_features)
            else:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            dummy_image_features = torch.zeros(len(image_features[0]), self.config.mm_hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue

                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                    raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                cur_image_idx += 1

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(ClimaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ClimaForCausalLM(LlamaForCausalLM):
    config_class = ClimaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ClimaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Union[BatchEncoding, torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
