from peft.tuners.lora import LoraLayer
from peft.utils import transpose
import torch
from torch.nn import Linear

# Backported to peft 0.2
def merge_and_unload(lora_model):
    key_list = [key for key, _ in lora_model.model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = lora_model._get_submodules(key)
        except AttributeError:
            continue
        if isinstance(target, LoraLayer):
            bias = target.bias is not None
            new_module = Linear(target.in_features, target.out_features, bias=bias)

            # manually merge if not merged
            if not target.merged:
                if target.r > 0:
                    target.weight.data += (
                        transpose(target.lora_B.weight @ target.lora_A.weight, target.fan_in_fan_out)
                        * target.scaling
                    ).to(target.weight.dtype)
                target.merged = True

            lora_model._replace_module(parent, target_name, new_module, target)

    return lora_model.model

# Adopted from peft int8 code, adapted for float16 trainig
def prepare_model_for_training(
    model,
    output_embedding_layer_name="lm_head",
    use_gradient_checkpointing=False,
    layer_norm_names=["layer_norm", "layernorm"],
    modules_to_save=[]
):
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        # cast layer norm in fp32 for stability for float16 models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        # fix ValueError: Attempting to unscale FP16 gradients.
        elif any(module in name for module in modules_to_save):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32

            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model

# adopted from https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model, exclude=["lm_head"]):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, Linear) and all(ex not in name for ex in exclude):
            lora_module_names.add(name.split('.')[-1])
    return list(lora_module_names)
