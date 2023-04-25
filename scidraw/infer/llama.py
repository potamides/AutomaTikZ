from peft.tuners.lora import LoraLayer
from peft.utils import transpose
from torch.nn import Linear

# Not available yet in peft 0.2
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
