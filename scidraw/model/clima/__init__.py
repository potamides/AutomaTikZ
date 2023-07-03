from transformers import AutoConfig, AutoModelForCausalLM

from .modeling_clima import ClimaConfig, ClimaForCausalLM

def register():
    try:
        AutoConfig.register("clima", ClimaConfig)
        AutoModelForCausalLM.register(ClimaConfig, ClimaForCausalLM)
    except ValueError:
        pass # already registered
