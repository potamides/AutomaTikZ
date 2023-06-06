from transformers import AutoConfig, AutoModelForCausalLM

from .modeling_clima import ClimaConfig, ClimaForCausalLM

def register():
    AutoConfig.register("clima", ClimaConfig)
    AutoModelForCausalLM.register(ClimaConfig, ClimaForCausalLM)
