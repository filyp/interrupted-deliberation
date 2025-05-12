# %%
# %load_ext autoreload
# %autoreload 2
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch as pt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # change it if you want
# device = "cpu"  # change it if you want
pt.set_default_device(device)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# %%
dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

question = dataset[0]

# %%
full_prompt = f"""\
{question["input"]}

Reason about the question step by step.

At the end output:
ANSWER=X

Reasoning:
"""
print(full_prompt)

# %%
batch = tokenizer(full_prompt, return_tensors="pt")
out = model.generate(**batch, max_new_tokens=500)

out_text = tokenizer.decode(out[0])
print(out_text)
# %%
