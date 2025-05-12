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

answer_tokens = [" A", " B", " C", " D"]
answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens]).reshape(4)

# %%
dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

question = dataset[1]

# %%
full_prompt = f"""\
{question["input"]}

Reason about the question step by step.

At the end output:
ANSWER: X

Reasoning:
"""
original_batch = tokenizer(full_prompt, return_tensors="pt")

pt.manual_seed(42)
# generate original CoT
original_out = model.generate(**original_batch, max_new_tokens=1000, temperature=1.0)

# print(tokenizer.decode(original_out[0]))
# print(question["target"])

# %%
def get_accuracy(out, question):
    last_token_logits = out.logits[0, -1]
    probs = pt.softmax(last_token_logits, dim=-1)

    answer_probs = probs[answer_ids]
    assert answer_probs.sum() > 0.8  # if it's smaller, we're prompting incorrectly

    answer_probs /= answer_probs.sum()  # normalize

    _targets = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
    target_idx = _targets.index(question["target"])

    return answer_probs[target_idx]


interrupt_prompt = "\n\nANSWER:"

out_text_trimmed = tokenizer.decode(original_out[0, :-180])
print(out_text_trimmed + interrupt_prompt)
batch = tokenizer(out_text_trimmed + interrupt_prompt, return_tensors="pt")

pt.cuda.empty_cache()
out = model(**batch)
get_accuracy(out, question)

# %%
