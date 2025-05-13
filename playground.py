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

from utils import *

device = "cuda"  # change it if you want
# device = "cpu"  # change it if you want
pt.set_default_device(device)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map=device
)

# For tokenization, you can use the transformers library with the base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

# %%
# ok questions: 12, 13, 19, 21

question_id = 23
question = dataset[question_id]

full_prompt = f"""\
{question["input"]}

Reason about the question step by step.

At the end output exactly one of the following:
ANSWER: A
ANSWER: B
ANSWER: C
"""
templated_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": full_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
original_batch = tokenizer(templated_prompt, return_tensors="pt")

pt.manual_seed(46)
# generate original CoT
# using temperature 0
original_out = model.generate(**original_batch, max_new_tokens=300, temperature=0.01)

print(tokenizer.decode(original_out[0]))
print("correct answer: ", question["target"])

# %%
acc_list, word_list, other_prob1_list, other_prob2_list = get_acc_list_templated(
# acc_list, word_list, other_prob1_list, other_prob2_list = get_acc_list(
    model,
    tokenizer,
    question,
    original_batch,
    original_out,
    interrupt_prompt="\n\n<trimmed_due_to_length>\n\nANSWER:",
    required_acc=0.8,
    return_all_probs=True,
)

plt.plot(acc_list, label="acc")
plt.plot(other_prob1_list, label="other_prob1")
plt.plot(other_prob2_list, label="other_prob2")
plt.title(f"{subset} Q{question_id}")
plt.legend()

# Create and display the HTML
from IPython.display import HTML, display

highlighted_text = create_html_highlighted_text(word_list, acc_list)
display(HTML(highlighted_text))

# # make it logarithmic
# anti_acc_list = [1 - acc for acc in acc_list]
# plt.plot(anti_acc_list)
# plt.yscale('log')

# # %%
# for swapping the answer
# (tokenizer.decode(original_out[0]))
# t = "<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 13 May 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThe following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. A fruit stand sells three fruits: apples, peaches, and mangoes. The mangoes are less expensive than the peaches. The apples are less expensive than the mangoes.\nOptions:\n(A) The peaches are the cheapest\n(B) The apples are the cheapest\n(C) The mangoes are the cheapest\n\nReason about the question step by step.\n\nAt the end output exactly one of the following:\nANSWER: A\nANSWER: B\nANSWER: C<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nTo solve this problem, let's analyze the given statements step by step:\n\n1. The mangoes are less expensive than the peaches.\n2. The apples are less expensive than the mangoes.\n\nFrom statement 1, we know that the mangoes are less expensive than the peaches. This means that the peaches are more expensive than the mangoes.\n\nFrom statement 2, we know that the apples are less expensive than the mangoes. This means that the mangoes are more expensive than the apples.\n\nNow, we have the following order: apples < mangoes < peaches.\n\nSince the apples are less expensive than the mangoes, and the mangoes are less expensive than the peaches, we can conclude that the apples are the cheapest.\n\nANSWER: A<|eot_id|>"
# original_out = tokenizer(t, return_tensors="pt")["input_ids"]
