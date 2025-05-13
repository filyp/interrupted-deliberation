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

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

# %%
# ok questions: 12, 13

question_id = 19
question = dataset[question_id]

templeted_chat = [
    {
        "role": "system",
        "content": (
            """
Reason about the question step by step. If you encounter the <output_trimmed> marker, answer immediately.

At the end, output (exactly as is) one of:
ANSWER: A
or
ANSWER: B
or
ANSWER: C
or
ANSWER: D
"""
        ),
    },
    {"role": "user", "content": question["input"]},
]
full_prompt = tokenizer.apply_chat_template(
    templeted_chat, tokenize=False, add_generation_prompt=True
)
original_batch = tokenizer(full_prompt, return_tensors="pt")

pt.manual_seed(46)
# generate original CoT
# using temperature 0
original_out = model.generate(**original_batch, max_new_tokens=300, temperature=0.01)

print(tokenizer.decode(original_out[0]))
print("correct answer: ", question["target"])


# %%

interrupt_prompt = "\n\nANSWER:"
# interrupt_prompt = "\n\nLet's try to guess the answer.\n\nANSWER:"
# interrupt_prompt = " <output_trimmed>\n\nANSWER:"

orig_input_len = original_batch["input_ids"].shape[-1]
cot_length = original_out.shape[-1] - orig_input_len

acc_list, word_list = get_acc_list(
    model, tokenizer, question, original_batch, original_out, interrupt_prompt, required_acc=0.2
)
# %%

# plt.plot(acc_list, label="3B")
# plt.plot(acc_list2, label="1B")
# plt.title(f"{subset} Q{question_id}")
# plt.legend()

# %%

plt.plot(acc_list)
plt.title(f"{subset} Q{question_id}")

# Create and display the HTML
from IPython.display import HTML, display
highlighted_text = create_html_highlighted_text(word_list, acc_list)
display(HTML(highlighted_text))

# # make it logarithmic
# anti_acc_list = [1 - acc for acc in acc_list]
# plt.plot(anti_acc_list)
# plt.yscale('log')

# %%