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

try:
    from llama_cpp import Llama
except ImportError:
    pass

from utils import *

device = "cuda"  # change it if you want
# device = "cpu"  # change it if you want
pt.set_default_device(device)

# del model
# pt.cuda.empty_cache()
model = Llama(
    model_path="data/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf",
    model_type="Q6_K",
    n_ctx=4096,  # Start with a smaller context
    n_gpu_layers=-1,  # Use all GPU layers if available
    offload_kqv=True,  # Enable KQV offloading to GPU
    n_batch=512,
    verbose=True,
)

# %%

dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

# %%
# ok questions: 12, 13, 19

question_id = 1
question = dataset[question_id]

full_prompt = f"""\
{question["input"]}

Try to reason for about 200 words, not longer.

At the end output exactly one of the following:
ANSWER: A
ANSWER: B
ANSWER: C
ANSWER: D
"""

# %%

out = model(full_prompt, max_tokens=1000, temperature=0.01)
print(out["choices"][0]["text"])
print("correct answer: ", question["target"])

# %%
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

original_batch = tokenizer(full_prompt, return_tensors="pt")
original_out = tokenizer(out["choices"][0]["text"], return_tensors="pt")
orig_input_len = original_batch["input_ids"].shape[-1]
cot_length = original_out["input_ids"].shape[-1] - orig_input_len

# %%

acc_list = []
word_list = []
for i in range(0, cot_length, 1):
    pt.cuda.empty_cache()

    out_text_trimmed = tokenizer.decode(original_out["input_ids"][0, : orig_input_len + i])
    input_text = out_text_trimmed + "\n\nANSWER:"

    current_token = tokenizer.decode(
        original_out[0, orig_input_len + i - 1 : orig_input_len + i]
    )

    with pt.no_grad():
        out = model(input_text, max_tokens=100, temperature=0.01)

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
