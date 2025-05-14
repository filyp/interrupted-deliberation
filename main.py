# %%
%load_ext autoreload
%autoreload 2
import logging
import os
from copy import deepcopy

from IPython.display import HTML, display
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
subset = "logical_deduction_three_objects"
# subset = "tracking_shuffled_objects_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

# %%
# ok questions: 12, 13, 19, 21

all_acc_lists = []
all_original_out = []
for question_id in range(0, 30):
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
    all_original_out.append(original_out)

    # print(tokenizer.decode(original_out[0]))
    # print("correct answer: ", question["target"])

    acc_list, word_list = get_acc_list_templated(
        model,
        tokenizer,
        question,
        original_batch,
        original_out,
        # interrupt_prompt="\n\n<trimmed_due_to_length>\n\nANSWER:",
        required_acc=0.6,
        stride=30,
        verbose=False,
    )

    plt.plot(acc_list, label="acc")
    plt.ylim(0, 1)
    plt.title(f"{subset} Q{question_id}")
    plt.show()

    all_acc_lists.append(acc_list)

    # # Create and display the HTML
    # highlighted_text = create_html_highlighted_text(word_list, acc_list)
    # display(HTML(highlighted_text))

# # %%
# plt.figure(figsize=(5, 4))
# for acc_list in filt:
#     plt.plot(acc_list, label="acc")
# plt.ylim(0, 1)
# # plt.legend()
# # Set both positions and labels for x-ticks
# plt.xticks([0, len(acc_list)//2, len(acc_list)-1], ["start", "middle", "end"])
# plt.title(f"30 questions from {subset}", pad=10)
# plt.tight_layout()
# plt.show()

# # save as svg
# plt.savefig(f"report/pictures/30_questions_from_{subset}.svg", format="svg")

# # %%
# filt = a[a[:, 0] < 0.5]
# # %%
# import numpy as np

# a = np.array(all_acc_lists)
# a
# # %%
# a[:, 0] < 0.5

# # %%
# filt.mean(axis=0)