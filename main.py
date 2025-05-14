# %%
%load_ext autoreload
%autoreload 2
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from datasets import load_dataset
from IPython.display import HTML, display
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

# Initialize markdown content
md_content = ["# Analysis Results\n", f"## {subset.replace('_', ' ').title()}\n"]

for question_id in [2, 7]: #, 9, 10, 13, 15, 21, 26, 28, 29]:
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
    original_out = model.generate(
        **original_batch, max_new_tokens=300, temperature=0.01
    )

    # print(tokenizer.decode(original_out[0]))
    # print("correct answer: ", question["target"])

    acc_list, word_list = get_acc_list_templated(
        model,
        tokenizer,
        question,
        original_batch,
        original_out,
        # interrupt_prompt="\n\n<trimmed_due_to_length>\n\nANSWER:",
        # required_acc=0.6,
        stride=1,
        verbose=False,
    )

    # Save the plot
    plt.plot(acc_list, label="acc")
    plt.ylim(0, 1)
    plt.title(f"{subset} Q{question_id}")
    image_path = f"report/sweep_images/30_questions_from_{subset}_Q{question_id}.svg"
    plt.savefig(image_path, format="svg")
    plt.close()  # Close the plot to free memory

    # Create and display the HTML
    highlighted_text = create_html_highlighted_text(word_list, acc_list)
    display(HTML(highlighted_text))

    # Add content to our collection
    md_content.extend([
        f"\n### Question {question_id}\n",
        f"![Question {question_id} Analysis]({image_path})\n",
        f"\n{highlighted_text}\n",
        "---\n"
    ])

# Save all content at the end
os.makedirs("report", exist_ok=True)
with open("report/results.md", "w") as md_file:
    md_file.write("\n".join(md_content))

