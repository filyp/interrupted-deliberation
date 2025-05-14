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
# subset = "web_of_lies"
# subset = "logical_deduction_three_objects"
subset = "logical_deduction_five_objects"
dataset = load_dataset(dataset_name, subset, split="train")

# %%
# ok questions: 12, 13, 19, 21

for question_id in range(30):
    question = dataset[question_id]

    full_prompt = f"""\
    {question["input"]}

    Reason about the question step by step.

    At the end output exactly one of the following:
    ANSWER: A
    ANSWER: B
    ANSWER: C
    ANSWER: D
    ANSWER: E
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
        stride=10,
        verbose=False,
    )

    plt.plot(acc_list, label="acc")
    plt.ylim(0, 1)
    plt.title(f"{subset} Q{question_id}")
    plt.show()

    # # Create and display the HTML
    # highlighted_text = create_html_highlighted_text(word_list, acc_list)
    # display(HTML(highlighted_text))
