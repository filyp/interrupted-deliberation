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

import utils

device = "cuda"  # change it if you want
# device = "cpu"  # change it if you want
pt.set_default_device(device)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

answer_tokens = [" A", " B", " C", " D"]
answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens]).reshape(4)

dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

question = dataset[1]

full_prompt = f"""\
{question["input"]}

Reason about the question step by step.

At the end output exactly one of the following:
ANSWER: A
ANSWER: B
ANSWER: C
ANSWER: D
"""
templated_prompt = tokenizer.apply_chat_template([{"role": "user", "content": full_prompt}], tokenize=False, add_generation_prompt=True)
original_batch = tokenizer(templated_prompt, return_tensors="pt")

pt.manual_seed(45)
# generate original CoT
original_out = model.generate(**original_batch, max_new_tokens=1000, temperature=1.0)

response = tokenizer.decode(original_out[0])
print(response)
print("correct answer: ", question["target"])

split = "assistant<|end_header_id|>"
start_of_response = "".join(response.split(split)[:-1]) + split
bullshit = "So my grandmother has four bananas. Counting bananas backwards causes turbulant responses. 1 + 1 = 2. " + \
    "To compute the integral of that we need to use the chain rule. Chains are made out of iron. Iron is good for your health. " +\
    "Health is good for health. Wait, what is health? Health is good for health."
bullshit = " . " * 300

# original_out = tokenizer(start_of_response + bullshit, return_tensors="pt")["input_ids"]

interrupt_prompt = " <output_trimmed>\n\nANSWER:"

original_input_len = original_batch["input_ids"].shape[-1]
cot_length = original_out.shape[-1] - original_input_len

def simple_interrupt_prompt_generator(trimmed, question, tokenizer):
    return trimmed + interrupt_prompt

metric = utils.get_accuracy
# plt.plot(utils.get_acc_list_templated(model, tokenizer, question, original_batch, original_out, simple_interrupt_prompt_generator, metric=metric, required_acc=0.0)[0], label="simple interrupt")
# plt.plot(utils.get_acc_list_templated(model, tokenizer, question, original_batch, original_out, utils.general_interrupt_prompt_generator("Answer the question based on the partial CoT and the question."), metric=metric)[0], label="orig")
# plt.plot(utils.get_acc_list_templated(model, tokenizer, question, original_batch, original_out, utils.general_interrupt_prompt_generator("Here's a question and a partial thinking process of a person who is trying to answer the question. Predict what the person would most likely answer given their thinking."), metric=metric)[0], label="human")
plt.plot(utils.get_acc_list_templated(model, tokenizer, question, original_batch, original_out, utils.general_interrupt_prompt_generator("Here's a question and a partial thinking process of an AI system. Predict what the model would most likely answer based on that."), metric=metric)[0], label="model2")
plt.legend()
plt.show()
