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


def get_accuracy(out, question):
    last_token_logits = out.logits[0, -1]
    probs = pt.softmax(last_token_logits, dim=-1)

    answer_probs = probs[answer_ids]
    assert answer_probs.sum() > 0.8  # if it's smaller, we're prompting incorrectly

    answer_probs /= answer_probs.sum()  # normalize

    _targets = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
    target_idx = _targets.index(question["target"])

    return answer_probs[target_idx].item()


# %%
dataset_name = "maveriq/bigbenchhard"
# subset = "web_of_lies"
subset = "logical_deduction_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

question = dataset[4]

# %%
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
original_out = model.generate(**original_batch, max_new_tokens=1000, temperature=1.0)

print(tokenizer.decode(original_out[0]))
print("correct answer: ", question["target"])


# %%

interrupt_prompt = "\n\nANSWER:"
# interrupt_prompt = "\n\nLet's try to guess the answer.\n\nANSWER:"
# interrupt_prompt = " <output_trimmed>\n\nANSWER:"

orig_input_len = original_batch["input_ids"].shape[-1]
cot_length = original_out.shape[-1] - orig_input_len

acc_list = []
word_list = []
for i in range(0, cot_length, 1):
    pt.cuda.empty_cache()

    out_text_trimmed = tokenizer.decode(original_out[0, : orig_input_len + i])
    batch = tokenizer(out_text_trimmed + interrupt_prompt, return_tensors="pt")

    current_token = tokenizer.decode(
        original_out[0, orig_input_len + i - 1 : orig_input_len + i]
    )

    with pt.no_grad():
        out = model(**batch)
    acc = get_accuracy(out, question)
    print(f"i: {i}, accuracy: {acc:.2f}, current_token: {current_token}")
    acc_list.append(acc)
    word_list.append(current_token)

plt.plot(acc_list)

# %%

def create_html_highlighted_text(words, accuracies):
    html_parts = []
    for word, acc in zip(words, accuracies):
        # Convert accuracy to green background color (0 to 1)
        max_color = 190
        green_value = int(acc * max_color)
        color = f'rgb(0, {green_value}, 0)'
        html_parts.append(f'<span style="background-color: {color}">{word}</span>')
    
    return ''.join(html_parts)

# Create and display the HTML
from IPython.display import HTML, display
highlighted_text = create_html_highlighted_text(word_list, acc_list)
display(HTML(highlighted_text))

# %%
