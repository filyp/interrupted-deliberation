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
answer_ids = pt.tensor([tokenizer.encode(t)[1:]
                       for t in answer_tokens]).reshape(4)

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

At the end output exactly one of the following:
ANSWER: A
ANSWER: B
ANSWER: C
ANSWER: D
"""
templated_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": full_prompt}], tokenize=False, add_generation_prompt=True)
original_batch = tokenizer(templated_prompt, return_tensors="pt")

pt.manual_seed(45)
# generate original CoT
original_out = model.generate(
    **original_batch, max_new_tokens=1000, temperature=1.0)

print(tokenizer.decode(original_out[0]))
print("correct answer: ", question["target"])


# %%
def get_accuracy(out, question):
    last_token_logits = out.logits[0, -1]
    probs = pt.softmax(last_token_logits, dim=-1)

    answer_probs = probs[answer_ids]
    if answer_probs.sum() < 0.8:  # if it's smaller, we're prompting incorrectly
        # print top-5 tokens
        print(
            f"top-5 tokens: {'___'.join(tokenizer.decode(t) for t in probs.topk(10).indices)}")

    answer_probs /= answer_probs.sum()  # normalize

    _targets = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
    target_idx = _targets.index(question["target"])

    return answer_probs[target_idx].item()


# %%

# interrupt_prompt = "\n\nANSWER:"
# interrupt_prompt = "\n\nLet's try to guess the answer.\n\nANSWER:"
interrupt_prompt = " <output_trimmed>\n\nANSWER:"

original_input_len = original_batch["input_ids"].shape[-1]
cot_length = original_out.shape[-1] - original_input_len


def interrupt_prompt_generator(trimmed, question):
    cot = trimmed.split("assistant<|end_header_id|>\n\n")[-1]
    prompt = f"""<question>{question}</question>
<partial_cot>{cot}</partial_cot>
<instructions>Answer the question based on the partial CoT and the question. Answer exactly one of the following:
ANSWER: A
ANSWER: B
ANSWER: C
ANSWER: D
</instructions>"""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": "ANSWER:"}], tokenize=False, add_generation_prompt=False, continue_final_message=True)


def interrupt_prompt_generator_2(trimmed, question):
    return trimmed + interrupt_prompt


def generate_acc_list(interrupt_prompt_generator):
    acc_list = []
    for i in range(0, cot_length, 1):
        pt.cuda.empty_cache()

        out_text_trimmed = tokenizer.decode(
            original_out[0, : original_input_len + i])
        final_prompt = interrupt_prompt_generator(
            out_text_trimmed, question["input"])
        batch = tokenizer(final_prompt, return_tensors="pt")

        current_token = tokenizer.decode(
            original_out[0, original_input_len + i - 1: original_input_len + i]
        )

        with pt.no_grad():
            out = model(**batch)
        acc = get_accuracy(out, question)
        print(f"i: {i}, accuracy: {acc:.2f}, current_token: {current_token}")
        acc_list.append(acc)
    return acc_list


def generate_acc_list_masked():
    acc_list = []
    for i in range(0, cot_length, 1):
        pt.cuda.empty_cache()
        mask_tokens = original_out[0, :original_input_len +
                                   i].tolist() + [2 for _ in range(cot_length-i)]
        out_text_masked = tokenizer.decode(mask_tokens)
        final_prompt = interrupt_prompt_generator(
            out_text_masked, question["input"])
        batch = tokenizer(final_prompt, return_tensors="pt")

        current_token = tokenizer.decode(
            original_out[0, original_input_len + i - 1: original_input_len + i]
        )

        with pt.no_grad():
            out = model(**batch)
        acc = get_accuracy(out, question)
        print(f"i: {i}, accuracy: {acc:.2f}, current_token: {current_token}")
        acc_list.append(acc)
    return acc_list


# %%
plt.plot(generate_acc_list_masked(), label="CoT masked")
plt.plot(generate_acc_list(interrupt_prompt_generator), label="clean pattern")
plt.plot(generate_acc_list(interrupt_prompt_generator_2), label="interrupt")
plt.legend()
plt.show()
