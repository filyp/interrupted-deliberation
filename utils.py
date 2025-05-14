# %%
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch as pt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

answer_tokens = [" A", " B", " C"]


def get_probabilities(out, tokenizer):
    last_token_logits = out.logits[0, -1].to(pt.float32)
    probs = pt.softmax(last_token_logits, dim=-1)

    answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens])
    answer_ids = answer_ids.reshape(len(answer_tokens))
    answer_probs = probs[answer_ids]
    return answer_probs


def get_accuracy(out, question, tokenizer, required_acc=0.8, return_all_probs=False):
    answer_probs = get_probabilities(out, tokenizer)
    # if it's smaller, we're prompting incorrectly
    assert answer_probs.sum() > required_acc, f"probs sum: {answer_probs.sum()}"
    if answer_probs.sum() < 0.5:
        print(f"warning: probs sum: {answer_probs.sum()}")

    answer_probs /= answer_probs.sum()  # normalize

    _targets = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
    target_idx = _targets.index(question["target"])

    correct_prob = answer_probs[target_idx].item()
    other_probs = [el.item() for i, el in enumerate(answer_probs) if i != target_idx]
    other_prob1 = other_probs[0]
    other_prob2 = other_probs[1]

    if return_all_probs:
        return correct_prob, other_prob1, other_prob2
    else:
        return correct_prob


def get_is_correct(out, question, tokenizer):
    answer_probs = get_probabilities(out, tokenizer)
    return f"({answer_tokens[pt.argmax(answer_probs).item()][-1]})" == question["target"]


def get_entropy(out, question, tokenizer, _):
    answer_probs = get_probabilities(out, tokenizer)
    entropy = -pt.sum(answer_probs * pt.log(answer_probs))
    return entropy.item()


def create_html_highlighted_text(words, accuracies):
    html_parts = []
    for word, acc in zip(words, accuracies):
        # Convert accuracy to green background color with alpha (0 to 1)
        green_value = 190
        alpha = acc  # alpha will be 0 when acc is 0, 1 when acc is 1
        color = f"rgba(0, {green_value}, 0, {alpha})"
        # Replace newline with HTML line break
        # word = word.replace("\n", "<br>")
        html_parts.append(f'<span style="background-color: {color}">{word}</span>')

    return "".join(html_parts)


def general_interrupt_prompt_generator(instructions):
    def inner(trimmed, question_text, tokenizer):
        cot = trimmed.split("assistant<|end_header_id|>\n\n")[-1]
        assert isinstance(question_text, str)
        prompt = f"""\
<question>{question_text}</question>
<partial_cot>{cot}</partial_cot>
<instructions>{instructions} Answer exactly one of the following:
ANSWER: A
ANSWER: B
ANSWER: C
</instructions>"""
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ANSWER:"},
            ],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    return inner


interrupt_prompt_generator = general_interrupt_prompt_generator(
    "Answer the question based on the partial CoT and the question."
)


def get_acc_list_templated(
    model,
    tokenizer,
    question,
    original_batch,
    original_out,
    template_function=interrupt_prompt_generator,
    required_acc=0.8,
    metric=get_accuracy,
    return_all_probs=False,
    stride=1,
    verbose=True,
):
    orig_input_len = original_batch["input_ids"].shape[-1]
    cot_length = original_out.shape[-1] - orig_input_len

    acc_list = []
    word_list = []
    other_prob1_list = []
    other_prob2_list = []
    for i in range(0, cot_length, stride):
        pt.cuda.empty_cache()

        out_text_trimmed = tokenizer.decode(original_out[0, : orig_input_len + i])
        question_text = question["input"]
        batch = tokenizer(
            template_function(out_text_trimmed, question_text, tokenizer),
            return_tensors="pt",
        )

        current_token = tokenizer.decode(
            original_out[0, orig_input_len + i - 1 : orig_input_len + i]
        )

        with pt.no_grad():
            out = model(**batch)
        acc = metric(out, question, tokenizer, required_acc, return_all_probs)
        if return_all_probs:
            acc, other_prob1, other_prob2 = acc
            other_prob1_list.append(other_prob1)
            other_prob2_list.append(other_prob2)
        if verbose:
            print(f"i: {i}, accuracy: {acc:.2f}, current_token: {current_token}")
        acc_list.append(acc)
        word_list.append(current_token)

    if return_all_probs:
        return acc_list, word_list, other_prob1_list, other_prob2_list
    else:
        return acc_list, word_list


def get_acc_list(
    model,
    tokenizer,
    question,
    original_batch,
    original_out,
    interrupt_prompt,
    required_acc=0.8,
    return_all_probs=False,
    verbose=True,
):
    def interrupt_prompt_generator(out_text_trimmed, question, tokenizer):
        return out_text_trimmed + interrupt_prompt

    return get_acc_list_templated(
        model,
        tokenizer,
        question,
        original_batch,
        original_out,
        interrupt_prompt_generator,
        required_acc=required_acc,
        metric=get_accuracy,
        return_all_probs=return_all_probs,
        verbose=verbose,
    )


# %%
