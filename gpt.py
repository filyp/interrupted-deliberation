# %%
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from IPython.display import HTML, display
from openai import OpenAI
from pydantic import BaseModel, conlist

plt.style.use("default")

client = OpenAI(api_key=Path("~/.config/openai.token").expanduser().read_text().strip())

dataset_name = "maveriq/bigbenchhard"
subset = "logical_deduction_seven_objects"
# subset = "tracking_shuffled_objects_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

# %%
system_prompt = "You are a helpful assistant."
chunk_size = 25
models = ["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"]

# %% get no COT accs

prompt = "Try to guess the final answer. Output ONLY one letter (A/B/C/...). Any other text will cause parsing errors."
no_cot_accs = []
for question_id in range(len(dataset)):
    question = dataset[question_id]
    model = models[0]
    correct_answer = question["target"].strip("()")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question["input"] + "\n\n" + prompt},
        ],
        temperature=1,
        n=100,
        max_completion_tokens=1,
    )
    _ans_sample = [ch.message.content for ch in completion.choices]
    acc = sum(1 for ch in _ans_sample if ch == correct_answer) / len(_ans_sample)
    print(f"question {question_id} has acc={acc}")
    no_cot_accs.append(acc)

with open(f"report/gpt_{subset}_accs.json", "w") as f:
    json.dump(no_cot_accs, f)


# %%
class Probs(BaseModel):
    A: int
    B: int
    C: int
    D: int
    E: int
    F: int
    G: int
    # short_justificaton: str


def get_accs_sampled(words, question, model):
    correct_answer = question["target"].strip("()")
    accs = []
    interrup_prompt = "Enough reasoning. Based on reasoning so far, try to guess the final answer. Output ONLY one letter (A/B/C/...). Any other text will cause parsing errors."
    # also use the incomplete remainder chunk, hence "+ chunk_size"
    for i in range(0, len(words) + chunk_size, chunk_size):
        trimmed_reasoning = " ".join(words[:i])

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question["input"]},
                {"role": "assistant", "content": trimmed_reasoning},
                {"role": "user", "content": interrup_prompt},
            ],
            temperature=1,
            n=100,
            max_completion_tokens=1,
        )
        _ans_sample = [ch.message.content for ch in completion.choices]
        acc = sum(1 for ch in _ans_sample if ch == correct_answer) / len(_ans_sample)
        accs.append(acc)
        if i == 0 and acc > 0.5:
            return
    return accs


def get_accs_estimated(words, question, model):
    correct_answer = question["target"].strip("()")
    accs = []
    interrup_prompt = """\
Enough reasoning. Based on the reasoning so far, try to guess the final answer. For each letter output a score from 0% to 100% with your estimated probability of that answer. If you're not confident yet, spread out the probability over many answers. There is a risk of being overconfident here."""

    for i in range(0, len(words), chunk_size):
        trimmed_reasoning = " ".join(words[:i])

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question["input"]},
                {"role": "assistant", "content": trimmed_reasoning},
                {"role": "user", "content": interrup_prompt},
            ],
            temperature=0,
            response_format=Probs,
        )
        all_estimations = completion.choices[0].message.parsed
        acc = getattr(all_estimations, correct_answer) / 100
        accs.append(acc)
        if i == 0 and acc > 0.5:
            return
    return accs


def process_question(question_id):
    question = dataset[question_id]
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question["input"]},
        ],
        temperature=0,
        max_completion_tokens=1500,
    )
    full_reasoning = completion.choices[0].message.content
    words = full_reasoning.split(" ")
    print(f"question {question_id} has {len(words)} words")

    accs_perlabel = dict()
    for model in models:
        acc = get_accs_sampled(words, question, model)
        if acc is None:
            print(f"question {question_id} is too easy")
            return None, None
        label = "sampled " + model
        accs_perlabel[label] = acc

        acc = get_accs_estimated(words, question, model)
        if acc is None:
            print(f"question {question_id} is too easy")
            return None, None
        label = "estimated " + model
        accs_perlabel[label] = acc

    # ! plot
    for label, accs in accs_perlabel.items():
        xs = np.arange(len(accs)) * chunk_size
        plt.plot(xs, accs, label=label)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.title(f"{subset} Q{question_id}", pad=10)
    plt.xlabel("CoT token position")
    plt.ylabel("Accuracy")
    image_path = f"images_gpt/{subset}_Q{question_id}.svg"
    plt.savefig("docs/" + image_path, format="svg")
    plt.show()
    plt.close()

    # create html
    highlighted_text_per_label = dict()
    for label in accs_perlabel:
        highlighted_text = []
        for i, acc in enumerate(accs_perlabel[label]):
            chunk = " ".join(words[i * chunk_size : (i + 1) * chunk_size]) + " "

            # Convert accuracy to green background color with alpha (0 to 1)
            green_value = 190
            alpha = acc  # alpha will be 0 when acc is 0, 1 when acc is 1
            color = f"rgba(0, {green_value}, 0, {alpha})"
            highlighted_text.append(
                f'<span style="background-color: {color}">{chunk}</span>'
            )
        highlighted_text_per_label[label] = "".join(highlighted_text)

    # Create foldable sections for each model's reasoning
    reasoning_sections = []
    for i, label in enumerate(accs_perlabel):
        is_open = "open" if i == 0 else ""  # First model is open by default
        reasoning_sections.append(
            f"""\
        <details {is_open}>
            <summary>Reasoning for {label}</summary>
            <div class="token-viz">
                {highlighted_text_per_label[label]}
            </div>
        </details>"""
        )

    return (
        accs_perlabel,
        f"""\
    <div class="question">
        <h3>Question {question_id}</h3>
        <pre>{question['input']}</pre>
        <p>Correct answer: {question['target']}</p>
        <img src="{image_path}" class="plot" alt="Accuracy plot for question {question_id}">
        <div class="reasoning-sections">
            {"".join(reasoning_sections)}
        </div>
    </div>""",
    )


# %%
html_body_parts = []
accs_perquestion_perlabel = []
for question_id in range(len(dataset)):
    no_cot_acc = no_cot_accs[question_id]
    if no_cot_acc > 0.5:
        print(f"question {question_id} is too easy, skipped")
        continue
    # for question_id in [2]:
    accs_perlabel, html_part = process_question(question_id)
    if accs_perlabel is not None:
        html_body_parts.append(html_part)
        accs_perquestion_perlabel.append(accs_perlabel)
    if len(html_body_parts) >= 10:
        break

# %%

# Initialize HTML content with header
html_header = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { 
            font-family: monospace;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
        }
        span { 
            display: inline-block;
            padding: 2px;
        }
        .question {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 5px;
        }
        .plot {
            width: 100%;
            max-width: 600px;
            margin: 20px 0;
        }
        pre, code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }
        details {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        summary {
            cursor: pointer;
            font-weight: bold;
            padding: 5px;
        }
        summary:hover {
            background-color: #f5f5f5;
        }
        .reasoning-sections {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Logical Deduction Three Objects Dataset</h1>
"""
# Combine header and body
html_content = html_header + "\n".join(html_body_parts) + "\n</body>\n</html>"

# Save HTML file
with open(f"docs/gpt_{subset}.html", "w") as f:
    f.write(html_content)

# %%
# save pickled accs_perquestion_perlabel
with open(f"report/gpt_{subset}_accs_perquestion_perlabel.pkl", "wb") as f:
    pickle.dump(accs_perquestion_perlabel, f)

# %%
