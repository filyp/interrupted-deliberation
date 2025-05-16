# %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from datasets import load_dataset
from IPython.display import HTML, display
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle

from utils import *

plt.style.use("default")

device = "cuda"  # change it if you want
# device = "cpu"  # change it if you want
pt.set_default_device(device)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map=device
)

model_small_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_small_id = "google/gemma-3-4b-it"
model_small = AutoModelForCausalLM.from_pretrained(
    model_small_id, torch_dtype=pt.bfloat16, device_map=device
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
_tokenizer_small = AutoTokenizer.from_pretrained(model_small_id)
_lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
if "gemma" not in model_small_id:
    assert _tokenizer_small.encode(_lorem) == tokenizer.encode(_lorem)

dataset_name = "maveriq/bigbenchhard"
subset = "logical_deduction_three_objects"
# subset = "tracking_shuffled_objects_three_objects"
dataset = load_dataset(dataset_name, subset, split="train")

prefix = "xxx"

# %%

# Initialize body content list
html_body_parts = []

all_acc_lists = []
all_acc_list_abrupt = []
all_acc_list_small = []
for question_id in [2, 7, 9, 10, 13, 15, 21, 26, 28, 29]:
# for question_id in range(10):
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
    # generate original CoT using temperature 0
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
        required_acc=0.1,
        # stride=1,
        verbose=False,
    )
    acc_list_abrupt, _ = get_acc_list(
        model,
        tokenizer,
        question,
        original_batch,
        original_out,
        required_acc=0.1,
        interrupt_prompt="\n\n<trimmed_due_to_length>\n\nANSWER:",
        # stride=1,
        verbose=False,
    )

    if "gemma" in model_small_id:
        original_batch = _tokenizer_small(question["input"], return_tensors="pt")
        _org_txt = tokenizer.decode(original_out[0])
        original_out = _tokenizer_small.encode(_org_txt, return_tensors="pt")

    acc_list_small, _ = get_acc_list_templated(
        model_small,
        _tokenizer_small,
        question,
        original_batch,
        original_out,
        required_acc=0.1,
        # stride=1,
        verbose=False,
    )
    all_acc_lists.append(acc_list)
    all_acc_list_abrupt.append(acc_list_abrupt)
    all_acc_list_small.append(acc_list_small)

    # Save the plots

    plt.plot(acc_list, label="smooth", color="blue")
    plt.plot(acc_list_abrupt, label="abrupt", color="orange")
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f"{subset} Q{question_id}", pad=10)
    plt.xlabel("CoT token position")
    plt.ylabel("Accuracy")
    image_path = (
        f"images_smooth_vs_abrupt/{prefix}_{subset}_Q{question_id}.svg"
    )
    plt.savefig("docs/" + image_path, format="svg")
    plt.show()
    plt.close()

    plt.plot(acc_list, label="Llama-3.2-3B-Instruct", color="blue")
    plt.plot(acc_list_small, label="GEMMA-3-4B-IT", color="green")
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f"{subset} Q{question_id}", pad=10)
    plt.xlabel("CoT token position")
    plt.ylabel("Accuracy")
    image_path = (
        f"images_smooth_vs_abrupt/{prefix}_{subset}_Q{question_id}.svg"
    )
    plt.savefig("docs/" + image_path, format="svg")
    plt.show()
    plt.close()

    plt.plot(acc_list, label="smooth-Llama-3.2-3B-Instruct", color="blue")
    plt.plot(acc_list_abrupt, label="abrupt-Llama-3.2-3B-Instruct", color="orange")
    plt.plot(acc_list_small, label="smooth-GEMMA-3-4B-IT", color="green")
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f"{subset} Q{question_id}", pad=10)
    plt.xlabel("CoT token position")
    plt.ylabel("Accuracy")
    image_path = f"images/{prefix}_{subset}_Q{question_id}.svg"
    plt.savefig("docs/" + image_path, format="svg")
    plt.show()
    plt.close()

    # Create HTML content with styling
    highlighted_text = create_html_highlighted_text(word_list, acc_list)

    # Add question content to body parts
    html_body_parts.append(
        f"""
    <div class="question">
        <h3>Question {question_id}</h3>
        <pre>{question['input']}</pre>
        <p>Correct answer: {question['target']}</p>
        <img src="{image_path}" class="plot" alt="Accuracy plot for question {question_id}">
        <div class="token-viz">
            {highlighted_text}
        </div>
    </div>
    """
    )

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
    </style>
</head>
<body>
    <h1>Logical Deduction Three Objects Dataset</h1>
"""
# Combine header and body
html_content = html_header + "\n".join(html_body_parts) + "\n</body>\n</html>"

# Save HTML file
with open(f"docs/analysis_{prefix}_{subset}.html", "w") as f:
    f.write(html_content)

# %%
# save the all acc lists
dir_name = f"reports/analysis_{prefix}_{subset}"
# mkdir if not exists
os.makedirs(dir_name, exist_ok=True)
with open(f"{dir_name}/all_acc_lists.pkl", "wb") as f:
    pickle.dump(all_acc_lists, f)
with open(f"{dir_name}/all_acc_list_abrupt.pkl", "wb") as f:
    pickle.dump(all_acc_list_abrupt, f)
with open(f"{dir_name}/all_acc_list_small.pkl", "wb") as f:
    pickle.dump(all_acc_list_small, f)

