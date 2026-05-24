"""
utils.py

Helpers for finetuning CivilTime models.
Includes prompt formatting to match the training/inference template.
"""

from datasets import load_dataset

BOS = "<|begin_of_text|>"
SYS_HDR = "<|start_header_id|>system<|end_header_id|>\n"
USR_HDR = "<|start_header_id|>user<|end_header_id|>\n"
AST_HDR = "<|start_header_id|>assistant<|end_header_id|>\n"
EOT = "<|eot_id|>"

DEFAULT_SYS = (
    "You are a precise time normalizer. Output ONE line in ISO-8601 with offset "
    "unless policy says ABSTAIN or A||B. No prose."
)

def load_jsonl_dataset(path):
    return load_dataset("json", data_files=path, split="train")

def format_instruction(ex):
    """
    Convert a raw example (input/gold) into SFT text format.
    Supports Llama-3 style headers (check your model's chat template specifics).
    """
    
    # Try different key pairs used in the raw data
    inp = ex.get("input_text") or ex.get("task_text") or ex.get("input") or ""
    out = ex.get("gold") or ex.get("output") or ex.get("target") or ex.get("answer") or ""
    
    # Handle chat messages format if present
    if "messages" in ex:
        # TODO: Implement messages parsing if dataset is in chat format
        pass

    prompt = f"{BOS}{SYS_HDR}{DEFAULT_SYS}{EOT}{USR_HDR}{inp}{EOT}{AST_HDR}"
    response = f"{out}{EOT}"
    
    return {"text": prompt + response}
