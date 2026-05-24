# Fine-Tuning Playground

A simple Hugging Face fine-tuning setup for causal language models using TRL SFTTrainer and PEFT (LoRA).

## What this does

- Loads a JSONL dataset using `utils.py`
- Formats each example into a prompt/response text string
- Fine-tunes a causal language model with LoRA adapters
- Saves the trained model and tokenizer to `output_dir`

## Files

- `train.py`: training script
- `utils.py`: dataset loading and formatting helper functions

## Dataset format

The dataset should be a JSONL file where each line is a JSON object containing one example.
The helper looks for the following fields:

- input text: `input_text`, `task_text`, `input`
- output text: `gold`, `output`, `target`, `answer`

Example line:

```json
{"input_text": "Convert 10am PST to UTC", "gold": "2024-05-24T17:00:00+00:00"}
```

If your examples use a different field names, you can update `utils.py`.

## How formatting works

The script creates training text with a custom chat-like prompt format:

- `BOS` and `EOT` are special tokens
- `SYS_HDR` includes a system instruction
- `USR_HDR` contains the user input
- `AST_HDR` introduces the assistant response

The formatted example looks like:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
...system instruction...<|eot_id|><|start_header_id|>user<|end_header_id|>
...input...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
...output...<|eot_id|>
```

## Run training

```bash
python train.py --data_path path/to/data.jsonl --output_dir path/to/output
```

Optional arguments:

- `--model_name`: model to fine-tune (default: `microsoft/phi-4`)
- `--token`: Hugging Face token for private models
- `--epochs`: number of training epochs (default: `1`)
- `--lr`: learning rate (default: `2e-4`)
- `--batch_size`: batch size (default: `2`)
- `--grad_accum`: gradient accumulation steps (default: `16`)
- `--use_4bit`: enable 4-bit quantization

## Requirements

This setup uses:

- `transformers`
- `datasets`
- `trl`
- `peft`
- `huggingface_hub` (optional, only if using `--token`)

Install them with pip as needed.

## Notes

- The script sets `tokenizer.pad_token` to the model `eos_token` if no pad token exists.
- The model is loaded with `trust_remote_code=True` so it can use custom model code from the repository.
- LoRA is configured on attention projection modules for efficient fine-tuning.

That's it! Use `train.py` and a properly formatted JSONL dataset to fine-tune your model.
