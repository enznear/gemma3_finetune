# gemma3_finetune

This repository provides an example script for fine-tuning a vision model with QLoRA using the Hugging Face ecosystem. The implementation loosely follows the sample shown in the Gemma documentation.

## Requirements

- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`
- `Pillow`
- `torch`

All dependencies are listed in `requirements.txt`.

## Setup

Create a Python virtual environment and install the requirements:

```bash
bash scripts/setup_venv.sh
```

The script creates a `venv` directory using Python's `venv` module and installs
the packages from `requirements.txt`. You can activate the environment
afterwards with:

```bash
source venv/bin/activate
```

## Usage

Run the training script specifying the model and dataset you want to use:

```bash
python scripts/finetune_vision_qlora.py --model google/vit-base-patch16-224 --dataset beans --epochs 3 --batch 4 --output finetuned_model
```

The script loads the dataset from the Hugging Face hub, applies a 4-bit quantization setup, and fine-tunes the model with LoRA adapters.

## Inference

After training completes, you can run inference on an image using:

```bash
python scripts/predict_vision_qlora.py --model finetuned_model --image path/to/image.jpg
```

The script loads the fine-tuned model with 4-bit weights and prints the predicted label for the input image.
