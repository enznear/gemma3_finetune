import argparse
from PIL import Image
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    BitsAndBytesConfig,
)
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a QLoRA fine-tuned vision model")
    parser.add_argument("--model", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--image", required=True, help="Path to an image file")
    return parser.parse_args()


def main():
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoImageProcessor.from_pretrained(args.model)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
    label = model.config.id2label.get(pred, str(pred))
    print(label)


if __name__ == "__main__":
    main()
