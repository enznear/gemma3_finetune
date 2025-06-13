import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a vision model with QLoRA")
    parser.add_argument("--model", default="google/vit-base-patch16-224", help="Model ID to fine-tune")
    parser.add_argument("--dataset", default="beans", help="Dataset name from the HF hub")
    parser.add_argument("--output", default="finetuned_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Train batch size")
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

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.IMAGE_CLASSIFICATION,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.dataset)

    def preprocess(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = batch["labels"]
        return inputs

    train_ds = dataset["train"].map(preprocess, batched=True)
    val_ds = dataset.get("validation")
    if val_ds:
        val_ds = val_ds.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        fp16=True,
        evaluation_strategy="steps" if val_ds else "no",
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
