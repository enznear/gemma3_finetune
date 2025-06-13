import argparse
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


SYSTEM_MSG = "You are an expert product description writer for Amazon."
USER_TEMPLATE = (
    "Create a Short Product description based on the provided <PRODUCT> and"
    " <CATEGORY> and image.\n"
    "Only return description. The description should be SEO optimized and for"
    " a better mobile search experience.\n\n"
    "<PRODUCT>\n{product}\n</PRODUCT>\n\n<CATEGORY>\n{category}\n</CATEGORY>\n"
)


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                image = element.get("image", element)
                image_inputs.append(image.convert("RGB"))
    return image_inputs


def collate_fn(examples, processor):
    texts = []
    images = []
    for ex in examples:
        image_inputs = process_vision_info(ex["messages"])
        text = processor.apply_chat_template(
            ex["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100
    batch["labels"] = labels
    return batch


def format_data(sample):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_TEMPLATE.format(
                            product=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ]
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma to generate product descriptions"
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-3-4b-pt",
        help="Gemma model identifier",
    )
    parser.add_argument(
        "--dataset",
        default="philschmid/amazon-product-descriptions-vlm",
        help="Dataset from the HF hub",
    )
    parser.add_argument(
        "--output_dir",
        default="gemma-product-description",
        help="Directory to save the model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16")

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    dataset = load_dataset(args.dataset, split="train")
    dataset = [format_data(sample) for sample in dataset]

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    sft_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=lambda ex: collate_fn(ex, processor),
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
