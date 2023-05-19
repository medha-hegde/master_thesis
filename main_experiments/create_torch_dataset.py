# Create Torch Dataset

# code adapted from: https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
from datasets import load_dataset
from torchvision import transforms
import torch
import os


def create_torch_dataset(config, tokenizer):
    data_files = {"train": os.path.join('word_imgs/words', "**")}

    dataset = load_dataset(
        "imagefolder",
        data_files=data_files, cache_dir=None
    )

    # Preprocessing the datasets
    train_transforms = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Generate token input ids
    def text_inputids(txt_list):
        inputs = tokenizer(txt_list, max_length=config["textmodel_maxtokens"], padding="max_length", truncation=True,
                           return_tensors="pt")
        return inputs.input_ids

    image_column = "image"
    caption_column = "text"

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        texts = [t for t in examples[caption_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = text_inputids(texts)
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config["batch_size"],
        num_workers=0)

    return train_dataloader
