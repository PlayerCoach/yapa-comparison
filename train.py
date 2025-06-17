import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset as HFDataset, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split
import evaluate
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,  # type:ignore
    Trainer,  # type: ignore
)


# === CONFIG ===
csv_path = "spectrogram_dataset.csv"
num_epochs = 5
batch_size = 64
model_name = "google/vit-base-patch16-224-in21k"

# === LOAD DATA ===
df = pd.read_csv(csv_path)
label_names = sorted(df["label"].unique())
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for name, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# === SPLIT DATA ===
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# === HUGGINGFACE FEATURES ===
features = Features(
    {
        "image_path": Value("string"),
        "label": ClassLabel(names=label_names),
        "label_id": Value("int64"),
    }
)
train_df = train_df.reset_index(drop=True)
train_dataset = HFDataset.from_pandas(train_df, features=features)

val_df = val_df.reset_index(drop=True)
val_dataset = HFDataset.from_pandas(val_df, features=features)

# === LOAD IMAGE PROCESSOR ===
processor = ViTImageProcessor.from_pretrained(model_name)


# === PREPROCESS FUNCTION ===
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB").resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"][0],  # remove batch dim
        "label": example["label_id"],
    }


train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# === LOAD MODEL ===
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# === METRIC ===
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


# === TRAINING ARGUMENTS ===
args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_dir="./logs",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="tensorboard",
    fp16=True,  # Enable mixed precision training if supported
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # type: ignore
)

# === TRAIN ===
trainer.train()
metrics = trainer.evaluate(eval_dataset=val_dataset)  # type: ignore
print(metrics)
