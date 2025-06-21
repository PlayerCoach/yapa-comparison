import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset as HFDataset, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split
import evaluate
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,  # type: ignore
    Trainer,  # type: ignore
)
from transformers.trainer_callback import EarlyStoppingCallback  # type: ignore


class Machine:
    def __init__(
        self,
        csv_path="spectrogram_dataset.csv",
        num_epochs=5,
        batch_size=64,
        model_name="google/vit-base-patch16-224-in21k",
        learning_rate=2e-4,
        max_grad_norm=1.0,
        weight_decay=0.05,
    ):
        # === CONFIG ===
        self.csv_path = csv_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay

        # === LOAD DATA ===
        df = pd.read_csv(self.csv_path)
        self.label_names = sorted(df["label"].unique())
        self.label2id = {name: i for i, name in enumerate(self.label_names)}
        self.id2label = {i: name for name, i in self.label2id.items()}
        df["label_id"] = df["label"].map(self.label2id)

        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )
        features = Features(
            {
                "image_path": Value("string"),
                "label": ClassLabel(names=self.label_names),
                "label_id": Value("int64"),
            }
        )
        self.train_dataset = HFDataset.from_pandas(
            train_df.reset_index(drop=True), features=features
        )
        self.val_dataset = HFDataset.from_pandas(
            val_df.reset_index(drop=True), features=features
        )

        # === LOAD IMAGE PROCESSOR ===
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)

        # === PREPROCESS FUNCTION ===
        def preprocess(example):
            image = Image.open(example["image_path"]).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "label": example["label_id"],
            }

        self.train_dataset = self.train_dataset.map(preprocess)
        self.val_dataset = self.val_dataset.map(preprocess)

        # === LOAD MODEL ===
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_names),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # === METRIC ===
        self.metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return self.metric.compute(predictions=preds, references=labels)

        # === TRAINING ARGS ===
        self.args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="tensorboard",
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            warmup_ratio=0.05,
            logging_dir="./logs",
        )

        # === TRAINER ===
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,  # type: ignore
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        )

    def learn(self):
        self.trainer.train()

    def evaluate(self):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        # Predict
        preds = self.trainer.predict(self.val_dataset)  # type: ignore
        y_pred = np.argmax(preds.predictions, axis=1)
        y_true = preds.label_ids

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)  # type: ignore
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.label_names
        )
        disp.plot(xticks_rotation=45, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()
        save_path = "confusion_matrix.png"
        disp.figure_.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        return self.trainer.evaluate()


if __name__ == "__main__":
    machine = Machine()
    machine.learn()
    results = machine.evaluate()
    print("Evaluation results:", results)
