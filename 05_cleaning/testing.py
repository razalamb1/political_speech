# %%
import pandas as pd
import numpy as np

# %%
democrats = pd.read_parquet("../10_datasets/democrats")
republicans = pd.read_parquet("../10_datasets/neutral.parquet")
neutral = pd.read_parquet("../10_datasets/republican.parquet")
df = pd.concat([democrats, republicans, neutral]).reset_index(drop=True)

# %%
cat_maps = {
    "democrats": "democrat",
    "Republican": "republican",
    "NeutralPolitics": "neutral",
}
df["category"] = df["subreddit"].map(cat_maps)
df["text"] = df["total_post"]
df = df[["text", "category"]]


# %%
labels = {
    "republican": 0,
    "democrat": 1,
    "neutral": 2,
}
df["category"] = df["category"].map(labels)
df.head()

# %%
df = df[df["category"] < 2]

# %%
import torch
import tez
import transformers
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics


# %%
class BERTDataset:
    def __init__(self, texts, targets, max_len=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=False
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }
        return resp


class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.unsqueeze(1))

    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        return {"accuracy": metrics.accuracy_score(targets, outputs)}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x, targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, None, {}


# %%
def train_model(fold):
    np.random.seed(112)
    df_train, df_valid, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    # %%

    # %%
    train_dataset = BERTDataset(df_train.text.values, df_train.category.values)
    valid_dataset = BERTDataset(df_valid.text.values, df_train.category.values)
    n_train_steps = int(len(train_dataset) / 32 * 10)
    model = TextModel(num_classes=1, num_train_steps=n_train_steps)
    es = tez.callbacks.EarlyStopping(
        monitor="valid_loss", patience=3, model_path="model.bin"
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        device="cpu",
        epochs=10,
        train_bs=32,
        callbacks=[es],
    )


if __name__ == "__main__":
    train_model(fold=0)
