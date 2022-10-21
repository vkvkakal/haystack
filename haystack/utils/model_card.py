import json
import os
from haystack.model_card.reader import BaseReaderModelCard


def create_model_card(model, repo_id, path):
    model_name = repo_id.split('/')[-1]
    print("My Model Card new: ", model_name)
    hyperparams = {
        "batch_size": model.inferencer.batch_size,
        "n_epochs": model._trainer.epochs,
        "learning_rate": model.learning_rate,
        "max_seq_len": model.max_seq_len,
        "dev_split": model.dev_split,
        "warmup_proportion": model.warmup_proportion
    }

    print(f"Hyperparams: {hyperparams}")

    model_card = BaseReaderModelCard(model_name=model_name, hyperparams=hyperparams).__MODEL_CARD__

    for param, val in hyperparams.items():
        model_card.replace(f"${param}$", val)

    with open(os.path.join(path, "README.md"), "w", encoding='utf8') as fOut:
        fOut.write(model_card.strip())
