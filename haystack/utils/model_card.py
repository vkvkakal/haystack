
import json
import os
from haystack.model_card.reader import BaseReaderModelCard

def create_model_card(model, repo_id, path):
    model_name = repo_id.split('/')[-1]
    print("My Model Card new: ", model_name)
    model_card = BaseReaderModelCard(model_name=model_name).__MODEL_CARD__

    with open(os.path.join(path, "README.md"), "w", encoding='utf8') as fOut:
            fOut.write(model_card.strip())
