class BaseReaderModelCard():

    def __init__(self, model_name: str = "", hyperparams = {}):
        self.model_name = model_name
        self.language = "en"
        self.dataset = "blablabla"
        self.hyperparams = hyperparams

        self.__MODEL_CARD__ = f"""
---
language: {self.language}
dataset: {self.dataset}
---

# {self.model_name}

## Hyperparameters

**batch_size:** {hyperparams["batch_size"]}     
**n_epochs:** {hyperparams["n_epochs"]}     
**learning_rate:** {hyperparams["learning_rate"]}       
**max_seq_len:** {hyperparams["max_seq_len"]}       
**dev_split:** {hyperparams["dev_split"]}       
**warmup_proportion:** {hyperparams["warmup_proportion"]}       

This model was trained with [Haystack](https://haystack.deeset.ai)

<div class="w-1/2 h-40 object-cover mb-2 rounded-lg flex items-center justify-center">
    <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored.png" class="w-40"/>
</div>

"""