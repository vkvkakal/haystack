class BaseReaderModelCard():

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.language = "en"
        self.dataset = "blablabla"

        self.__MODEL_CARD__ = f"""
---
language: {self.language}
dataset: {self.dataset}
---

# {self.model_name}

This model was trained with [Haystack](https://haystack.deeset.ai)

<div class="w-full h-40 object-cover mb-2 rounded-lg flex items-center justify-center">
    <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored.png" class="w-40"/>
</div>
"""