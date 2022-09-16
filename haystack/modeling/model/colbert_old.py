# copied from https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/hf_colbert.py
import datetime
import os
from typing import Any
import torch
import torch.nn as nn
import ujson
import dataclasses
from dataclasses import dataclass, fields

from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

# from haystack.modeling.model.colbert.utils.utils import torch_load_dnn


@dataclass
class DefaultVal:
    val: Any


@dataclass
class CoreConfig:
    def __post_init__(self):
        """
        Source: https://stackoverflow.com/a/58081120/1493011
        """

        self.assigned = {}

        for field in fields(self):
            field_val = getattr(self, field.name)

            if isinstance(field_val, DefaultVal) or field_val is None:
                setattr(self, field.name, field.default.val)

            if not isinstance(field_val, DefaultVal):
                self.assigned[field.name] = True

    def assign_defaults(self):
        for field in fields(self):
            setattr(self, field.name, field.default.val)
            self.assigned[field.name] = True

    def configure(self, ignore_unrecognized=True, **kw_args):
        ignored = set()

        for key, value in kw_args.items():
            self.set(key, value, ignore_unrecognized) or ignored.update({key})

        return ignored

        """
        # TODO: Take a config object, not kw_args.

        for key in config.assigned:
            value = getattr(config, key)
        """

    def set(self, key, value, ignore_unrecognized=False):
        if hasattr(self, key):
            setattr(self, key, value)
            self.assigned[key] = True
            return True

        if not ignore_unrecognized:
            raise Exception(f"Unrecognized key `{key}` for {type(self)}")

    def help(self):
        print(ujson.dumps(dataclasses.asdict(self), indent=4))

    def __export_value(self, v):
        v = v.provenance() if hasattr(v, "provenance") else v

        if isinstance(v, list) and len(v) > 100:
            v = (f"list with {len(v)} elements starting with...", v[:3])

        if isinstance(v, dict) and len(v) > 100:
            v = (f"dict with {len(v)} keys starting with...", list(v.keys())[:3])

        return v

    def export(self):
        d = dataclasses.asdict(self)

        for k, v in d.items():
            d[k] = self.__export_value(v)

        return d


@dataclass
class BaseConfig(CoreConfig):
    @classmethod
    def from_existing(cls, *sources):
        kw_args = {}

        for source in sources:
            if source is None:
                continue

            local_kw_args = dataclasses.asdict(source)
            local_kw_args = {k: local_kw_args[k] for k in source.assigned}
            kw_args = {**kw_args, **local_kw_args}

        obj = cls(**kw_args)

        return obj

    @classmethod
    def from_deprecated_args(cls, args):
        obj = cls()
        ignored = obj.configure(ignore_unrecognized=True, **args)

        return obj, ignored

    @classmethod
    def from_path(cls, name):
        with open(name) as f:
            args = ujson.load(f)

            if "config" in args:
                args = args["config"]

        return cls.from_deprecated_args(args)  # the new, non-deprecated version functions the same at this level.

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        # if checkpoint_path.endswith('.dnn'):
        #     dnn = torch_load_dnn(checkpoint_path)
        #     config, _ = cls.from_deprecated_args(dnn.get('arguments', {}))

        #     # TODO: FIXME: Decide if the line below will have any unintended consequences. We don't want to overwrite those!
        #     config.set('checkpoint', checkpoint_path)

        #     return config

        loaded_config_path = os.path.join(checkpoint_path, "artifact.metadata")
        if os.path.exists(loaded_config_path):
            loaded_config, _ = cls.from_path(loaded_config_path)
            loaded_config.set("checkpoint", checkpoint_path)

            return loaded_config

        return None  # can happen if checkpoint_path is something like 'bert-base-uncased'

    @classmethod
    def load_from_index(cls, index_path):
        # FIXME: We should start here with initial_config = ColBERTConfig(config, Run().config).
        # This should allow us to say initial_config.index_root. Then, below, set config = Config(..., initial_c)

        # default_index_root = os.path.join(Run().root, Run().experiment, 'indexes/')
        # index_path = os.path.join(default_index_root, index_path)

        # CONSIDER: No more plan/metadata.json. Only metadata.json to avoid weird issues when loading an index.

        try:
            metadata_path = os.path.join(index_path, "metadata.json")
            loaded_config, _ = cls.from_path(metadata_path)
        except:
            metadata_path = os.path.join(index_path, "plan.json")
            loaded_config, _ = cls.from_path(metadata_path)

        return loaded_config

    def save(self, path, overwrite=False):
        assert overwrite or not os.path.exists(path), path

        with open(path, "w") as f:
            args = self.export()  # dict(self.__config)
            args["meta"] = {}  # TODO get_metadata_only()
            args["meta"]["version"] = "colbert-v0.4"
            # TODO: Add git_status details.. It can't be too large! It should be a path that Runs() saves on exit, maybe!

            f.write(ujson.dumps(args, indent=4) + "\n")

    def save_for_checkpoint(self, checkpoint_path):
        assert not checkpoint_path.endswith(
            ".dnn"
        ), f"{checkpoint_path}: We reserve *.dnn names for the deprecated checkpoint format."

        output_config_path = os.path.join(checkpoint_path, "artifact.metadata")
        self.save(output_config_path, overwrite=True)


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


@dataclass
class RunSettings:
    """
    The defaults here have a special status in Run(), which initially calls assign_defaults(),
    so these aren't soft defaults in that specific context.
    """

    overwrite: bool = DefaultVal(False)  # type: ignore

    root: str = DefaultVal(os.path.join(os.getcwd(), "experiments"))  # type: ignore
    experiment: str = DefaultVal("default")  # type: ignore

    index_root: str = DefaultVal(None)  # type: ignore
    name: str = DefaultVal(timestamp(daydir=True))  # type: ignore

    rank: int = DefaultVal(0)  # type: ignore
    nranks: int = DefaultVal(1)  # type: ignore
    amp: bool = DefaultVal(True)  # type: ignore

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = DefaultVal(total_visible_gpus)  # type: ignore

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(",")

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(device_idx in range(0, self.total_visible_gpus) for device_idx in value), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, "indexes/")

    @property
    def script_name_(self):
        if "__file__" in dir(__main__):  # pylint: disable=undefined-variable
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)  # pylint: disable=undefined-variable
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd) :]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath) :]
                except:
                    pass

            assert script_path.endswith(".py")
            script_name = script_path.replace("/", ".").strip(".")[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return "none"

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class ResourceSettings:
    checkpoint: str = DefaultVal(None)  # type: ignore
    triples: str = DefaultVal(None)  # type: ignore
    collection: str = DefaultVal(None)  # type: ignore
    queries: str = DefaultVal(None)  # type: ignore
    index_name: str = DefaultVal(None)  # type: ignore


@dataclass
class DocSettings:
    dim: int = DefaultVal(128)  # type: ignore
    doc_maxlen: int = DefaultVal(220)  # type: ignore
    mask_punctuation: bool = DefaultVal(True)  # type: ignore


@dataclass
class QuerySettings:
    query_maxlen: int = DefaultVal(32)  # type: ignore
    attend_to_mask_tokens: bool = DefaultVal(False)  # type: ignore
    interaction: str = DefaultVal("colbert")  # type: ignore


@dataclass
class TrainingSettings:
    similarity: str = DefaultVal("cosine")  # type: ignore

    bsize: int = DefaultVal(32)  # type: ignore

    accumsteps: int = DefaultVal(1)  # type: ignore

    lr: float = DefaultVal(3e-06)  # type: ignore

    maxsteps: int = DefaultVal(500_000)  # type: ignore

    save_every: int = DefaultVal(None)  # type: ignore

    resume: bool = DefaultVal(False)  # type: ignore

    ## NEW:
    warmup: int = DefaultVal(None)  # type: ignore

    warmup_bert: int = DefaultVal(None)  # type: ignore

    relu: bool = DefaultVal(False)  # type: ignore

    nway: int = DefaultVal(2)  # type: ignore

    use_ib_negatives: bool = DefaultVal(False)  # type: ignore

    reranker: bool = DefaultVal(False)  # type: ignore

    distillation_alpha: float = DefaultVal(1.0)  # type: ignore

    ignore_scores: bool = DefaultVal(False)  # type: ignore


@dataclass
class IndexingSettings:
    index_path: str = DefaultVal(None)  # type: ignore

    nbits: int = DefaultVal(1)  # type: ignore

    kmeans_niters: int = DefaultVal(4)  # type: ignore

    resume: bool = DefaultVal(False)  # type: ignore

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)


@dataclass
class SearchSettings:
    ncells: int = DefaultVal(None)  # type: ignore
    centroid_score_threshold: float = DefaultVal(None)  # type: ignore
    ndocs: int = DefaultVal(None)  # type: ignore


@dataclass
class RunConfig(BaseConfig, RunSettings):
    pass


@dataclass
class ColBERTConfig(
    RunSettings,
    ResourceSettings,
    DocSettings,
    QuerySettings,
    TrainingSettings,
    IndexingSettings,
    SearchSettings,
    BaseConfig,
):
    pass


class HF_ColBERT(BertPreTrainedModel):
    """
    Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.

    This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """

    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config, colbert_config):
        super().__init__(config)

        self.dim = colbert_config.dim
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        # if colbert_config.relu:
        #     self.score_scaler = nn.Linear(1, 1)

        self.init_weights()

        # if colbert_config.relu:
        #     self.score_scaler.weight.data.fill_(1.0)
        #     self.score_scaler.bias.data.fill_(-8.0)

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        # if name_or_path.endswith('.dnn'):
        #     dnn = torch_load_dnn(name_or_path)
        #     base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

        #     obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
        #     obj.base = base

        #     return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
        obj.base = name_or_path

        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        # if name_or_path.endswith('.dnn'):
        #     dnn = torch_load_dnn(name_or_path)
        #     base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

        #     obj = AutoTokenizer.from_pretrained(base)
        #     obj.base = base

        #     return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj


"""
TODO: It's easy to write a class generator that takes "name_or_path" and loads AutoConfig to check the Architecture's
      name, finds that name's *PreTrainedModel and *Model in dir(transformers), and then basically repeats the above.

      It's easy for the BaseColBERT class to instantiate things from there.
"""


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset : offset + bsize], mask[offset : offset + bsize]))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


class QueryTokenizer:
    def __init__(self, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)

        self.config = config
        self.query_maxlen = config.query_maxlen
        self.background_maxlen = 512 - self.query_maxlen + 1  # FIXME: Make this configurable

        self.Q_marker_token, self.Q_marker_token_id = "[Q]", self.tok.convert_tokens_to_ids("[unused0]")
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [Q] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text, padding="max_length", truncation=True, return_tensors="pt", max_length=self.query_maxlen
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if context is not None:
            assert len(context) == len(batch_text), (len(context), len(batch_text))

            obj_2 = self.tok(
                context, padding="longest", truncation=True, return_tensors="pt", max_length=self.background_maxlen
            )

            ids_2, mask_2 = obj_2["input_ids"][:, 1:], obj_2["attention_mask"][:, 1:]  # Skip the first [SEP]

            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)

        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        if self.used is False:
            self.used = True

            firstbg = (context is None) or context[0]

            print()
            print("#> QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            print(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
            print(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
            print()

        return ids, mask


class DocTokenizer:
    def __init__(self, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)

        self.config = config
        self.doc_maxlen = config.doc_maxlen

        self.D_marker_token, self.D_marker_token_id = "[D]", self.tok.convert_tokens_to_ids("[unused1]")
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        assert self.D_marker_token_id == 2

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [D] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text, padding="longest", truncation="longest_first", return_tensors="pt", max_length=self.doc_maxlen
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
