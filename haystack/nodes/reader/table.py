from abc import abstractmethod
from typing import List, Optional, Tuple, Dict, Union, Any

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
from statistics import mean
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from quantulum3 import parser
from transformers import (
    TapasTokenizer,
    TapasForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BatchEncoding,
    TapasModel,
    TapasConfig,
)
from transformers.models.tapas.modeling_tapas import TapasPreTrainedModel

from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.nodes.reader.base import BaseReader
from haystack.modeling.utils import initialize_device_settings
from haystack.utils.torch_utils import ensure_tensor_on_device

torch_scatter_installed = True
torch_scatter_wrong_version = False
try:
    import torch_scatter  # pylint: disable=unused-import
except ImportError:
    torch_scatter_installed = False
except OSError:
    torch_scatter_wrong_version = True


logger = logging.getLogger(__name__)


class TableReader(BaseReader):
    """
    Transformer-based model for extractive Question Answering on Tables with TaPas
    using the HuggingFace's transformers framework (https://github.com/huggingface/transformers).
    With this reader, you can directly get predictions via predict()

    Example:
    ```python
    from haystack import Document
    from haystack.nodes import TableReader
    import pandas as pd

    table_reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq")
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["57", "46", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
    }
    table = pd.DataFrame(data)
    document = Document(content=table, content_type="table")
    query = "When was DiCaprio born?"
    prediction = table_reader.predict(query=query, documents=[document])
    answer = prediction["answers"][0].answer  # "10 june 1996"
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "google/tapas-base-finetuned-wtq",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        top_k: int = 10,
        top_k_per_candidate: int = 3,
        return_no_answer: bool = False,
        max_seq_len: Optional[int] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Load a TableQA model from Transformers.
        Available models include:

        - ``'google/tapas-base-finetuned-wtq`'``
        - ``'google/tapas-base-finetuned-wikisql-supervised``'
        - ``'deepset/tapas-large-nq-hn-reader'``
        - ``'deepset/tapas-large-nq-reader'``

        See https://huggingface.co/models?pipeline_tag=table-question-answering
        for full list of available TableQA models.

        The nq-reader models are able to provide confidence scores, but cannot handle questions that need aggregation
        over multiple cells. The returned answers are sorted first by a general table score and then by answer span
        scores.
        All the other models can handle aggregation questions, but don't provide reasonable confidence scores.

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
        See https://huggingface.co/models?pipeline_tag=table-question-answering for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name,
                              or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU or CPU. Falls back on CPU if no GPU is available.
        :param top_k: The maximum number of answers to return.
        :param top_k_per_candidate: How many answers to extract for each candidate table that is coming from
                                    the retriever.
        :param return_no_answer: Whether to include no_answer predictions in the results.
                                 (Only applicable with nq-reader models.)
        :param max_seq_len: Max sequence length of one input table for the model. If the number of tokens of
                            query + table exceed max_seq_len, the table will be truncated by removing rows until the
                            input size fits the model. If not provided the max length is automatically determined by the
                            `model_max_length` variable of the tokenizer.
        :param use_auth_token:  The API token used to download private models from Huggingface.
                                If this parameter is set to `True`, then the token generated when running
                                `transformers-cli login` (stored in ~/.huggingface) will be used.
                                Additional information can be found here
                                https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        if not torch_scatter_installed:
            raise ImportError(
                "Please install torch_scatter to use TableReader. You can follow the instructions here: https://github.com/rusty1s/pytorch_scatter"
            )
        if torch_scatter_wrong_version:
            raise ImportError(
                "torch_scatter could not be loaded. This could be caused by a mismatch between your cuda version and the one used by torch_scatter."
                "Please try to reinstall torch-scatter. You can follow the instructions here: https://github.com/rusty1s/pytorch_scatter"
            )
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        config = TapasConfig.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        self.table_encoder: Union[_TapasEncoder, _TapasScoredEncoder]
        if config.architectures[0] == "TapasForQuestionAnswering":
            self.table_encoder = _TapasEncoder(
                device=self.devices[0],
                model_name_or_path=model_name_or_path,
                model_version=model_version,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                use_auth_token=use_auth_token,
            )
        elif config.architectures[0] == "TapasForScoredQA":
            self.table_encoder = _TapasScoredEncoder(
                device=self.devices[0],
                model_name_or_path=model_name_or_path,
                model_version=model_version,
                tokenizer=tokenizer,
                top_k_per_candidate=top_k_per_candidate,
                return_no_answer=return_no_answer,
                max_seq_len=max_seq_len,
                use_auth_token=use_auth_token,
            )
        else:
            logger.error(
                "Unrecognized model architecture %s. Only the architectures TapasForQuestionAnswering and TapasForScoredQA are supported",
                config.architectures[0],
            )
        self.table_encoder.model.to(str(self.devices[0]))

        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.max_seq_len = max_seq_len
        self.return_no_answer = return_no_answer

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Use loaded TableQA model to find answers for a query in the supplied list of Documents
        of content_type ``'table'``.

        Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
        WARNING: The answer scores are not reliable, as they are always extremely high, even if
                 a question cannot be answered by a given table.

        :param query: Query string
        :param documents: List of Document in which to search for the answer. Documents should be
                          of content_type ``'table'``.
        :param top_k: The maximum number of answers to return
        :param batch_size:
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k
        return self.table_encoder.predict(query=query, documents=documents, top_k=top_k, batch_size=1)

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Use loaded TableQA model to find answers for the supplied queries in the supplied Documents
        of content_type ``'table'``.

        Returns dictionary containing query and list of Answer objects sorted by (descending) score.
        WARNING: The answer scores are not reliable, as they are always extremely high, even if
        a question cannot be answered by a given table.

        - If you provide a list containing a single query...
            - ... and a single list of Documents, the query will be applied to each Document individually.
            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
              will be aggregated per Document list.

        - If you provide a list of multiple queries...
            - ... and a single list of Documents, each query will be applied to each Document individually.
            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
              and the Answers will be aggregated per query-Document pair.

        :param queries: Single query string or list of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
                          Documents should be of content_type ``'table'``.
        :param top_k: The maximum number of answers to return per query.
        :param batch_size:
        """
        if top_k is None:
            top_k = self.top_k

        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
        else:
            single_doc_list = False

        inputs = self._flatten_inputs(queries, documents)

        results: Dict = {"queries": queries, "answers": []}
        for query, docs in zip(inputs["queries"], inputs["docs"]):
            preds = self.table_encoder.predict(query=query, documents=docs, top_k=top_k, batch_size=batch_size)
            results["answers"].append(preds["answers"])

        # Group answers by question in case of multiple queries and single doc list
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results["answers"]) / len(queries))
            answers = []
            for i in range(0, len(results["answers"]), answers_per_query):
                answer_group = results["answers"][i : i + answers_per_query]
                answers.append(answer_group)
            results["answers"] = answers

        return results

    @staticmethod
    def _flatten_inputs(queries: List[str], documents: Union[List[Document], List[List[Document]]]) -> Dict[str, List]:
        """Flatten (and copy) the queries and documents into lists of equal length.

        - If you provide a list containing a single query...
            - ... and a single list of Documents, the query will be applied to each Document individually.
            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
              will be aggregated per Document list.

        - If you provide a list of multiple queries...
            - ... and a single list of Documents, each query will be applied to each Document individually.
            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
              and the Answers will be aggregated per query-Document pair.

        :param queries: Single query string or list of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
                          Documents should be of content_type ``'table'``.
        """
        # Docs case 1: single list of Documents -> apply each query to all Documents
        inputs = {"queries": [], "docs": []}
        if len(documents) > 0 and isinstance(documents[0], Document):
            for query in queries:
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    inputs["queries"].append(query)
                    inputs["docs"].append([doc])

        # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents, if queries
        # contains only one query, apply it to each list of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                inputs["queries"].append(query)
                inputs["docs"].append(cur_docs)
        return inputs


class _BaseTapasEncoder:
    @staticmethod
    def _calculate_answer_offsets(answer_coordinates: List[Tuple[int, int]], table: pd.DataFrame) -> List[Span]:
        """
        Calculates the answer cell offsets of the linearized table based on the answer cell coordinates.
        """
        answer_offsets = []
        n_rows, n_columns = table.shape
        for coord in answer_coordinates:
            answer_cell_offset = (coord[0] * n_columns) + coord[1]
            answer_offsets.append(Span(start=answer_cell_offset, end=answer_cell_offset + 1))
        return answer_offsets

    @staticmethod
    def _check_documents(documents: List[Document]) -> List[Document]:
        table_documents = []
        for document in documents:
            if document.content_type != "table":
                logger.warning("Skipping document with id '%s' in TableReader as it is not of type table.", document.id)
                continue

            table: pd.DataFrame = document.content
            if table.shape[0] == 0:
                logger.warning(
                    "Skipping document with id '%s' in TableReader as it does not contain any rows.", document.id
                )
                continue

            table_documents.append(document)
        return table_documents

    @staticmethod
    def _preprocess(query: str, table: pd.DataFrame, tokenizer, max_seq_len) -> BatchEncoding:
        """Tokenize the query and table."""
        model_inputs = tokenizer(
            table=table,
            queries=query,
            max_length=max_seq_len,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        return model_inputs

    @staticmethod
    def _new_preprocess(query: List[str], table: List[pd.DataFrame], tokenizer, max_seq_len) -> BatchEncoding:
        """Tokenize the query and table."""
        # TODO If number of rows in table is greater than max_row_id, they will be dropped by the tokenizer.
        # max_row_id = self.tokenizer.max_row_id
        # max_column_id = self.tokenizer.max_column_id
        #
        # TODO num_tokens measures max number of tokens in a single cell entry of the table
        # tokenized_table = self.tokenizer._tokenize_table(table)
        # _, _, num_tokens = self.tokenizer._get_table_boundaries(tokenized_table)
        model_inputs = tokenizer(
            table=table,
            queries=query,
            max_length=max_seq_len,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        return model_inputs

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: int) -> Dict:
        pass


class _TapasEncoder(_BaseTapasEncoder):
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str = "google/tapas-base-finetuned-wtq",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        self.model = TapasForQuestionAnswering.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        if tokenizer is None:
            self.tokenizer = TapasTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        else:
            self.tokenizer = TapasTokenizer.from_pretrained(tokenizer, use_auth_token=use_auth_token)
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.tokenizer.model_max_length
        self.device = device

    def _postprocess_tapas(self, predictions: Dict, documents: Document) -> Answer:

        table: pd.DataFrame = documents.content
        predicted_answer_coordinates = predictions["predicted_answer_coordinates"]
        predicted_aggregation_indices = predictions["predicted_aggregation_indices"]
        current_score = predictions["answer_scores"]

        # Get cell values
        current_answer_coordinates = predicted_answer_coordinates[0]
        current_answer_cells = []
        for coordinate in current_answer_coordinates:
            current_answer_cells.append(table.iat[coordinate])

        # Get aggregation operator
        if self.model.config.aggregation_labels is not None:
            current_aggregation_operator = self.model.config.aggregation_labels[predicted_aggregation_indices[0]]
        else:
            current_aggregation_operator = "NONE"

        if current_aggregation_operator == "NONE":
            answer_str = ", ".join(current_answer_cells)
        else:
            answer_str = self._aggregate_answers(current_aggregation_operator, current_answer_cells)

        answer_offsets = self._calculate_answer_offsets(current_answer_coordinates, table)

        answer = Answer(
            answer=answer_str,
            type="extractive",
            score=current_score,
            context=table,
            offsets_in_document=answer_offsets,
            offsets_in_context=answer_offsets,
            document_id=documents[0].id,
            meta={"aggregation_operator": current_aggregation_operator, "answer_cells": current_answer_cells},
        )
        return answer

    def _predict_tapas(self, inputs: BatchEncoding, document: Document, batch_size: int = 1) -> Answer:
        table: pd.DataFrame = document.content

        # Forward query and table through model and convert logits to predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        inputs.to("cpu")
        outputs_logits = outputs.logits.cpu()

        if self.model.config.num_aggregation_labels > 0:
            aggregation_logits = outputs.logits_aggregation.cpu()
            predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
                inputs, outputs_logits, logits_agg=aggregation_logits, cell_classification_threshold=0.5
            )
        else:
            predicted_answer_coordinates = self.tokenizer.convert_logits_to_predictions(
                inputs, outputs_logits, logits_agg=None, cell_classification_threshold=0.5
            )

        # Get cell values
        current_answer_coordinates = predicted_answer_coordinates[0]
        current_answer_cells = []
        for coordinate in current_answer_coordinates:
            current_answer_cells.append(table.iat[coordinate])

        # Get aggregation operator
        if self.model.config.aggregation_labels is not None:
            current_aggregation_operator = self.model.config.aggregation_labels[predicted_aggregation_indices[0]]
        else:
            current_aggregation_operator = "NONE"

        # Calculate answer score
        current_score = self._calculate_answer_score(outputs_logits, inputs, current_answer_coordinates)

        if current_aggregation_operator == "NONE":
            answer_str = ", ".join(current_answer_cells)
        else:
            answer_str = self._aggregate_answers(current_aggregation_operator, current_answer_cells)

        answer_offsets = self._calculate_answer_offsets(current_answer_coordinates, document.content)

        answer = Answer(
            answer=answer_str,
            type="extractive",
            score=current_score,
            context=document.content,
            offsets_in_document=answer_offsets,
            offsets_in_context=answer_offsets,
            document_id=document.id,
            meta={"aggregation_operator": current_aggregation_operator, "answer_cells": current_answer_cells},
        )
        return answer

    def _new_calculate_answer_score(
        self, logits: torch.Tensor, inputs: Dict, answer_coordinates: List[List[Tuple[int, int]]]
    ) -> List[np.float64]:
        # Calculate answer score
        # Values over 88.72284 will overflow when passed through exponential, so logits are truncated.
        logits[logits < -88.7] = -88.7
        token_probabilities = 1 / (1 + np.exp(-logits)) * inputs["attention_mask"]
        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        segment_ids = inputs["token_type_ids"][:, :, token_types.index("segment_ids")]
        row_ids = inputs["token_type_ids"][:, :, token_types.index("row_ids")]
        column_ids = inputs["token_type_ids"][:, :, token_types.index("column_ids")]

        answer_scores = []
        num_batch = logits.shape[0]
        assert (
            num_batch == len(answer_coordinates) == inputs["input_ids"].shape[0]
        ), "Ensure all inputs have same batch dimension"
        for i in range(num_batch):
            cell_coords_to_prob = self.tokenizer._get_mean_cell_probs(
                token_probabilities[i].tolist(), segment_ids[i].tolist(), row_ids[i].tolist(), column_ids[i].tolist()
            )
            # _get_mean_cell_probs seems to index cells by (col, row). DataFrames are, however, indexed by (row, col).
            all_cell_probabilities = {(row, col): prob for (col, row), prob in cell_coords_to_prob.items()}
            answer_cell_probabilities = [all_cell_probabilities[coord] for coord in answer_coordinates[i]]
            answer_scores.append(np.mean(answer_cell_probabilities))

        return answer_scores

    def _calculate_answer_score(
        self, logits: torch.Tensor, inputs: BatchEncoding, answer_coordinates: List[Tuple[int, int]]
    ) -> float:
        # Calculate answer score
        # Values over 88.72284 will overflow when passed through exponential, so logits are truncated.
        logits[logits < -88.7] = -88.7
        token_probabilities = 1 / (1 + np.exp(-logits)) * inputs.attention_mask
        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        segment_ids = inputs.token_type_ids[0, :, token_types.index("segment_ids")].tolist()
        column_ids = inputs.token_type_ids[0, :, token_types.index("column_ids")].tolist()
        row_ids = inputs.token_type_ids[0, :, token_types.index("row_ids")].tolist()
        all_cell_probabilities = self.tokenizer._get_mean_cell_probs(
            token_probabilities[0].tolist(), segment_ids, row_ids, column_ids
        )
        # _get_mean_cell_probs seems to index cells by (col, row). DataFrames are, however, indexed by (row, col).
        all_cell_probabilities = {(row, col): prob for (col, row), prob in all_cell_probabilities.items()}
        answer_cell_probabilities = [all_cell_probabilities[coord] for coord in answer_coordinates]

        return np.mean(answer_cell_probabilities)

    @staticmethod
    def _aggregate_answers(agg_operator: Literal["COUNT", "SUM", "AVERAGE"], answer_cells: List[str]) -> str:
        if agg_operator == "COUNT":
            return str(len(answer_cells))

        # No aggregation needed as only one cell selected as answer_cells
        if len(answer_cells) == 1:
            return answer_cells[0]
        # Return empty string if model did not select any cell as answer
        if len(answer_cells) == 0:
            return ""

        # Parse answer cells in order to aggregate numerical values
        parsed_answer_cells = [parser.parse(cell) for cell in answer_cells]
        # Check if all cells contain at least one numerical value and that all values share the same unit
        try:
            if all(parsed_answer_cells) and all(
                cell[0].unit.name == parsed_answer_cells[0][0].unit.name for cell in parsed_answer_cells
            ):
                numerical_values = [cell[0].value for cell in parsed_answer_cells]
                unit = parsed_answer_cells[0][0].unit.symbols[0] if parsed_answer_cells[0][0].unit.symbols else ""

                if agg_operator == "SUM":
                    answer_value = sum(numerical_values)
                elif agg_operator == "AVERAGE":
                    answer_value = mean(numerical_values)
                else:
                    raise ValueError("unknown aggregator")

                return f"{answer_value}{' ' + unit if unit else ''}"

        except ValueError as e:
            if "unknown aggregator" in str(e):
                pass

        # Not all selected answer cells contain a numerical value or answer cells don't share the same unit
        return f"{agg_operator} > {', '.join(answer_cells)}"

    def predict(self, query: str, documents: List[Document], top_k: int, batch_size: int = 1) -> Dict:
        answers = []
        table_documents = self._check_documents(documents)
        for document in table_documents:
            table: pd.DataFrame = document.content
            model_inputs = self._preprocess(query, table, self.tokenizer, self.max_seq_len)
            model_inputs.to(self.device)

            current_answer = self._predict_tapas(model_inputs, document, batch_size)
            answers.append(current_answer)

        answers = sorted(answers, reverse=True)
        results = {"query": query, "answers": answers[:top_k]}
        return results

    @staticmethod
    def _flatten_inputs(queries: List[str], documents: Union[List[Document], List[List[Document]]]) -> Dict[str, List]:
        """Flatten (and copy) the queries and documents into lists of equal length.

        - If you provide a list containing a single query...
            - ... and a single list of Documents, the query will be applied to each Document individually.
            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
              will be aggregated per Document list.

        - If you provide a list of multiple queries...
            - ... and a single list of Documents, each query will be applied to each Document individually.
            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
              and the Answers will be aggregated per query-Document pair.

        :param queries: Single query string or list of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
                          Documents should be of content_type ``'table'``.
        """
        # Docs case 1: single list of Documents -> apply each query to all Documents
        inputs = {"queries": [], "docs": []}
        if len(documents) > 0 and isinstance(documents[0], Document):
            for query in queries:
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    inputs["queries"].append(query)
                    inputs["docs"].append([doc])

        # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents, if queries
        # contains only one query, apply it to each list of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                inputs["queries"].append(query)
                inputs["docs"].append(cur_docs)
        return inputs

    def _forward(self, batch):
        outputs = {}
        model_outputs = self.model(**batch)
        if self.model.config.num_aggregation_labels > 0:
            aggregation_logits = model_outputs["logits_aggregation"]
            predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
                batch, model_outputs["logits"], logits_agg=aggregation_logits, cell_classification_threshold=0.5
            )
            outputs["predicted_answer_coordinates"] = predicted_answer_coordinates
            outputs["predicted_aggregation_indices"] = predicted_aggregation_indices
        else:
            predicted_answer_coordinates = self.tokenizer.convert_logits_to_predictions(
                batch, model_outputs["logits"], logits_agg=None, cell_classification_threshold=0.5
            )
            outputs["predicted_answer_coordinates"] = predicted_answer_coordinates
            outputs["predicted_aggregation_indices"] = None
        answer_scores = self._new_calculate_answer_score(model_outputs["logits"], batch, predicted_answer_coordinates)
        outputs["answer_scores"] = answer_scores
        return outputs

    def new_predict(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: int,
        batch_size: int = 1,
    ) -> Dict:

        if top_k is None:
            # Fixme
            # top_k = self.top_k
            top_k = 10

        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
        else:
            single_doc_list = False

        # TODO Add self._check_documents here
        inputs = self._flatten_inputs(queries, documents)

        assert len(inputs["queries"]) == len(
            inputs["docs"]
        ), "Length of queries and documents should be the same at this point"

        all_queries = []
        all_docs = []
        for i, query in enumerate(inputs["queries"]):
            for j, doc in enumerate(inputs["docs"][i]):
                all_queries.append(query)
                all_docs.append(doc)

        data = {"queries": all_queries, "table_documents": all_docs}

        # Setup up Pytorch Dataloader
        dataset = TableDataset(data, self.tokenizer)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)

        predictions = []
        for batch in tqdm(dataloader, disable=False, total=len(dataloader), desc="Processing Tables"):
            batch = ensure_tensor_on_device(batch, device=self.device)
            with torch.no_grad():
                model_outputs = self._forward(batch)
            predictions.append(model_outputs)

        predictions = self._flatten_predictions(predictions)

        assert len(all_docs) == len(predictions["predicted_answer_coordinates"]), "Should have a prediction per table."
        answers = []
        for i, doc in enumerate(all_docs):
            single_pred = {
                "predicted_answer_coordinates": predictions["predicted_answer_coordinates"][i],
                "predicted_aggregation_indices": predictions["predicted_aggregation_indices"][i],
                "answer_scores": predictions["answer_scores"][i],
            }
            answers.append(self._postprocess_tapas(single_pred, doc))

        # TODO Group answers correctly (depending on input).
        #      Answer sorting occurs on list of docs.
        #      1. Single query + List of docs:  -> [query], [[a1], [a2], [a3]]
        #      2. Single query + List of list of docs: -> [query], [[a1, a2], [a3]]
        #      3. List of queries + List of docs
        #      4. List of queries + List of list of docs
        import pdb

        pdb.set_trace()
        # TODO Put sort into _postprocess_tapas
        answers = sorted(answers, reverse=True)
        results = {"queries": queries, "answers": answers[:top_k]}
        return results

    def _flatten_predictions(self, predictions):
        flattened_predictions: Dict[str, Any] = {
            "predicted_answer_coordinates": [],
            "predicted_aggregation_indices": [],
            "answer_scores": [],
        }
        for pred in predictions:
            flattened_predictions["predicted_answer_coordinates"].extend(pred["predicted_answer_coordinates"])
            flattened_predictions["predicted_aggregation_indices"].extend(pred["predicted_aggregation_indices"])
            flattened_predictions["answer_scores"].extend(pred["answer_scores"])
        return flattened_predictions


class _TapasScoredEncoder(_BaseTapasEncoder):
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str = "deepset/tapas-large-nq-hn-reader",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        top_k_per_candidate: int = 3,
        return_no_answer: bool = False,
        max_seq_len: Optional[int] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        self.model = self._TapasForScoredQA.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        if tokenizer is None:
            self.tokenizer = TapasTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        else:
            self.tokenizer = TapasTokenizer.from_pretrained(tokenizer, use_auth_token=use_auth_token)
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.tokenizer.model_max_length
        self.device = device
        self.top_k_per_candidate = top_k_per_candidate
        self.return_no_answer = return_no_answer

    def _predict_tapas_scored(
        self, inputs: BatchEncoding, document: Document, batch_size: int = 1
    ) -> Tuple[List[Answer], float]:
        table: pd.DataFrame = document.content

        # Forward pass through model
        with torch.no_grad():
            outputs = self.model.tapas(**inputs)

        # Get general table score
        table_score = self.model.classifier(outputs.pooler_output)
        table_score_softmax = torch.nn.functional.softmax(table_score, dim=1)
        table_relevancy_prob = table_score_softmax[0][1].item()
        no_answer_score = table_score_softmax[0][0].item()

        # Get possible answer spans
        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]
        row_ids: List[int] = inputs.token_type_ids[:, :, token_types.index("row_ids")].tolist()[0]
        column_ids: List[int] = inputs.token_type_ids[:, :, token_types.index("column_ids")].tolist()[0]

        possible_answer_spans: List[
            Tuple[int, int, int, int]
        ] = []  # List of tuples: (row_idx, col_idx, start_token, end_token)
        current_start_token_idx = -1
        current_column_id = -1
        for token_idx, (row_id, column_id) in enumerate(zip(row_ids, column_ids)):
            if row_id == 0 or column_id == 0:
                continue
            # Beginning of new cell
            if column_id != current_column_id:
                if current_start_token_idx != -1:
                    possible_answer_spans.append(
                        (
                            row_ids[current_start_token_idx] - 1,
                            column_ids[current_start_token_idx] - 1,
                            current_start_token_idx,
                            token_idx - 1,
                        )
                    )
                current_start_token_idx = token_idx
                current_column_id = column_id
        possible_answer_spans.append(
            (
                row_ids[current_start_token_idx] - 1,
                column_ids[current_start_token_idx] - 1,
                current_start_token_idx,
                len(row_ids) - 1,
            )
        )

        # Concat logits of start token and end token of possible answer spans
        sequence_output = outputs.last_hidden_state
        concatenated_logits = []
        for possible_span in possible_answer_spans:
            start_token_logits = sequence_output[0, possible_span[2], :]
            end_token_logits = sequence_output[0, possible_span[3], :]
            concatenated_logits.append(torch.cat((start_token_logits, end_token_logits)))
        concatenated_logit_tensors = torch.unsqueeze(torch.stack(concatenated_logits), dim=0)

        # Calculate score for each possible span
        span_logits = (
            torch.einsum("bsj,j->bs", concatenated_logit_tensors, self.model.span_output_weights)
            + self.model.span_output_bias
        )
        span_logits_softmax = torch.nn.functional.softmax(span_logits, dim=1)

        top_k_answer_spans = torch.topk(span_logits[0], min(self.top_k_per_candidate, len(possible_answer_spans)))

        answers = []
        for answer_span_idx in top_k_answer_spans.indices:
            current_answer_span = possible_answer_spans[answer_span_idx]
            answer_str = table.iat[current_answer_span[:2]]
            answer_offsets = self._calculate_answer_offsets([current_answer_span[:2]], document.content)
            # As the general table score is more important for the final score, it is double weighted.
            current_score = ((2 * table_relevancy_prob) + span_logits_softmax[0, answer_span_idx].item()) / 3

            answers.append(
                Answer(
                    answer=answer_str,
                    type="extractive",
                    score=current_score,
                    context=document.content,
                    offsets_in_document=answer_offsets,
                    offsets_in_context=answer_offsets,
                    document_id=document.id,
                    meta={"aggregation_operator": "NONE", "answer_cells": table.iat[current_answer_span[:2]]},
                )
            )

        return answers, no_answer_score

    def predict(self, query: str, documents: List[Document], top_k: int, batch_size: int = 1) -> Dict:
        answers = []
        no_answer_score = 1.0
        table_documents = self._check_documents(documents)
        for document in table_documents:
            table: pd.DataFrame = document.content
            model_inputs = self._preprocess(query, table, self.tokenizer, self.max_seq_len)
            model_inputs.to(self.device)

            current_answers, current_no_answer_score = self._predict_tapas_scored(model_inputs, document, batch_size)
            answers.extend(current_answers)
            if current_no_answer_score < no_answer_score:
                no_answer_score = current_no_answer_score

        if self.return_no_answer:
            answers.append(
                Answer(
                    answer="",
                    type="extractive",
                    score=no_answer_score,
                    context=None,
                    offsets_in_context=[Span(start=0, end=0)],
                    offsets_in_document=[Span(start=0, end=0)],
                    document_id=None,
                    meta=None,
                )
            )

        answers = sorted(answers, reverse=True)
        results = {"query": query, "answers": answers[:top_k]}
        return results

    class _TapasForScoredQA(TapasPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)

            # base model
            self.tapas = TapasModel(config)

            # dropout (only used when training)
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

            # answer selection head
            self.span_output_weights = torch.nn.Parameter(torch.zeros(2 * config.hidden_size))
            self.span_output_bias = torch.nn.Parameter(torch.zeros([]))

            # table scoring head
            self.classifier = torch.nn.Linear(config.hidden_size, 2)

            # Initialize weights
            self.init_weights()


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        question = self.data["queries"][idx]
        table = self.data["table_documents"][idx].content
        encoding = self.tokenizer(
            table=table, queries=question, truncation=True, padding="max_length", return_tensors="pt"
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding

    def __len__(self):
        return len(self.data["queries"])


class RCIReader(BaseReader):
    """
    Table Reader model based on Glass et al. (2021)'s Row-Column-Intersection model.
    See the original paper for more details:
    Glass, Michael, et al. (2021): "Capturing Row and Column Semantics in Transformer Based Question Answering over Tables"
    (https://aclanthology.org/2021.naacl-main.96/)

    Each row and each column is given a score with regard to the query by two separate models. The score of each cell
    is then calculated as the sum of the corresponding row score and column score. Accordingly, the predicted answer is
    the cell with the highest score.

    Pros and Cons of RCIReader compared to TableReader:
    + Provides meaningful confidence scores
    + Allows larger tables as input
    - Does not support aggregation over table cells
    - Slower
    """

    def __init__(
        self,
        row_model_name_or_path: str = "michaelrglass/albert-base-rci-wikisql-row",
        column_model_name_or_path: str = "michaelrglass/albert-base-rci-wikisql-col",
        row_model_version: Optional[str] = None,
        column_model_version: Optional[str] = None,
        row_tokenizer: Optional[str] = None,
        column_tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        top_k: int = 10,
        max_seq_len: int = 256,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Load an RCI model from Transformers.
        Available models include:

        - ``'michaelrglass/albert-base-rci-wikisql-row'`` + ``'michaelrglass/albert-base-rci-wikisql-col'``
        - ``'michaelrglass/albert-base-rci-wtq-row'`` + ``'michaelrglass/albert-base-rci-wtq-col'``



        :param row_model_name_or_path: Directory of a saved row scoring model or the name of a public model
        :param column_model_name_or_path: Directory of a saved column scoring model or the name of a public model
        :param row_model_version: The version of row model to use from the HuggingFace model hub.
                                  Can be tag name, branch name, or commit hash.
        :param column_model_version: The version of column model to use from the HuggingFace model hub.
                                     Can be tag name, branch name, or commit hash.
        :param row_tokenizer: Name of the tokenizer for the row model (usually the same as model)
        :param column_tokenizer: Name of the tokenizer for the column model (usually the same as model)
        :param use_gpu: Whether to use GPU or CPU. Falls back on CPU if no GPU is available.
        :param top_k: The maximum number of answers to return
        :param max_seq_len: Max sequence length of one input table for the model. If the number of tokens of
                            query + table exceed max_seq_len, the table will be truncated by removing rows until the
                            input size fits the model.
        :param use_auth_token:  The API token used to download private models from Huggingface.
                                If this parameter is set to `True`, then the token generated when running
                                `transformers-cli login` (stored in ~/.huggingface) will be used.
                                Additional information can be found here
                                https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        self.row_model = AutoModelForSequenceClassification.from_pretrained(
            row_model_name_or_path, revision=row_model_version, use_auth_token=use_auth_token
        )
        self.column_model = AutoModelForSequenceClassification.from_pretrained(
            row_model_name_or_path, revision=column_model_version, use_auth_token=use_auth_token
        )
        self.row_model.to(str(self.devices[0]))
        self.column_model.to(str(self.devices[0]))

        if row_tokenizer is None:
            try:
                self.row_tokenizer = AutoTokenizer.from_pretrained(
                    row_model_name_or_path, use_auth_token=use_auth_token
                )
            # The existing RCI models on the model hub don't come with tokenizer vocab files.
            except TypeError:
                self.row_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_auth_token=use_auth_token)
        else:
            self.row_tokenizer = AutoTokenizer.from_pretrained(row_tokenizer, use_auth_token=use_auth_token)

        if column_tokenizer is None:
            try:
                self.column_tokenizer = AutoTokenizer.from_pretrained(
                    column_model_name_or_path, use_auth_token=use_auth_token
                )
            # The existing RCI models on the model hub don't come with tokenizer vocab files.
            except TypeError:
                self.column_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_auth_token=use_auth_token)
        else:
            self.column_tokenizer = AutoTokenizer.from_pretrained(column_tokenizer, use_auth_token=use_auth_token)

        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.return_no_answers = False

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Use loaded RCI models to find answers for a query in the supplied list of Documents
        of content_type ``'table'``.

        Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
        The existing RCI models on the HF model hub don"t allow aggregation, therefore, the answer will always be
        composed of a single cell.

        :param query: Query string
        :param documents: List of Document in which to search for the answer. Documents should be
                          of content_type ``'table'``.
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k

        answers = []
        for document in documents:
            if document.content_type != "table":
                logger.warning("Skipping document with id '%s' in RCIReader as it is not of type table.", document.id)
                continue

            table: pd.DataFrame = document.content
            if table.shape[0] == 0:
                logger.warning(
                    "Skipping document with id '%s' in RCIReader as it does not contain any rows.", document.id
                )
                continue
            table = table.astype(str)
            # Create row and column representations
            row_reps, column_reps = self._create_row_column_representations(table)

            # Get row logits
            row_inputs = self.row_tokenizer(
                [(query, row_rep) for row_rep in row_reps],
                max_length=self.max_seq_len,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True,
            )
            row_inputs.to(self.devices[0])
            row_logits = self.row_model(**row_inputs)[0].detach().cpu().numpy()[:, 1]

            # Get column logits
            column_inputs = self.column_tokenizer(
                [(query, column_rep) for column_rep in column_reps],
                max_length=self.max_seq_len,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True,
            )
            column_inputs.to(self.devices[0])
            column_logits = self.column_model(**column_inputs)[0].detach().cpu().numpy()[:, 1]

            # Calculate cell scores
            current_answers: List[Answer] = []
            cell_scores_table: List[List[float]] = []
            for row_idx, row_score in enumerate(row_logits):
                cell_scores_table.append([])
                for col_idx, col_score in enumerate(column_logits):
                    current_cell_score = float(row_score + col_score)
                    cell_scores_table[-1].append(current_cell_score)

                    answer_str = table.iloc[row_idx, col_idx]
                    answer_offsets = self._calculate_answer_offsets(row_idx, col_idx, table)
                    current_answers.append(
                        Answer(
                            answer=answer_str,
                            type="extractive",
                            score=current_cell_score,
                            context=table,
                            offsets_in_document=[answer_offsets],
                            offsets_in_context=[answer_offsets],
                            document_id=document.id,
                        )
                    )

            # Add cell scores to Answers' meta to be able to use as heatmap
            for answer in current_answers:
                answer.meta = {"table_scores": cell_scores_table}
            answers.extend(current_answers)

        # Sort answers by score and select top-k answers
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        results = {"query": query, "answers": answers}

        return results

    @staticmethod
    def _create_row_column_representations(table: pd.DataFrame) -> Tuple[List[str], List[str]]:
        row_reps = []
        column_reps = []
        columns = table.columns

        for idx, row in table.iterrows():
            current_row_rep = " * ".join([header + " : " + cell for header, cell in zip(columns, row)])
            row_reps.append(current_row_rep)

        for col_name in columns:
            current_column_rep = f"{col_name} * "
            current_column_rep += " * ".join(table[col_name])
            column_reps.append(current_column_rep)

        return row_reps, column_reps

    @staticmethod
    def _calculate_answer_offsets(row_idx, column_index, table) -> Span:
        n_rows, n_columns = table.shape
        answer_cell_offset = (row_idx * n_columns) + column_index

        return Span(start=answer_cell_offset, end=answer_cell_offset + 1)

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        # TODO: Currently, just calls naively predict method, so there is room for improvement.

        results: Dict = {"queries": queries, "answers": []}

        single_doc_list = False
        # Docs case 1: single list of Documents -> apply each query to all Documents
        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
            for query in queries:
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    preds = self.predict(query=query, documents=[doc], top_k=top_k)
                    results["answers"].append(preds["answers"])

        # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents, if queries
        # contains only one query, apply it to each list of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                preds = self.predict(query=query, documents=cur_docs, top_k=top_k)
                results["answers"].append(preds["answers"])

        # Group answers by question in case of multiple queries and single doc list
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results["answers"]) / len(queries))
            answers = []
            for i in range(0, len(results["answers"]), answers_per_query):
                answer_group = results["answers"][i : i + answers_per_query]
                answers.append(answer_group)
            results["answers"] = answers

        return results
