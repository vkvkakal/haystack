from typing import List, Optional, Tuple, Dict, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
from statistics import mean
import torch
import numpy as np
import pandas as pd
from quantulum3 import parser
from transformers import (
    TapasTokenizer,
    TapasForQuestionAnswering,
    BatchEncoding,
    TapasModel,
    TapasConfig,
    TableQuestionAnsweringPipeline,
)
from transformers.models.tapas.modeling_tapas import TapasPreTrainedModel

from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.nodes.reader.base import BaseReader
from haystack.modeling.utils import initialize_device_settings

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
        max_seq_len: int = 512,
        batch_size: int = 1,
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
                            input size fits the model.
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
                batch_size=batch_size,
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
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k
        return self.table_encoder.predict(query=query, documents=documents, top_k=top_k)

    def predict_batch(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int] = None
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
        """
        if top_k is None:
            top_k = self.top_k

        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
        else:
            single_doc_list = False

        inputs = _flatten_inputs(queries, documents)

        results: Dict = {"queries": queries, "answers": []}
        for query, docs in zip(inputs["queries"], inputs["docs"]):
            preds = self.table_encoder.predict(query=query, documents=docs, top_k=top_k)
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


class _TableQuestionAnsweringPipeline(TableQuestionAnsweringPipeline):
    def _calculate_answer_score(
        self,
        logits: torch.Tensor,
        inputs: Dict,
        answer_coordinates: List[List[Tuple[int, int]]],
        temperature: Optional[float] = None,
    ) -> List[np.float64]:
        if temperature is not None:
            token_probabilities = torch.sigmoid(logits * temperature) * inputs["attention_mask"]
        else:
            token_probabilities = torch.sigmoid(logits) * inputs["attention_mask"]

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
            if len(answer_coordinates[i]) == 0:
                answer_scores.append(None)
            else:
                cell_coords_to_prob = self.tokenizer._get_mean_cell_probs(
                    token_probabilities[i].tolist(),
                    segment_ids[i].tolist(),
                    row_ids[i].tolist(),
                    column_ids[i].tolist(),
                )
                # _get_mean_cell_probs seems to index cells by (col, row). DataFrames are, however, indexed by (row, col).
                all_cell_probabilities = {(row, col): prob for (col, row), prob in cell_coords_to_prob.items()}
                answer_cell_probabilities = [all_cell_probabilities[coord] for coord in answer_coordinates[i]]
                answer_scores.append(np.mean(answer_cell_probabilities))

        return answer_scores

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

    def postprocess(self, model_outputs):
        """Modified from HuggingFace"""
        inputs = model_outputs["model_inputs"]
        table = model_outputs["table"]
        outputs = model_outputs["outputs"]
        if self.type == "tapas":
            if self.aggregate:
                logits, logits_agg = outputs[:2]
                copy_logits = logits.clone()
                predictions = self.tokenizer.convert_logits_to_predictions(
                    inputs, copy_logits, logits_agg, cell_classification_threshold=0.5
                )
                answer_coordinates_batch, agg_predictions = predictions
                aggregators = {i: self.model.config.aggregation_labels[pred] for i, pred in enumerate(agg_predictions)}
            else:
                logits = outputs[0]
                copy_logits = logits.clone()
                predictions = self.tokenizer.convert_logits_to_predictions(
                    inputs, copy_logits, cell_classification_threshold=0.5
                )
                answer_coordinates_batch = predictions[0]
                aggregators = {}
            answer_scores = self._calculate_answer_score(logits, inputs, answer_coordinates_batch)
            answers = []
            for index, coordinates in enumerate(answer_coordinates_batch):
                if len(coordinates) != 0:
                    cells = [table.iat[coordinate] for coordinate in coordinates]
                    aggregator = aggregators.get(index, "")  # type: ignore
                    if aggregator == "NONE":
                        answer_str = ", ".join(cells)
                    else:
                        answer_str = self._aggregate_answers(aggregator, cells)
                    answer_offsets = _calculate_answer_offsets(coordinates, table)
                    current_score = answer_scores[index]
                    answer = Answer(
                        answer=answer_str,
                        type="extractive",
                        score=current_score,
                        context=table,
                        offsets_in_document=answer_offsets,
                        offsets_in_context=answer_offsets,
                        meta={
                            "aggregation_operator": aggregator,
                            "answer_cells": [table.iat[coordinate] for coordinate in coordinates],
                        },
                    )
                    answers.append(answer)
                else:
                    answers.append(
                        Answer(
                            answer="",
                            type="extractive",
                            score=0.0,
                            context=None,
                            offsets_in_context=[Span(start=0, end=0)],
                            offsets_in_document=[Span(start=0, end=0)],
                            document_id=None,
                            meta=None,
                        )
                    )
        else:
            raise NotImplementedError("Only TAPAS models are supported")

        return answers


class _TapasEncoder:
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str = "google/tapas-base-finetuned-wtq",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        max_seq_len: int = 512,
        batch_size: int = 1,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        self.model = TapasForQuestionAnswering.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        if tokenizer is None:
            self.tokenizer = TapasTokenizer.from_pretrained(
                model_name_or_path, use_auth_token=use_auth_token, model_max_length=max_seq_len
            )
        else:
            self.tokenizer = TapasTokenizer.from_pretrained(
                tokenizer, use_auth_token=use_auth_token, model_max_length=max_seq_len
            )
        self.max_seq_len = max_seq_len
        self.device = device
        self.pipeline = _TableQuestionAnsweringPipeline(
            task="table-question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            batch_size=batch_size,
            device=self.device,
        )

    def predict(self, query: str, documents: List[Document], top_k: int) -> Dict:
        answers = []
        table_documents = _check_documents(documents)
        for document in table_documents:
            table: pd.DataFrame = document.content
            pipeline_inputs = {
                "query": query,
                "table": table,
                "sequential": False,
                "padding": False,
                "truncation": False,
            }
            current_answer = self.pipeline(pipeline_inputs)[0]
            current_answer.document_id = document.id
            answers.append(current_answer)

        answers = sorted(answers, reverse=True)
        results = {"query": query, "answers": answers[:top_k]}
        return results


class _TapasScoredEncoder:
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str = "deepset/tapas-large-nq-hn-reader",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        top_k_per_candidate: int = 3,
        return_no_answer: bool = False,
        max_seq_len: int = 512,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        self.model = self._TapasForScoredQA.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        if tokenizer is None:
            self.tokenizer = TapasTokenizer.from_pretrained(
                model_name_or_path, use_auth_token=use_auth_token, model_max_length=max_seq_len
            )
        else:
            self.tokenizer = TapasTokenizer.from_pretrained(
                tokenizer, use_auth_token=use_auth_token, model_max_length=max_seq_len
            )
        self.max_seq_len = max_seq_len
        self.device = device
        self.top_k_per_candidate = top_k_per_candidate
        self.return_no_answer = return_no_answer

    def _predict_tapas_scored(self, inputs: BatchEncoding, document: Document) -> Tuple[List[Answer], float]:
        table: pd.DataFrame = document.content

        # Forward pass through model
        with torch.no_grad():
            outputs = self.model.tapas(**inputs)
            table_score = self.model.classifier(outputs.pooler_output)
            table_score_softmax = torch.nn.functional.softmax(table_score, dim=1)

        # Get general table score
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
            answer_offsets = _calculate_answer_offsets([current_answer_span[:2]], document.content)
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

    @staticmethod
    def _preprocess(query: str, table: pd.DataFrame, tokenizer) -> BatchEncoding:
        """Tokenize the query and table."""
        model_inputs = tokenizer(
            table=table, queries=query, max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True
        )
        return model_inputs

    def predict(self, query: str, documents: List[Document], top_k: int) -> Dict:
        answers = []
        no_answer_score = 1.0
        table_documents = _check_documents(documents)
        for document in table_documents:
            table: pd.DataFrame = document.content
            model_inputs = self._preprocess(query, table, self.tokenizer)
            model_inputs.to(self.device)

            current_answers, current_no_answer_score = self._predict_tapas_scored(model_inputs, document)
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
