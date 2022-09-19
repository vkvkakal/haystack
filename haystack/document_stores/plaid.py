from abc import abstractmethod
import itertools
import logging
import os
from pathlib import Path
import random
import time
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm
import torch
import ujson

from haystack.document_stores.sql import SQLDocumentStore
from haystack.modeling.model.colbert.infra.config.config import ColBERTConfig
from haystack.modeling.model.colbert.indexing.index_saver import IndexSaver
from haystack.modeling.model.colbert.indexing.collection_indexer import compute_faiss_kmeans
from haystack.modeling.model.colbert.indexing.codecs.residual import ResidualCodec
from haystack.modeling.model.colbert.indexing.utils import optimize_ivf
from haystack.modeling.model.colbert.search.index_storage import IndexScorer
from haystack.schema import Document
from haystack.document_stores.base import get_batches_from_generator
from haystack.errors import DocumentStoreError

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever


logger = logging.getLogger(__name__)


class PlaidIndex:
    def __init__(self, index_path: Union[str, Path]):
        self.index_path = Path(index_path)
        self.ntotal = 0

    def reset(self):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        if directory.exists():
            for filename in sorted(os.listdir(directory)):
                filename = os.path.join(directory, filename)

                delete = filename.endswith(".json")
                delete = delete and ("metadata" in filename or "doclen" in filename or "plan" in filename)
                delete = delete or filename.endswith(".pt")

                if delete:
                    deleted.append(filename)

            for filename in deleted:
                os.remove(filename)


class PlaidDocumentStore(SQLDocumentStore):
    def __init__(
        self,
        sql_url: str = "sqlite:///plaid_document_store.db",
        embedding_dim: int = 128,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "dot_product",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        plaid_index_path: Union[str, Path] = None,
        plaid_config_path: Union[str, Path] = None,
        isolation_level: str = None,
        validate_index_sync: bool = True,
        use_gpu: bool = False,
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param embedding_dim: The embedding vector size. Default: 768.
        :param return_embedding: To return document embedding. Unlike other document stores, FAISS will return normalized embeddings
        :param index: Name of index in document store to use.
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default since it is
                   more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence-Transformer model.
                   In both cases, the returned values in Document.score are normalized to be in range [0,1]:
                   For `dot_product`: expit(np.asarray(raw_score / 100))
                   FOr `cosine`: (raw_score + 1) / 2
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param plaid_index_path: Stored PLAID index file. Can be created via calling `save()`.
            If specified no other params besides plaid_config_path must be specified.
        :param plaid_config_path: Stored PLAID initial configuration parameters.
            Can be created via calling `save()`
        :param isolation_level: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)
        :param validate_index_sync: Whether to check that the document count equals the embedding count at initialization time
        """
        self.similarity = similarity
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.progress_bar = progress_bar
        self.use_gpu = use_gpu

        self.plaid_indexes: Dict[str, PlaidIndex] = {}
        if plaid_index_path:
            self.plaid_indexes[index] = PlaidIndex(plaid_index_path)

        super().__init__(
            url=sql_url, index=index, duplicate_documents=duplicate_documents, isolation_level=isolation_level
        )

        # if validate_index_sync:
        #     self._validate_index_sync()

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

        :return: None
        """
        if headers:
            raise NotImplementedError("PlaidDocumentStore does not support headers.")

        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents
        )
        if len(document_objects) > 0:
            add_vectors = False if document_objects[0].embedding is None else True

            if self.duplicate_documents == "overwrite" and add_vectors:
                logger.warning(
                    "You have to provide `duplicate_documents = 'overwrite'` arg and "
                    "`PlaidDocumentStore` does not support update in existing `plaid_index`.\n"
                    "Please call `update_embeddings` method to repopulate `plaid_index`"
                )

            vector_id = self.plaid_indexes[index].ntotal
            with tqdm(
                total=len(document_objects), disable=not self.progress_bar, position=0, desc="Writing Documents"
            ) as progress_bar:
                for i in range(0, len(document_objects), batch_size):
                    # if add_vectors:
                    #     embeddings = [doc.embedding for doc in document_objects[i : i + batch_size]]
                    #     embeddings_to_index = np.array(embeddings, dtype="float32")

                    #     if self.similarity == "cosine":
                    #         self.normalize_embedding(embeddings_to_index)

                    #     self.faiss_indexes[index].add(embeddings_to_index)

                    docs_to_write_in_sql = []
                    for doc in document_objects[i : i + batch_size]:
                        meta = doc.meta
                        if add_vectors:
                            meta["vector_id"] = vector_id
                            vector_id += 1
                        docs_to_write_in_sql.append(doc)

                    super(PlaidDocumentStore, self).write_documents(
                        docs_to_write_in_sql,
                        index=index,
                        duplicate_documents=duplicate_documents,
                        batch_size=batch_size,
                    )
                    progress_bar.update(batch_size)
            progress_bar.close()

    def update_embeddings(
        self,
        retriever: "BaseRetriever",
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in FAISSDocStore
        batch_size: int = 10_000,
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: Index name for which embeddings are to be updated. If set to None, the default self.index is used.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        index = index or self.index

        if update_existing_embeddings is True:
            if filters is None:
                self.plaid_indexes[index].reset()
                self.reset_vector_ids(index)
            else:
                raise Exception("update_existing_embeddings=True is not supported with filters.")

        if not self.plaid_indexes.get(index):
            raise ValueError("Couldn't find a PLAID index. Try to init the PlaidDocumentStore() again ...")

        document_count = self.get_document_count(index=index)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info(f"Updating embeddings for {document_count} docs...")
        vector_id = self.plaid_indexes[index].ntotal

        documents = list(
            self._query(
                index=index,
                vector_ids=None,
                batch_size=batch_size,
                filters=filters,
                only_documents_without_embedding=not update_existing_embeddings,
            )
        )

        index_path = self.plaid_indexes[index].index_path
        config: ColBERTConfig = retriever.embedding_encoder.colbert_config
        config.configure(
            doc_maxlen=300, nbits=2, index_name=index, resume=False, partitions=None, index_path=str(index_path)
        )

        if not os.path.exists(index_path):
            os.makedirs(index_path)

        saver = IndexSaver(config)

        num_passages = len(documents)
        chunk_size = min(25_000, 1 + num_passages // 1)  # min(25_000, 1 + len(self) // Run().nranks)
        num_chunks = int(np.ceil(num_passages / chunk_size))

        # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
        # Keep in mind that, say, 15% still means at least 100k.
        # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
        # Then we subsample the vectors to 100 * num_partitions

        typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
        k = int(16 * np.sqrt(typical_doclen * num_passages))
        # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
        k = min(1 + k, num_passages)
        sampled_pids = set(random.sample(range(num_passages), k))
        local_sample = [doc for pid, doc in enumerate(documents) if pid in sampled_pids]
        local_sample_embs, doclens = retriever.embed_documents(local_sample)  # type: ignore

        if torch.cuda.is_available():
            num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
            # torch.distributed.all_reduce(num_sample_embs)

            avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
            avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()
            # torch.distributed.all_reduce(avg_doclen_est)

            nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cuda()
            # torch.distributed.all_reduce(nonzero_ranks)
        else:
            if torch.distributed.is_initialized():
                num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cpu()
                torch.distributed.all_reduce(num_sample_embs)

                avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
                avg_doclen_est = torch.tensor([avg_doclen_est]).cpu()
                torch.distributed.all_reduce(avg_doclen_est)

                nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cpu()
                torch.distributed.all_reduce(nonzero_ranks)
            else:
                num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cpu()

                avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
                avg_doclen_est = torch.tensor([avg_doclen_est]).cpu()

                nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cpu()

        avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        torch.save(local_sample_embs.half(), os.path.join(index_path, f"sample.pt"))

        # Select the number of partitions
        num_embeddings_est = num_passages * avg_doclen_est
        num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(num_embeddings_est))))

        plan_path = os.path.join(index_path, "plan.json")
        with open(plan_path, "w") as f:
            d = {"config": config.export()}
            d["num_chunks"] = num_chunks
            d["num_partitions"] = num_partitions
            d["num_embeddings_est"] = num_embeddings_est
            d["avg_doclen_est"] = avg_doclen_est
            f.write(ujson.dumps(d, indent=4) + "\n")

        # TODO: Allocate a float16 array. Load the samples from disk, copy to array.
        sample = torch.empty(num_sample_embs, config.dim, dtype=torch.float16)
        offset = 0
        # for r in range(self.nranks):
        sub_sample_path = os.path.join(index_path, f"sample.pt")
        sub_sample = torch.load(sub_sample_path)
        os.remove(sub_sample_path)

        endpos = offset + sub_sample.size(0)
        sample[offset:endpos] = sub_sample
        offset = endpos

        assert endpos == sample.size(0), (endpos, sample.size())

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample = sample[torch.randperm(sample.size(0))]

        heldout_fraction = 0.05
        heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
        sample, sample_heldout = sample.split([sample.size(0) - heldout_size, heldout_size], dim=0)

        torch.cuda.empty_cache()

        args_ = [config.dim, num_partitions, config.kmeans_niters]
        args_ = args_ + [[[sample]]]
        centroids = compute_faiss_kmeans(*args_)

        centroids = torch.nn.functional.normalize(centroids, dim=-1)

        if self.use_gpu:
            centroids = centroids.half()
        else:
            centroids = centroids.float()

        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, sample_heldout, config)
        codec = ResidualCodec(
            config=config,
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )
        saver.save_codec(codec)

        with saver.thread():
            batches = self.enumerate_batches(documents, chunk_size)
            for chunk_idx, offset, passages in tqdm(batches):
                if config.resume and saver.check_chunk_exists(chunk_idx):
                    continue
                embs, doclens = retriever.embed_documents(passages)

                if embs.shape[1] != self.embedding_dim:
                    raise RuntimeError(
                        f"Embedding dimensions of the model ({embs.shape[1]}) doesn't match the embedding dimensions of the document store ({self.embedding_dim}). Please reinitiate FAISSDocumentStore() with arg embedding_dim={embs.shape[1]}."
                    )

                if self.use_gpu:
                    assert embs.dtype == torch.float16
                else:
                    assert embs.dtype == torch.float32
                    embs = embs.half()

                saver.save_chunk(chunk_idx, offset, embs, doclens)

        passage_offset = 0
        embedding_offset = 0

        embedding_offsets = []

        for chunk_idx in range(num_chunks):
            metadata_path = os.path.join(index_path, f"{chunk_idx}.metadata.json")

            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)

                chunk_metadata["embedding_offset"] = embedding_offset
                embedding_offsets.append(embedding_offset)

                assert chunk_metadata["passage_offset"] == passage_offset, (chunk_idx, passage_offset, chunk_metadata)

                passage_offset += chunk_metadata["num_passages"]
                embedding_offset += chunk_metadata["num_embeddings"]

            with open(metadata_path, "w") as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + "\n")

        num_embeddings = embedding_offset
        assert len(embedding_offsets) == num_chunks

        if len(documents) != passage_offset:
            raise DocumentStoreError(
                "The number of embeddings does not match the number of documents "
                f"({passage_offset} != {len(documents)})"
            )

        # build ivf
        codes = torch.empty(num_embeddings)

        for chunk_idx in tqdm(range(num_chunks)):
            offset = embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(index_path, chunk_idx)
            codes[offset : offset + chunk_codes.size(0)] = chunk_codes

        assert offset + chunk_codes.size(0) == codes.size(0), (offset, chunk_codes.size(0), codes.size())

        codes = codes.sort()
        ivf, values = codes.indices, codes.values
        partitions, ivf_lengths = values.unique_consecutive(return_counts=True)

        # All partitions should be non-empty. (We can use torch.histc otherwise.)
        assert partitions.size(0) == num_partitions, (partitions.size(), num_partitions)

        _, _ = optimize_ivf(ivf, ivf_lengths, index_path)

        metadata_path = os.path.join(index_path, "metadata.json")
        with open(metadata_path, "w") as f:
            d = {"config": config.export()}
            d["num_chunks"] = num_chunks
            d["num_partitions"] = num_partitions
            d["num_embeddings"] = num_embeddings
            d["avg_doclen"] = num_embeddings / len(documents)

            f.write(ujson.dumps(d, indent=4) + "\n")

        vector_id_map = {}
        for doc in documents:
            vector_id_map[str(doc.id)] = str(vector_id)
            vector_id += 1
        self.update_vector_ids(vector_id_map, index=index)

    def enumerate_batches(self, documents, chunksize):
        offset = 0
        iterator = iter(documents)

        for chunk_idx, owner in enumerate(itertools.cycle(range(1))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == 0:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return

    def _compute_avg_residual(self, centroids, heldout, config):
        compressor = ResidualCodec(config=config, centroids=centroids, avg_residual=None)

        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device="cuda" if self.use_gpu else "cpu")
        heldout_reconstruct = compressor.lookup_centroids(
            heldout_reconstruct, out_device="cuda" if self.use_gpu else "cpu"
        )
        if self.use_gpu:
            heldout_avg_residual = heldout.cuda() - heldout_reconstruct
        else:
            heldout_avg_residual = heldout - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        print([round(x, 3) for x in avg_residual.squeeze().tolist()])

        num_options = 2**config.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if headers:
            raise NotImplementedError("FAISSDocumentStore does not support headers.")

        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                        __Example__:
                        ```python
                        filters = {
                            "$and": {
                                "type": {"$eq": "article"},
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": {"$in": ["economy", "politics"]},
                                    "publisher": {"$eq": "nytimes"}
                                }
                            }
                        }
                        ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if headers:
            raise NotImplementedError("PlaidDocumentStore does not support headers.")

        index = index or self.index
        documents = super(PlaidDocumentStore, self).get_all_documents_generator(
            index=index, filters=filters, batch_size=batch_size, return_embedding=False
        )
        if return_embedding is None:
            return_embedding = self.return_embedding

        for doc in documents:
            if return_embedding:
                pass
                # TODO
                # if doc.meta and doc.meta.get("vector_id") is not None:
                #     doc.embedding = self.faiss_indexes[index].reconstruct(int(doc.meta["vector_id"]))
            yield doc

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("FAISSDocumentStore does not support headers.")

        if filters:
            logger.warning("Query filters are not implemented for the FAISSDocumentStore.")

        index = index or self.index
        if not self.plaid_indexes.get(index):
            raise Exception(f"Index named '{index}' does not exists. Use 'update_embeddings()' to create an index.")

        index_path = self.plaid_indexes[index].index_path
        config: ColBERTConfig = ColBERTConfig.load_from_index(index_path)
        if top_k <= 10:
            if config.ncells is None:
                config.configure(ncells=1)
            if config.centroid_score_threshold is None:
                config.configure(centroid_score_threshold=0.5)
            if config.ndocs is None:
                config.configure(ndocs=256)
        elif top_k <= 100:
            if config.ncells is None:
                config.configure(ncells=2)
            if config.centroid_score_threshold is None:
                config.configure(centroid_score_threshold=0.45)
            if config.ndocs is None:
                config.configure(ndocs=1024)
        else:
            if config.ncells is None:
                config.configure(ncells=4)
            if config.centroid_score_threshold is None:
                config.configure(centroid_score_threshold=0.4)
            if config.ndocs is None:
                config.configure(ndocs=max(top_k * 4, 4096))

        ranker = IndexScorer(index_path, self.use_gpu)
        # add dimension
        query_emb = query_emb.unsqueeze(0)
        pids, scores = ranker.rank(config, query_emb)
        pids, scores = pids[:top_k], scores[:top_k]
        pids = [str(pid) for pid in pids]

        documents = self.get_documents_by_vector_ids(pids, index=index)

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(pids, scores)}
        for doc in documents:
            score = scores_for_vector_ids[doc.meta["vector_id"]]
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            doc.score = score

        return documents

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete all documents from the document store.
        """
        if headers:
            raise NotImplementedError("FAISSDocumentStore does not support headers.")

        logger.warning(
            """DEPRECATION WARNINGS: 
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        # TODO: delete from plaid index

        super().delete_documents(index=index, ids=ids, filters=filters)

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        if self.plaid_indexes.get(index):
            self.plaid_indexes[index].reset()
            del self.plaid_indexes[index]
        super().delete_index(index)

    def _create_document_field_map(self) -> Dict:
        return {}

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("PlaidDocumentStore does not support headers.")

        index = index or self.index
        documents = super(PlaidDocumentStore, self).get_documents_by_id(ids=ids, index=index, batch_size=batch_size)
        if self.return_embedding:
            for doc in documents:
                if doc.meta and doc.meta.get("vector_id") is not None:
                    doc.embedding = None  # TODO
        return documents
