from ...database.rag_classes import DocumentText, Document
from tqdm.auto import tqdm
import os
from ...utils.agent import get_Agent
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunkingRagIndexation:
    def __init__(
        self,
        data_manager,
        config_server: dict,
        dbs_name: list[str],
        data_folders_name: list[str],
        breakpoint_method: str = "percentile",
        threshold: int = 98,
    ) -> None:
        """
        Args:
            data_path (str): path of the folder containing texts you want to have a RAG on
            storage_path (str): path of the folder you want to store the data base and the vector base
            embedding_model (str): name of the model you want using to embed chunks (has to be pulled in Ollama)
            base_url (str): url used for Ollama
            db_name (str): name you want to give to the data base storing the chunks
            vb_name (str): name you want to give to the vector base storing the chunks' embeddings

        Returns:
            None
        """
        self.embedding_model = config_server["embedding_model"]
        self.agent = get_Agent(config_server)
        self.data_manager = data_manager
        self.breakpoint_method = breakpoint_method
        self.threshold = threshold

    def compute_breakpoints(self, similarities, method="percentile", threshold=90):
        """
        Computes chunking breakpoints based on similarity drops.

        Args:
        similarities (List[float]): List of similarity scores between sentences.
        method (str): 'percentile', 'standard_deviation', or 'interquartile'.
        threshold (float): Threshold value (percentile for 'percentile', std devs for 'standard_deviation').

        Returns:
        List[int]: Indices where chunk splits should occur.
        """
        try:
            # Determine the threshold value based on the selected method
            if method == "percentile":
                # Calculate the Xth percentile of the similarity scores
                threshold_value = np.percentile(similarities, threshold)
            elif method == "standard_deviation":
                # Calculate the mean and standard deviation of the similarity scores
                mean = np.mean(similarities)
                std_dev = np.std(similarities)
                # Set the threshold value to mean minus X standard deviations
                threshold_value = mean - (threshold * std_dev)
            elif method == "interquartile":
                # Calculate the first and third quartiles (Q1 and Q3)
                q1, q3 = np.percentile(similarities, [25, 75])
                # Set the threshold value using the IQR rule for outliers
                threshold_value = q1 - 1.5 * (q3 - q1)
            else:
                # Raise an error if an invalid method is provided
                raise ValueError(
                    "Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'."
                )

            # Identify indices where similarity drops below the threshold value
        except Exception:
            print("Couldn't compute breakpoints due to empty similarities")
        return [i for i, sim in enumerate(similarities) if sim < threshold_value]

    def get_embedding(self, input):
        """
        Returns the embedding of an str element or a list of elements

        Args:
            input (list[str] or str) : Char to be embedded

        Returns:
            embedding (list or list[list]) : embedded input
        """
        taille_batch = 500
        embeddings = None
        for i in range(0, len(input), taille_batch):
            results = self.agent.embeddings(texts=input[i:i + taille_batch],
                                            model=self.embedding_model)
            if embeddings is None:
                embeddings = results
            else:
                embeddings["embeddings"] = embeddings["embeddings"] + results["embeddings"]
                embeddings["nb_tokens"]+=np.sum(results["nb_tokens"])
        return embeddings
    
    def __batch_indexation__(self, doc_chunks, name_docs, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        elements = []
        for k, chunk in enumerate(doc_chunks):
            elements.append(chunk.replace("\n", "").replace("'", ""))

        tokens = 0
        taille_batch = 500
        for i in range(0, len(elements), taille_batch):
            tokens += np.sum(self.data_manager.add_str_batch_elements(
                                                                    elements=elements[i:i + taille_batch],
                                                                    docs_name=name_docs[i : i + taille_batch],
                                                                    path_docs=path_docs[i : i + taille_batch],
                                                                    display_message=False
            ))
        return tokens

    def __serial_indexation__(self, doc_chunks, name_docs, path_docs):
        """
        Adds a batch of chunks from doc_chunks to the indexation verctorbase
        Args:
            doc_chunks (list[str]) : Chunks to be indexed
            name_docs (list[str]) : Name of docs each chunk is from

        Returns
            None
        """
        tokens = 0
        for k, chunk in enumerate(doc_chunks):
            try:
                tokens += self.data_manager.add_str_elements(
                    elements=[chunk.replace("\n", "").replace("'", "")],
                    docs_name=[name_docs[k]],
                    path_docs=[path_docs[k]],
                    display_message=False,
                )
            except Exception:
                None
        return tokens        
        

    def run_pipeline(
        self, batch: bool = True, chunk_size: int = 150, overlap_size: int = 15
    ) -> None:
        """
        Split texts from the self.data_path, embed them and save them in a vector base.

        Args:
            chunk_size (int): Size of chunks for text splitting.
            overlap_size (int): Size of the overlap between chunks

        Returns:
            None
        """
        docs_already_processed = [res[0] for res in self.data_manager.query(Document.path)]
        to_process_norm = [Path(p).resolve().as_posix() for p in self.data_manager.get_list_path_documents()]
        docs_already_norm = [Path(p).resolve().as_posix() for p in docs_already_processed]

        docs_to_process = [
            doc
            for doc in to_process_norm
            if doc not in docs_already_norm
        ]

        self.data_manager.create_collection()
        with tqdm(docs_to_process) as progress_bar:

            for i, path_doc in enumerate(progress_bar):
                embedding_tokens, input_tokens, output_tokens = 0, 0, 0
                progress_bar.set_description(f"Embedding chunks - {path_doc}")

                doc = DocumentText(path=path_doc)
                text = doc.content
                res = []
                for i in range(0, len(text), chunk_size - overlap_size):
                    if i + chunk_size < len(text):
                        res.append(text[i : i + chunk_size])
                    else:
                        res.append(text[i:])
                # res = text.split(sep=".")

                sentences = [sentence for sentence in res if sentence]

                sentences_embeddings = self.get_embedding(sentences)
                embedding_tokens += np.sum(sentences_embeddings["nb_tokens"])
                similarities = []
                for i in range(len(sentences_embeddings["embeddings"]) - 1):
                    similarities.append(
                        float(
                            cosine_similarity(
                                [sentences_embeddings["embeddings"][i]],
                                [sentences_embeddings["embeddings"][i + 1]],
                            )
                        )
                    )
                breakpoints = self.compute_breakpoints(
                    similarities,
                    method=self.breakpoint_method,
                    threshold=self.threshold,
                )
                doc_chunks = []
                left = 0
                for bp in breakpoints:
                    doc_chunks.append(". ".join(sentences[left:bp]) + ".")
                    left = bp
                doc_chunks.append(". ".join(sentences[left:]) + ".")
                name_docs = [str(Path(path_doc).name) for i in range(len(doc_chunks))]
                path_docs = [str(Path(path_doc).parent) for i in range(len(doc_chunks))]

                if batch:
                    embedding_tokens += self.__batch_indexation__(doc_chunks=doc_chunks, 
                                                                  name_docs=name_docs,
                                                                  path_docs=path_docs
                    )

                else:
                    embedding_tokens += self.__serial_indexation__(doc_chunks=doc_chunks, 
                                                                   name_docs=name_docs,
                                                                   path_docs=path_docs
                    )

                if i == len(progress_bar) - 1:
                    progress_bar.set_description("Embedding chunks - ✅")
                new_doc = Document(name=str(Path(path_doc).name),
                                path=str(Path(path_doc)), 
                                   embedding_tokens=int(embedding_tokens),
                                input_tokens=int(input_tokens), output_tokens=int(output_tokens))
                self.data_manager.add_instance(instance=new_doc,
                                               path=str(Path(path_doc).parent))
