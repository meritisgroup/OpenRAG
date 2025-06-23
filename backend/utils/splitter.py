import re
import time
import nltk
nltk.download('punkt_tab')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base_classes import Splitter


def get_splitter(type_text_splitter, agent=None, embedding_model=None):
    if type_text_splitter == "TextSplitter":
        splitter = TextSplitter()
    elif type_text_splitter == "Recursive_TextSplitter":
        splitter = Recursive_TextSplitter()
    elif type_text_splitter == "Semantic_TextSplitter":
        splitter = Semantic_TextSplitter(agent=agent,
                                         embedding_model=embedding_model)
    return splitter


class TextSplitter(Splitter):

    def __init__(self):
        pass

    def pieces_from_text(
        self, text: str, characters: list[str]
    ) -> tuple[list[str], list[str | None]]:
        """
        Splits the input text into pieces, separated by the specified characters.
        """
        pattern = f"[{''.join(map(re.escape, characters))}]"

        pieces = re.split(pattern, text)
        separators = re.findall(pattern, text)
        separators.append(None)

        real_pieces = [piece for piece in pieces if piece]
        real_separators = [
            separator for separator, piece in zip(separators, pieces) if piece
        ]

        final_pieces = [
            piece + sep
            for piece, sep in zip(real_pieces, real_separators)
            if sep is not None
        ]

        return final_pieces

    def split_text(
        self, text: str, chunk_size: int = 500, overlap: bool = True
    ) -> list[str]:
        """
        Add text's pieces together to have chunk with approximatively the rigth size
        """
        pieces = self.pieces_from_text(text=text, characters=[".", "!", "?", "\n"])
        index = 0
        chunks = []

        start_time = time.time()

        for _ in range(len(pieces)):
            start_index = index
            chunk = ""
            while len(chunk) < chunk_size and index < len(pieces):
                chunk += pieces[index]
                index += 1

                if time.time() - start_time > 3:
                    index += 1

            if chunk not in chunks:
                chunks.append(chunk)

            else:
                index += 1

            if index >= len(pieces):
                chunks.append("".join(pieces[start_index:]))
                break

            if overlap:
                index -= 1

        final_chunks = []
        for chunk in chunks:
            if chunk not in final_chunks:
                final_chunks.append(chunk)

        return final_chunks


class Recursive_TextSplitter(Splitter):
    def __init__(self):
        pass

    def split_markdown(self, text, max_chunk_size=1, overlap=0):

        sections = re.split(r"(?=\n#{1,6} )", text)
        chunks = []
        current_chunk = ""

        for section in sections:
            if len(current_chunk) + len(section) <= max_chunk_size:
                current_chunk += section
            else:
                chunks.append(current_chunk.strip())
                current_chunk = section

        if current_chunk:
            chunks.append(current_chunk.strip())

        if overlap > 0 and len(chunks) > 1:
            overlapping_chunks = []
            for i in range(len(chunks)):
                if i > 0:
                    overlap_text = chunks[i - 1][-overlap:]
                    overlapping_chunks.append(overlap_text + "\n" + chunks[i])
                else:
                    overlapping_chunks.append(chunks[i])
            return overlapping_chunks

        return chunks

    def __split(self, text, max_chunk_size=500, overlap=0):
        paragraphs = text.strip().split("\n\n")
        chunks = []

        temp_paragraph = ""
        for paragraph in paragraphs:
            if len(temp_paragraph) + len(paragraph) <= max_chunk_size:
                temp_paragraph += "\n\n" + paragraph
            else:
                chunks.append(temp_paragraph)
                temp_paragraph = ""
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                temp_chunk = ""

                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= max_chunk_size:
                        temp_chunk += sentence + " "
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + " "

                if len(temp_chunk) > 0:
                    chunks.append(temp_chunk.strip())

        return chunks
    
    def split_text(
        self, text: str, chunk_size: int = 500, overlap: bool = False
    ) -> list[str]:
        chunks = self.split_markdown(text=text)
        final_chunks = []
        for i in range(len(chunks)):
            if len(chunks[i]) > 0:
                if len(chunks[i]) < chunk_size:
                    final_chunks.append(chunks[i])
                else:
                    temp = self.__split(text=chunks[i], max_chunk_size=chunk_size)
                    for j in range(len(temp)):
                        if len(temp[j]) > 0:
                            final_chunks.append(temp[j])

        final_chunks = self.break_chunks(chunks=final_chunks, 
                                         max_size_chunk=chunk_size)
        return final_chunks


class Semantic_TextSplitter(Splitter):

    def __init__(self, agent, embedding_model):
        self.agent = agent
        self.model = embedding_model

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
            if method == "percentile":
                threshold_value = np.percentile(similarities, threshold)
            elif method == "standard_deviation":
                mean = np.mean(similarities)
                std_dev = np.std(similarities)
                threshold_value = mean - (threshold * std_dev)
            elif method == "interquartile":
                q1, q3 = np.percentile(similarities, [25, 75])
                threshold_value = q1 - 1.5 * (q3 - q1)
            else:
                raise ValueError(
                    "Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'."
                )

            # Identify indices where similarity drops below the threshold value
        except Exception as e:
            print("Couldn't compute breakpoints due to empty similarities")
        return [i for i, sim in enumerate(similarities) if sim < threshold_value]


    def split_text(self, text, chunk_size: int = 500, overlap: bool = False):
        """
        Segmente un texte en chunks sémantiques en fonction de la similarité cosinus entre phrases.

        - text: Texte à segmenter.
        - threshold: Seuil de découpe (plus bas = segments plus grands, plus haut = plus petits).

        Retourne : Liste de chunks sémantiques.
        """
        # Découper le texte en phrases
        sentences = nltk.sent_tokenize(text)
        sentences = self.break_chunks(chunks=sentences,
                                      max_size_chunk=chunk_size)

        taille_batch = 500
        sentence_embeddings = None
        for i in range(0, len(sentences), taille_batch):
            results = self.agent.embeddings(texts=sentences[i:i + taille_batch],
                                            model=self.model)
            if sentence_embeddings is None:
                sentence_embeddings = results
            else:
                sentence_embeddings["embeddings"] = sentence_embeddings["embeddings"] + results["embeddings"]
                sentence_embeddings["nb_tokens"]+=np.sum(results["nb_tokens"])

        
        sentence_embeddings = sentence_embeddings["embeddings"]
        similarities = [
            float(
                cosine_similarity(
                    [sentence_embeddings[i]], [sentence_embeddings[i + 1]]
                )[0][0]
            )
            for i in range(len(sentence_embeddings) - 1)
        ]

        breakpoints = self.compute_breakpoints(
                    similarities,
                    threshold=0.5,
                    method="standard_deviation"
                )
        current_chunk = []
        left = 0
        for bp in breakpoints:
            current_chunk.append(". ".join(sentences[left:bp]) + ".")
            left = bp
        current_chunk.append(". ".join(sentences[left:]) + ".")
        chunks = self.break_chunks(chunks=current_chunk,
                                   max_size_chunk=chunk_size)
        return chunks
