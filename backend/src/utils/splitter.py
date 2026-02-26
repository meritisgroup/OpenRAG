import re
import time
import nltk
nltk.download('punkt_tab')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base_classes import Splitter
from typing import Any, TypedDict, Union

class LineType(TypedDict):
    metadata: dict[str, str]
    content: str

class HeaderType(TypedDict):
    level: int
    name: str
    data: str

def extract_images_and_tables_blocks(section: str) -> list[str]:
    parts = []
    last_idx = 0
    image_block_pattern = re.compile('<img>(.*?)</img>', re.DOTALL)
    image_blocks = []
    for match in image_block_pattern.finditer(section):
        (start, end) = match.span()
        if start > last_idx:
            before = section[last_idx:start]
            image_blocks.append(('text', before))
        image_block = match.group(1)
        image_blocks.append(('image', image_block))
        last_idx = end
    if last_idx < len(section):
        image_blocks.append(('text', section[last_idx:]))
    table_pattern = re.compile('(<table[\\s\\S]*?</table>)', re.IGNORECASE)
    for (block_type, content) in image_blocks:
        if block_type == 'image':
            parts.append(content.strip())
        else:
            last_idx = 0
            for match in table_pattern.finditer(content):
                (start, end) = match.span()
                if start > last_idx:
                    before = content[last_idx:start].strip()
                    if before:
                        parts.append(before)
                table_block = match.group(1).strip()
                parts.append(table_block)
                last_idx = end
            if last_idx < len(content):
                after = content[last_idx:].strip()
                if after:
                    parts.append(after)
    return parts

def get_splitter(type_text_splitter, data_preprocessing, agent=None, embedding_model=None):
    if type_text_splitter == 'TextSplitter':
        splitter = TextSplitter()
    elif type_text_splitter == 'Recursive_TextSplitter':
        splitter = Recursive_TextSplitter()
    elif type_text_splitter == 'Semantic_TextSplitter':
        splitter = Semantic_TextSplitter(agent=agent, embedding_model=embedding_model)
    if data_preprocessing == 'md_with_images' or data_preprocessing == 'md_without_images':
        splitter = MarkdownHeaderTextSplitter(strip_headers=False, splitter=splitter)
    return splitter

class MarkdownHeaderTextSplitter(Splitter):

    def __init__(self, splitter, headers_to_split_on: list[tuple[str, str]]=[('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')], return_each_line: bool=False, strip_headers: bool=False):
        self.return_each_line = return_each_line
        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda split: len(split[0]), reverse=True)
        self.strip_headers = strip_headers
        self.text_splitter = splitter

    def aggregate_lines_to_chunks(self, lines: list[LineType]) -> list[dict]:
        aggregated_chunks: list[LineType] = []
        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]['metadata'] == line['metadata']:
                aggregated_chunks[-1]['content'] += '  \n' + line['content']
            elif aggregated_chunks and aggregated_chunks[-1]['metadata'] != line['metadata'] and (len(aggregated_chunks[-1]['metadata']) < len(line['metadata'])) and (aggregated_chunks[-1]['content'].split('\n')[-1][0] == '#') and (not self.strip_headers):
                aggregated_chunks[-1]['content'] += '  \n' + line['content']
                aggregated_chunks[-1]['metadata'] = line['metadata']
            else:
                aggregated_chunks.append(line)
        return [dict(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in aggregated_chunks]

    def split_text(self, text: str, chunk_size: int=1024, overlap: bool=True) -> list[str]:
        lines = text.split('\n')
        lines_with_metadata: list[LineType] = []
        current_content: list[str] = []
        current_metadata: dict[str, str] = {}
        header_stack: list[HeaderType] = []
        initial_metadata: dict[str, str] = {}
        in_code_block = False
        opening_fence = ''
        for line in lines:
            stripped_line = line.strip()
            stripped_line = ''.join(filter(str.isprintable, stripped_line))
            if not in_code_block:
                if stripped_line.startswith('```') and stripped_line.count('```') == 1:
                    in_code_block = True
                    opening_fence = '```'
                elif stripped_line.startswith('~~~'):
                    in_code_block = True
                    opening_fence = '~~~'
            elif stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ''
            if in_code_block:
                current_content.append(stripped_line)
                continue
            for (sep, name) in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (len(stripped_line) == len(sep) or stripped_line[len(sep)] == ' '):
                    if name is not None:
                        current_header_level = sep.count('#')
                        while header_stack and header_stack[-1]['level'] >= current_header_level:
                            popped_header = header_stack.pop()
                            if popped_header['name'] in initial_metadata:
                                initial_metadata.pop(popped_header['name'])
                        header: HeaderType = {'level': current_header_level, 'name': name, 'data': stripped_line[len(sep):].strip()}
                        header_stack.append(header)
                        initial_metadata[name] = header['data']
                    if current_content:
                        lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata.copy()})
                        current_content.clear()
                    if not self.strip_headers:
                        current_content.append(stripped_line)
                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata.copy()})
                    current_content.clear()
            current_metadata = initial_metadata.copy()
        if current_content:
            lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata})
        chunks = []
        md_sections = self.aggregate_lines_to_chunks(lines_with_metadata)
        for section in md_sections:
            section_text = section['page_content']
            section_headers = section['metadata']
            headers_str = ' '.join((str(v) for v in section_headers.values() if v)).strip()
            parts = extract_images_and_tables_blocks(section_text)
            section_chunks = []
            for part in parts:
                if re.match('^<IMAGE\\s+\\d+\\s*-->', part):
                    section_chunks.append(part)
                elif re.match('^\\s*\\|.*\\|\\s*$', part, re.MULTILINE):
                    section_chunks.append(part)
                else:
                    sub_chunks = self.text_splitter.split_text(text=part, chunk_size=chunk_size, overlap=overlap)
                    section_chunks.extend(sub_chunks)
            for i in range(len(section_chunks)):
                if headers_str:
                    section_chunks[i] = headers_str + '\n' + section_chunks[i]
            chunks += section_chunks
        chunks = self.break_chunks(chunks=chunks, max_size_chunk=chunk_size * 1.2)
        return chunks

class TextSplitter(Splitter):

    def __init__(self):
        pass

    def pieces_from_text(self, text: str, characters: list[str]) -> tuple[list[str], list[str | None]]:
        pattern = f"[{''.join(map(re.escape, characters))}]"
        pieces = re.split(pattern, text)
        separators = re.findall(pattern, text)
        separators.append(None)
        real_pieces = [piece for piece in pieces if piece]
        real_separators = [separator for (separator, piece) in zip(separators, pieces) if piece]
        final_pieces = [piece + sep for (piece, sep) in zip(real_pieces, real_separators) if sep is not None]
        return final_pieces

    def split_text(self, text: str, chunk_size: int=1024, overlap: bool=True) -> list[str]:
        pieces = self.pieces_from_text(text=text, characters=['.', '!', '?', '\n'])
        if not pieces:
            return []
        chunks = []
        seen_chunks = set()
        index = 0
        while index < len(pieces):
            start_index_of_current_chunk = index
            current_chunk_pieces = []
            current_chunk_len = 0
            while current_chunk_len < chunk_size and index < len(pieces):
                piece = pieces[index]
                current_chunk_pieces.append(piece)
                current_chunk_len += len(piece)
                index += 1
                if len(current_chunk_pieces) == 1 and current_chunk_len >= chunk_size:
                    break
            chunk = ''.join(current_chunk_pieces)
            if chunk and chunk not in seen_chunks:
                seen_chunks.add(chunk)
                chunks.append(chunk)
            if overlap and index < len(pieces):
                index = max(start_index_of_current_chunk + 1, index - 1)
        le = [len(c) for c in chunks]
        chunks = self.break_chunks(max_size_chunk=chunk_size * 1.2, chunks=chunks)
        le1 = [len(c) for c in chunks]
        return chunks

class Recursive_TextSplitter(Splitter):

    def __init__(self):
        super().__init__()

    def split_markdown(self, text):
        sections = re.split('(?=\\n#{1,6} )', text)
        chunks = []
        current_chunk = ''
        for section in sections:
            chunks.append(current_chunk.strip())
            current_chunk = section
            chunks.append(current_chunk.strip())
            current_chunk = section
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def __split(self, text, max_chunk_size=500):
        paragraphs = text.strip().split('\n\n')
        chunks = []
        temp_paragraph = ''
        for paragraph in paragraphs:
            paragraph_to_add = paragraph
            if temp_paragraph:
                paragraph_to_add = '\n\n' + paragraph
            if len(temp_paragraph) + len(paragraph_to_add) <= max_chunk_size:
                temp_paragraph += paragraph_to_add
            else:
                if temp_paragraph:
                    chunks.append(temp_paragraph.strip())
                if len(paragraph) <= max_chunk_size:
                    temp_paragraph = paragraph
                else:
                    temp_paragraph = ''
                    sentences = re.split('(?<=[.!?])\\s+', paragraph)
                    temp_chunk = ''
                    for sentence in sentences:
                        sentence_to_add = sentence
                        if temp_chunk:
                            sentence_to_add = ' ' + sentence
                        if len(temp_chunk) + len(sentence_to_add) <= max_chunk_size:
                            temp_chunk += sentence_to_add
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
        if temp_paragraph:
            chunks.append(temp_paragraph.strip())
        return [c for c in chunks if c]

    def split_text(self, text: str, chunk_size: int=1024, overlap: int=0) -> list[str]:
        chunks = self.split_markdown(text=text)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                if len(chunk) > 0:
                    final_chunks.append(chunk)
            else:
                temp = self.__split(text=chunk, max_chunk_size=chunk_size)
                for item in temp:
                    if len(item) > 0:
                        final_chunks.append(item)
        if overlap > 0 and len(final_chunks) > 1:
            overlapping_chunks = []
            for i in range(len(final_chunks)):
                if i > 0:
                    overlap_text = final_chunks[i - 1][-overlap:]
                    overlapping_chunks.append(overlap_text + '\n' + final_chunks[i])
                else:
                    overlapping_chunks.append(final_chunks[i])
            return overlapping_chunks
        final_chunks = self.break_chunks(max_size_chunk=chunk_size * 1.2, chunks=final_chunks)
        return final_chunks

class Semantic_TextSplitter(Splitter):

    def __init__(self, agent, embedding_model):
        self.agent = agent
        self.model = embedding_model

    def compute_breakpoints(self, similarities, method='percentile', threshold=90):
        try:
            if method == 'percentile':
                threshold_value = np.percentile(similarities, threshold)
            elif method == 'standard_deviation':
                mean = np.mean(similarities)
                std_dev = np.std(similarities)
                threshold_value = mean - threshold * std_dev
            elif method == 'interquartile':
                (q1, q3) = np.percentile(similarities, [25, 75])
                threshold_value = q1 - 1.5 * (q3 - q1)
            else:
                raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")
        except Exception as e:
            print("Couldn't compute breakpoints due to empty similarities")
        return [i for (i, sim) in enumerate(similarities) if sim < threshold_value]

    def split_text(self, text, chunk_size: int=1024, overlap: bool=False):
        sentences = nltk.sent_tokenize(text)
        sentences = self.break_chunks(chunks=sentences, max_size_chunk=chunk_size)
        taille_batch = 100
        sentence_embeddings = None
        for i in range(0, len(sentences), taille_batch):
            results = self.agent.embeddings(texts=sentences[i:i + taille_batch], model=self.model)
            if sentence_embeddings is None:
                sentence_embeddings = results
            else:
                sentence_embeddings['embeddings'] = sentence_embeddings['embeddings'] + results['embeddings']
                sentence_embeddings['nb_tokens'] += np.sum(results['nb_tokens'])
        sentence_embeddings = sentence_embeddings['embeddings']
        similarities = [float(cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]) for i in range(len(sentence_embeddings) - 1)]
        breakpoints = self.compute_breakpoints(similarities, threshold=0.5, method='standard_deviation')
        current_chunk = []
        left = 0
        for bp in breakpoints:
            current_chunk.append('. '.join(sentences[left:bp]) + '.')
            left = bp
        current_chunk.append('. '.join(sentences[left:]) + '.')
        chunks = self.break_chunks(chunks=current_chunk, max_size_chunk=chunk_size * 1.2)
        return chunks