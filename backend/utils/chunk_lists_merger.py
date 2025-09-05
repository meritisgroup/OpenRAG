from backend.database.rag_classes import Chunk


def merge_chunk_lists(chunk_lists: list[list[Chunk]]) -> list[Chunk]:
    merged_chunk_list = [chunk for chunk_list in chunk_lists for chunk in chunk_list]
    return merged_chunk_list
