from methods.naive_rag.indexation import NaiveRagIndexation

def concat_chunks(chunks):
    """Concatenate chunks into a single context string."""
    return "\n\n---\n\n".join([f"[Source: {chunk.document}]\n{chunk.text}" for chunk in chunks])

def contexts_to_prompts(contexts, query):
    """Convert contexts to prompt format."""
    return [f"Context: {context}\nQuery: {query}" for context in contexts]
