from ..contextual_retrieval_rag.contextual import (
    run_batch_contextual,
    run_serial_contextual,
)
from ...database.rag_classes import Chunk


class Processor_chunks:

    def __init__(self, agent, type_processor_chunks=[], language="EN"):

        self.agent = agent
        self.type_processor_chunks = type_processor_chunks
        if type(self.type_processor_chunks) == str:
            self.type_processor_chunks = [self.type_processor_chunks]
        self.language = language

    def process_chunk(self, chunks, doc_content, batch=True):
        if len(self.type_processor_chunks) == 0:
            return {"chunks": chunks, "nb_output_tokens": 0, "nb_input_tokens": 0}
        for i in range(len(self.type_processor_chunks)):
            if self.type_processor_chunks[i] == "Contextual":
                data = self.run_contextual(
                    chunks=chunks, doc_content=doc_content
                )
        return data

    def run_contextual(self, chunks, doc_content):
        data = run_batch_contextual(
                agent=self.agent,
                doc_chunks=chunks,
                doc_content=doc_content,
                language=self.language,
            )
        results = {
            "chunks": [],
            "nb_output_tokens": data["nb_output_tokens"],
            "nb_input_tokens": data["nb_input_tokens"],
        }
        for i in range(len(chunks)):
            results["chunks"].append(
                Chunk(
                    text=data["texts"][i], document=chunks[i].document, id=chunks[i].id
                )
            )
        return results
