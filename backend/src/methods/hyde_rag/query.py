from utils.agent import Agent
from base_classes import Search
from database.database_class import Merger_Database_Vectorbase
from database.rag_classes import Chunk
from .hypothetical_document import HypotheticalDocumentGenerator

class HydeSearch(Search):

    def __init__(self, data_manager: Merger_Database_Vectorbase, nb_chunks: int=10, doc_generator=None) -> None:
        super().__init__(Agent)
        self.data_manager = data_manager
        self.nb_chunks = nb_chunks
        self.doc_generator = doc_generator

    def get_context(self, query: str) -> tuple[list[list[Chunk]], dict]:
        """
        Get context using HyDE approach:
        1. Generate hypothetical document from query
        2. Use hypothetical document for embedding-based retrieval
        3. Return retrieved chunks and metadata
        """
        if type(query) is str:
            query = [query]

        contexts = []
        metadata = {
            'hypothetical_documents': [],
            'nb_input_tokens': 0,
            'nb_output_tokens': 0,
            'impacts': [0, 0, ''],
            'energy': [0, 0, '']
        }

        for q in query:
            if self.doc_generator:
                # Generate hypothetical document
                (hypothetical_doc, input_t, output_t, impacts, energy) = \
                    self.doc_generator.generate_hypothetical_document(q)

                metadata['hypothetical_documents'].append(hypothetical_doc)
                metadata['nb_input_tokens'] += input_t
                metadata['nb_output_tokens'] += output_t
                metadata['impacts'][0] += impacts[0]
                metadata['impacts'][1] += impacts[1]
                metadata['impacts'][2] = impacts[2]
                metadata['energy'][0] += energy[0]
                metadata['energy'][1] += energy[1]
                metadata['energy'][2] = energy[2]

                # Use hypothetical document for search
                search_res = self.data_manager.k_search(
                    queries=[hypothetical_doc],
                    k=self.nb_chunks,
                    output_fields=['text', 'doc_name']
                )
                contexts.append(search_res[0])
            else:
                # Fallback to regular search if no generator
                search_res = self.data_manager.k_search(
                    queries=[q],
                    k=self.nb_chunks,
                    output_fields=['text', 'doc_name']
                )
                contexts.append(search_res[0])

        return (contexts, metadata)
