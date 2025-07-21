import numpy as np
from .indexation import GraphRagIndexation
from .local_search import LocalSearch
from .global_search import GlobalSearch
from ...base_classes import RagAgent
from ...utils.factory_vectorbase import get_vectorbase
from ...database.database_class import get_graph_database
from ...utils.agent import get_Agent
from ...utils.agent_functions import get_system_prompt
from .prompts import PROMPTS
from ..query_reformulation.query_reformulation import query_reformulation


class GraphRagAgent(RagAgent):
    """
    This RAG method is inspired of Microsoft GrapRAG method. The indexation phase is performed thanks to knowledge graphs and utilisation phase is split
    in two methods according the nature of the information searched (local or global)
    """

    def __init__(
        self,
        config_server: dict,
        db_name: str = "db_graph_rag",
        vb_name: str = "vb_graph_rag",
    ) -> None:
        """
        Be careful the folder associated with storage path has to contain both the data base and the vector base
        """

        self.language = config_server["language"]
        self.params_vectorbase = config_server["params_vectorbase"]
        self.nb_chunks = config_server["nb_chunks"]
        self.storage_path = config_server["storage_path"]
        self.embedding_model = config_server["embedding_model"]
        self.db_name = db_name
        self.vb_name = vb_name
        self.type_text_splitter = config_server["TextSplitter"]
        self.db = get_graph_database(db_name=db_name, storage_path=self.storage_path)

        self.agent = get_Agent(config_server)

        self.vb = get_vectorbase(
            vb_name=vb_name,
            config_server=config_server,
            agent=self.agent,
        )
        self.reformulate_query = config_server["reformulate_query"]
        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language
            )

        self.prompts = PROMPTS[self.language]
        self.system_prompt = get_system_prompt(config_server, self.prompts)

        self.chunk_size = config_server["chunk_length"]

    def indexation_phase(
        self,
        path_input: str,
        reset_index: bool = False,
        overlap: bool = True,
    ) -> None:

        if reset_index:
            self.vb.delete_collection()
            self.db.clean_database()

        index = GraphRagIndexation(
            data_path=path_input,
            storage_path=self.storage_path,
            language=self.language,
            agent=self.agent,
            type_text_splitter=self.type_text_splitter,
            embedding_model=self.embedding_model,
            db=self.db,
            vb=self.vb,
        )

        index.run_pipeline(chunk_size=self.chunk_size, overlap=overlap)

    def get_rag_context(
        self,
        query: str,
        method: str = "global",
        nb_chunks: int = 2,
    ) -> str:
        """ """

        if method == "global":
            search = GlobalSearch(
                agent=self.agent,
                data_base=self.db,
                vector_base=self.vb,
                pre_filter_size=nb_chunks,
                language=self.language,
            )

        else:
            search = LocalSearch(
                agent=self.agent,
                data_base=self.db,
                vector_base=self.vb,
                start_node=nb_chunks,
                language=self.language,
            )

        return search.get_context(query=query)

    def generate_answer(
        self,
        query: str,
        method: str = "global",
        nb_chunks: str = 2,
    ) -> str:
        context, tokens_counter = self.get_rag_context(
            query=query, method=method, nb_chunks=nb_chunks
        )

        input_t = tokens_counter["nb_input_tokens"]
        output_t = tokens_counter["nb_output_tokens"]
        impacts = [0, 0, ""]
        energies = [0, 0, ""]

        if self.reformulate_query:
            query, input_t, output_t, impacts, energies = self.reformulater.reformulate(
                query=query, nb_reformulation=1
            )
            query = query[0]
        prompt_template: str = self.prompts["smooth_generation"]["QUERY_TEMPLATE"]

        context_base = dict(context=context, language=self.language, query=query)

        prompt = prompt_template.format(**context_base)

        answer = self.agent.predict(prompt=prompt, system_prompt=self.system_prompt)
        impacts[2] = answer["impacts"][2]
        impacts[0] += answer["impacts"][0]
        impacts[1] += answer["impacts"][1]

        energies[2] = answer["energy"][2]
        energies[0] += answer["energy"][0]
        energies[1] += answer["energy"][1]

        answer_dict = {
            "answer": answer["texts"],
            "nb_input_tokens": np.sum(answer["nb_input_tokens"] + input_t),
            "nb_output_tokens": np.sum(answer["nb_output_tokens"] + output_t),
            "context": context,
            "impacts": impacts,
            "energy": energies,
        }

        return answer_dict

    def generate_answers(self, queries: list[str], nb_chunks: int = 2):
        answers = []
        for query in queries:
            answer = self.generate_answer(query=query, nb_chunks=nb_chunks)
            answers.append(answer)
        return answers
