import numpy as np
from sqlalchemy import func
from .indexation import GraphRagIndexation
from .local_search import LocalSearch
from .global_search import GlobalSearch
from ...base_classes import RagAgent
from ...utils.factory_vectorbase import get_vectorbase
from ...utils.agent import get_Agent
from ...utils.agent_functions import get_system_prompt
from .prompts import PROMPTS
from ..query_reformulation.query_reformulation import query_reformulation
from ...database.database_class import get_management_data
from ...database.rag_classes import Chunk, Entity, Relation, Tokens


class GraphRagAgent(RagAgent):
    """
    This RAG method is inspired of Microsoft GrapRAG method. The indexation phase is performed thanks to knowledge graphs and utilisation phase is split
    in two methods according the nature of the information searched (local or global)
    """

    def __init__(
        self,
        config_server: dict,
        models_infos: dict,
        dbs_name: list[str],
        data_folders_name: list[str]
    ) -> None:
        """
        Be careful the folder associated with storage path has to contain both the data base and the vector base
        """

        self.dbs_name = dbs_name
        self.data_folders_name = data_folders_name

        self.language = config_server["language"]
        self.params_vectorbase = config_server["params_vectorbase"]
        self.nb_chunks = config_server["nb_chunks"]
        self.storage_path = config_server["storage_path"]
        self.embedding_model = config_server["embedding_model"]
        self.llm_model = config_server["model"]
        self.type_text_splitter = config_server["TextSplitter"]
        self.config_server = config_server

        self.agent = get_Agent(config_server,
                               models_infos=models_infos)
        self.data_manager = get_management_data(dbs_name=self.dbs_name,
                                                data_folders_name=self.data_folders_name,
                                                storage_path=self.storage_path,
                                                config_server=config_server,
                                                agent=self.agent)
        self.data_manager.add_table(Entity)
        self.data_manager.add_table(Relation)
        self.data_manager.add_table(Tokens)

        self.reformulate_query = config_server["reformulate_query"]
        if self.reformulate_query:
            self.reformulater = query_reformulation(
                agent=self.agent, language=self.language,
                model=self.llm_model
            )

        self.prompts = PROMPTS[self.language]
        self.system_prompt = get_system_prompt(config_server, self.prompts)

        self.chunk_size = config_server["chunk_length"]

    def indexation_phase(
        self,
        reset_index: bool = False,
        overlap: bool = True,
        reset_preprocess = False
    ) -> None:

        if reset_preprocess:
            reset_index = True
            
        if reset_index:
            self.data_manager.delete_collection(name="local_search")
            self.data_manager.delete_collection(name="global_search")
            self.data_manager.clean_database()

        index = GraphRagIndexation(
            data_manager=self.data_manager,
            storage_path=self.storage_path,
            language=self.language,
            agent=self.agent,
            type_text_splitter=self.type_text_splitter,
            data_preprocessing=self.config_server["data_preprocessing"],
            embedding_model=self.embedding_model,
            llm_model = self.llm_model
        )

        index.run_pipeline(chunk_size=self.chunk_size, 
                           overlap=overlap,
                           config_server=self.config_server,
                           reset_preprocess=reset_preprocess)

    def get_infos_embeddings(self):
        infos = {}
        infos["embedding_tokens"] = np.sum(self.data_manager.query(func.sum(Tokens.embedding_tokens)))
        infos["input_tokens"] = np.sum(self.data_manager.query(func.sum(Tokens.input_tokens)))
        infos["output_tokens"] = np.sum(self.data_manager.query(func.sum(Tokens.output_tokens)))
        return infos
                                                                                                        
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
                model=self.llm_model,
                data_manager = self.data_manager,
                pre_filter_size=nb_chunks,
                language=self.language,
            )

        else:
            search = LocalSearch(
                agent=self.agent,
                model=self.llm_model,
                data_manager = self.data_manager,
                start_node=nb_chunks,
                language=self.language,
            )

        return search.get_context(query=query)


    def generate_answer(
        self,
        query: str,
        method: str = "global",
        nb_chunks: str = 2,
        options_generation = None
    ) -> str:
        context, chunks, tokens_counter = self.get_rag_context(
            query=query, method=method, nb_chunks=nb_chunks
        )

        input_t = tokens_counter["nb_input_tokens"]
        output_t = tokens_counter["nb_output_tokens"]
        impacts = [0, 0, ""]
        energies = [0, 0, ""]

        prompt_template: str = self.prompts["smooth_generation"]["QUERY_TEMPLATE"]

        context_base = dict(context=context, 
                            language=self.language,
                            query=query)

        prompt = prompt_template.format(**context_base)

        if options_generation is None:
            options_generation = self.config_server["options_generation"]

        answer = self.agent.predict(prompt=prompt,
                                    system_prompt=self.system_prompt,
                                    options_generation=options_generation,
                                    model=self.llm_model)
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
            "context": chunks,
            "impacts": impacts,
            "energy": energies,
            "original_query": query
        }

        return answer_dict

