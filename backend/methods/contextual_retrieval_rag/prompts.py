prompts = {
    "EN": {
        "generate_context": {
            "SYSTEM_PROMPT": """Please give a short succinct context to situate the given chunk within the overall document for the purposes of improving search retrieval of the chunk.
                                Answer only with the succinct context and nothing else.
                                You must answer in English""",
            "QUERY_TEMPLATE": """
                                <document> 
                                {WHOLE_DOCUMENT}
                                </document> 
                                Here is the chunk we want to situate within the whole document 
                                <chunk> 
                                {CHUNK_CONTENT}
                                </chunk> """,
        },
        "generate_context2": {
            "SYSTEM_PROMPT": """Given a document, we want to explain what the chunk captures in the document. Answer ONLY with a succinct explaination of the meaning of the chunk in the context of the whole document above and add key words to help for a retrieval process. You must answer in English""",
            "QUERY_TEMPLATE": """
                                Here is the document 
                                <document> 
                                {WHOLE_DOCUMENT}
                                </document> 
                                Here is the chunk we want to situate within the whole document 
                                <chunk> 
                                {CHUNK_CONTENT}
                                </chunk> """,
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": " You are an accurate and reliable AI assistant that can answer questions with the help of external documents. You should only provide the correct answer without repeating the question and instruction. You must answer in English",
            "QUERY_TEMPLATE": " Given the following documents: {context}, Answer the following question: {query}",
        },
    },
    "FR": {
        "generate_context": {
            "SYSTEM_PROMPT": """Veuillez fournir un contexte court et concis pour situer le chunk donné dans l'ensemble du document afin d'améliorer la récupération de recherche du chunk.
                                        Répondez uniquement avec le contexte concis et rien d'autre.
                                        Tu dois répondre en Français""",
            "QUERY_TEMPLATE": """
                                        <document>
                                        {WHOLE_DOCUMENT}
                                        </document>
                                        Voici le chunk que nous voulons situer dans l'ensemble du document
                                        <chunk>
                                        {CHUNK_CONTENT}
                                        </chunk> """,
        },
        "generate_context2": {
            "SYSTEM_PROMPT": """Étant donné un document, nous voulons expliquer ce que représente un extrait (chunk) dans le document. Répondez UNIQUEMENT avec une explication succincte du sens de l’extrait dans le contexte global du document ci-dessus, et ajoutez des mots-clés pour faciliter un processus de recherche pour un bm25. Vous devez répondre en Français""",
            "QUERY_TEMPLATE": """
                                Voici le document
                                <document> 
                                {WHOLE_DOCUMENT}
                                </document> 
                                Voici le chunk que nous voulons situer dans l'ensemble du document 
                                <chunk> 
                                {CHUNK_CONTENT}
                                </chunk> """,
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA précis et fiable qui peut répondre aux questions à l'aide de documents externes. Vous devez uniquement fournir la réponse correcte sans répéter la question ni les instructions. Vous devez répondre en Français",
            "QUERY_TEMPLATE": "Étant donné les documents suivants : {context}, Répondez à la question suivante : {query}",
        },
    },
}
