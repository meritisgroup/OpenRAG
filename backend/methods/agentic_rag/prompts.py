prompts = {
    "EN": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using your knowledge and the provided context. You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context}\n\n Using your knowledge and the provided context, answer my question : {query}",
        },
        "evaluate": {"SYSTEM_PROMPT": """You are an expert assistant tasked with evaluating whether an answer is complete with respect to a question.
                    You must respond only in valid JSON of the form {"Decision": true} if the answer is complete and correct, or {"Decision": false} otherwise.""",
                    "QUERY_TEMPLATE": """Question:
                            {query}

                            Answer:
                            {answer}

                            Does the answer fully and satisfactorily address the question?
                            Reply only in JSON format, e.g. {"Decision": true} or {"Decision": false}."""
                    },
        "reformulate": {"SYSTEM_PROMPT": "You are an AI assistant tasked with improving a generated answer.\n You are given an original user question and an answer previously produced by another model.\n You must analyze what is missing from the answer in relation to the question.\n Then generate ONE NEW QUERY that targets ONLY the missing information, and that can be used to complete the original answer.\n Your new query must be directly reusable by an LLM to produce a complementary answer.\n Do NOT rephrase the entire question. Do NOT repeat what has already been answered. Focus ONLY on the missing parts.\n",
                        "QUERY_TEMPLATE": """f"--- Original question ---\n{query}\n\n--- Generated answer ---\n{answer}\n\n What new query would you ask to retrieve only the missing information?"""},
        "concatenete": {"SYSTEM_PROMPT":"""You are an AI assistant. Your task is to synthesize two partial answers into a single, well-written and complete response to a given user question.\n Do not repeat content unnecessarily. Combine both answers intelligently and logically.\n The final answer must be clear, comprehensive, and directly address the user's query.""",
                        "QUERY_TEMPLATE": """--- User Question ---\n{query}\n\n--- Initial Answer ---\n{answer_init}\n\n--- Additional Answer ---\n{answer_add}\n\n--- Task ---\nCombine the two answers above into a single complete response that fully addresses the user's question."""},
    },
    "FR": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit répondre de manière honnête et exhaustive à l'utilisateur en utilisant tes connaissances et le contexte qui te sera fourni. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma question : {query}",
        },
        "evaluate": {
            "SYSTEM_PROMPT": """Tu es un assistant expert chargé d'évaluer si une réponse est complète par rapport à une question.
                                Tu dois répondre uniquement en JSON valide sous la forme {"Decision": true} si la réponse est complète et correcte, ou {"Decision": false} dans le cas contraire.""",

            "QUERY_TEMPLATE": """Question :
                        {query}

                        Réponse :
                        {answer}

                        La réponse répond-elle entièrement et de manière satisfaisante à la question ?
                        Réponds uniquement au format JSON, par exemple {"Decision": true} ou {"Decision": false}."""
    },

        "reformulate": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA chargé d'améliorer une réponse générée.\nOn vous fournit une question initiale et une réponse produite précédemment par un autre modèle.\nVous devez analyser ce qui manque dans la réponse par rapport à la question.\nPuis, générez UNE NOUVELLE QUESTION visant UNIQUEMENT les informations manquantes, qui pourra être utilisée pour compléter la réponse originale.\nVotre nouvelle question doit pouvoir être directement réutilisée par un LLM pour produire une réponse complémentaire.\nNe reformulez pas toute la question. Ne répétez pas ce qui a déjà été répondu. Concentrez-vous UNIQUEMENT sur les éléments manquants.",
            "QUERY_TEMPLATE": "--- Question originale ---\n{query}\n\n--- Réponse générée ---\n{answer}\n\nQuelle nouvelle question poseriez-vous pour obtenir uniquement les informations manquantes ?"
        },

        "concatenete": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA. Votre tâche est de synthétiser deux réponses partielles en une seule réponse complète et bien rédigée à une question utilisateur donnée.\nNe répétez pas le contenu inutilement. Combinez les deux réponses de manière intelligente et logique.\nLa réponse finale doit être claire, complète et traiter directement la question de l'utilisateur.",
            "QUERY_TEMPLATE": "--- Question de l'utilisateur ---\n{query}\n\n--- Réponse initiale ---\n{answer_init}\n\n--- Réponse supplémentaire ---\n{answer_add}\n\n--- Tâche ---\nCombinez les deux réponses ci-dessus en une seule réponse complète qui traite entièrement la question de l'utilisateur."
        }
    },
}
