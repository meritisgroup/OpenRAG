prompts = {
    "EN": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using your knowledge and the provided context. You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context}\n\nEUsing your knowledge and the provided context, answer my question : {query}",
        },
        "query_reformulation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to reformulate in {language} the user query to make a good query for retrieval without changing the meaning of the query. Do not add any additional text.",
            "QUERY_TEMPLATE": "-- Query --\n\n {query}",
        },
    },
    "FR": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit répondre de manière honnête et exhaustive à l'utilisateur en utilisant tes connaissances et le contexte qui te sera fourni. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma question : {query}",
        },
        "query_reformulation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit reformuler la requête de l'utilisateur en {language} pour en faire une bonne requête de recherche. N'ajoute aucun texte supplémentaire.",
            "QUERY_TEMPLATE": "-- Query --\n\n {query}",
        }
    },
}
