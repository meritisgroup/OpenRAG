prompts = {
    "EN": {
        "rewrite_web_query": {
            "SYSTEM_PROMPT": "You are an AI assistant who has rewrite a query to a websearch query. Output only the websearch query without any comment or description. You must answer in English",
            "QUERY_TEMPLATE": "-- Query --\n\n{query}",
        },
        "document_relevance2": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to determine if the provided context is relevante to answer the query. Output only 'relevant', 'irrelevant' or ambiguous.You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context} \n\n -- Query --\n\n{query}",
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using your knowledge and the provided context. You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context}\n\nEUsing your knowledge and the provided context, answer my question : {query}",
        },
    },
    "FR": {
        "rewrite_web_query": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit reformuler une requête en une requête de recherche web. Ne fournis que la requête de recherche web, sans aucun commentaire ni description. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Question --\n\n{query}",
        },
        "document_relevance2": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit déterminer si le contexte fourni est pertinent pour répondre à la requête. Réponds uniquement par 'pertinent', 'non pertinent' ou 'ambigu'",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context} \n\n -- Question --\n\n{query}",
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit répondre de manière honnête et exhaustive à l'utilisateur en utilisant tes connaissances et le contexte qui te sera fourni. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma question : {query}",
        },
    },
}
