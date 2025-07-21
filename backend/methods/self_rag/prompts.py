prompts = {
    "EN": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using the provided context. You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context}\n\n Using the provided context, answer my question : {query}",
        },
        "document_relevance": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to determine if the provided context is relevante to answer the query. Output only 'relevant', 'irrelevant' or 'ambiguous'.",
            "QUERY_TEMPLATE": "-- Context --\n\n{context} \n\n -- Query --\n\n{query}",
        },
        "retrieval_necessary": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to determine if retrieval of documents is necessary to answer the query. Output only 'Yes' or 'No'",
            "QUERY_TEMPLATE": "{query}",
        },
        "rate_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to rate the response to the query. Rate from 1 to 5 where 1 is the worst, 5 the best. Output only '1','2','3','4' or '5' without explanation.",
            "QUERY_TEMPLATE": "-- Query --\n\n{query}\n\n -- Response -- \n\n{context}",
        },
        "supported_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to determine if the response is supported by the context. Output only 'fully supported', 'partially supported' or 'no support' without explanation.",
            "QUERY_TEMPLATE": "-- Response --\n\n{query}\n\n -- Context -- \n\n{context}",
        },
    },
    "FR": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA qui doit répondre honnêtement et de manière exhaustive à l'utilisateur en utilisant le contexte fourni. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant le contexte fourni, réponds à ma question: {query}",
        },
        "document_relevance": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit déterminer si le contexte fourni est pertinent pour répondre à la requête. Réponds uniquement par 'relevant', 'irrelevant' ou 'ambiguous'",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context} \n\n -- Question --\n\n{query}",
        },
        "conversationnal": {
            "SYSTEM_PROMPT": "Tu es un assistant IA. Tu réponds toujours avec précision et honnêteté. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "Réponds à cette question : {query}",
        },
        "retrieval_necessary": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA qui doit déterminer si la récupération de documents est nécessaire pour répondre à la requête. Répondez uniquement par 'Yes' ou 'No'",
            "QUERY_TEMPLATE": "---QUERY---\n {query}",
        },
        "rate_generation": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA qui doit évaluer la réponse à la requête. Évaluez de 1 à 5, où 1 est le pire et 5 le meilleur. Répondez uniquement par '1', '2', '3', '4' ou '5' sans explication.",
            "QUERY_TEMPLATE": "-- Question --\n\n{query}\n\n -- Réponse -- \n\n{context}",
        },
        "supported_generation": {
            "SYSTEM_PROMPT": "Vous êtes un assistant IA qui doit déterminer si la réponse est soutenue par le contexte. Répondez uniquement par 'fully supported', 'partially supported' ou 'no support' sans explication.",
            "QUERY_TEMPLATE": "-- Réponse --\n\n{query}\n\n -- Contexte -- \n\n{context}",
        },
    },
}
