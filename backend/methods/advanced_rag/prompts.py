prompts = {
    "EN": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using your knowledge and the provided context. You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context}\n\nEUsing your knowledge and the provided context, answer my question : {query}",
        }
    },
    "FR": {
        "smooth_generation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit répondre de manière honnête et exhaustive à l'utilisateur en utilisant tes connaissances et le contexte qui te sera fourni. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma question : {query}",
        },
    },
}
