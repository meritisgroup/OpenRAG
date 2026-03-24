prompts = {
    'EN': {
        'hypothetical_document': {
            'SYSTEM_PROMPT': 'You are an AI assistant who generates a hypothetical document that would answer the user query. Write a comprehensive and detailed answer to the query using your knowledge. The document should be factual and informative.',
            'QUERY_TEMPLATE': '-- Query --\n\n{query}\n\n-- Task --\n\nWrite a detailed and comprehensive answer to the above query. The answer should be informative and factual.'
        },
        'smooth_generation': {
            'SYSTEM_PROMPT': 'You are an AI assistant who has to answer honestly and exhaustively to the user using your knowledge and the provided context. You must answer in English.',
            'QUERY_TEMPLATE': '-- Context --\n\n{context}\n\n-- Query --\n\n{query}\n\nUsing your knowledge and the provided context, answer my query: {query}'
        }
    },
    'FR': {
        'hypothetical_document': {
            'SYSTEM_PROMPT': "Tu es un assistant IA qui génère un document hypothétique qui répondrait à la requête de l'utilisateur. Rédige une réponse complète et détaillée à la requête en utilisant tes connaissances. Le document doit être factuel et informatif.",
            'QUERY_TEMPLATE': '-- Requête --\n\n{query}\n\n-- Tâche --\n\nRédige une réponse détaillée et complète à la requête ci-dessus. La réponse doit être informative et factuelle.'
        },
        'smooth_generation': {
            'SYSTEM_PROMPT': "Tu es un assistant IA qui doit répondre de manière honnête et exhaustive à l'utilisateur en utilisant tes connaissances et le contexte qui te sera fourni. Tu dois répondre en Français.",
            'QUERY_TEMPLATE': '-- Contexte --\n\n{context}\n\n-- Requête --\n\n{query}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma requête : {query}'
        }
    }
}
