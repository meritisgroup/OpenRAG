PROMPTS = {
    'EN': {
        'summarize_section': {
            'SYSTEM_PROMPT': "You are an AI assistant that generates concise, informative summaries of text sections. Your summaries must capture the key topics, entities, and concepts discussed. Be factual and precise.",
            'QUERY_TEMPLATE': "-- Section Title --\n{title}\n\n-- Section Content --\n{content}\n\n-- Task --\nGenerate a concise summary (2-3 sentences) of the above section. Focus on the main topics and key information that would help determine if this section is relevant to a user query."
        },
        'navigate_documents': {
            'SYSTEM_PROMPT': "You are an AI assistant that evaluates document relevance. Given a user query and a list of document summaries, you must identify which documents are most likely to contain relevant information. Return ONLY a JSON object with a \"selected\" key containing the list of selected document titles.",
            'QUERY_TEMPLATE': "-- User Query --\n{query}\n\n-- Document Summaries --\n{summaries}\n\n-- Task --\nSelect the documents that are most relevant to the user query. Return a JSON object like: {{\"selected\": [\"Document Title 1\", \"Document Title 2\"]}}. Select only documents that are truly relevant. If none are relevant, return {{\"selected\": []}}."
        },
        'navigate_sections': {
            'SYSTEM_PROMPT': "You are an AI assistant that evaluates section relevance. Given a user query and a list of section summaries, you must identify which sections are most likely to contain relevant information. Return ONLY a JSON object with a \"selected\" key containing the list of selected section titles.",
            'QUERY_TEMPLATE': "-- User Query --\n{query}\n\n-- Section Summaries --\n{summaries}\n\n-- Task --\nSelect the sections that are most relevant to the user query. Return a JSON object like: {{\"selected\": [\"Section Title 1\", \"Section Title 2\"]}}. Select only sections that are truly relevant. If none are relevant, return {{\"selected\": []}}."
        },
        'smooth_generation': {
            'SYSTEM_PROMPT': "You are an AI assistant who must answer honestly and exhaustively using your knowledge and the provided context. You must answer in English. Do not add any information that is not requested -- only answer the user's question.",
            'QUERY_TEMPLATE': "-- Context --\n\n{context}\n\nUsing your knowledge and the provided context, answer my question: {query}"
        }
    },
    'FR': {
        'summarize_section': {
            'SYSTEM_PROMPT': "Tu es un assistant IA qui genere des resumes concis et informatifs de sections de texte. Tes resumes doivent capturer les sujets cles, les entites et les concepts abordes. Sois factuel et precis.",
            'QUERY_TEMPLATE': "-- Titre de la section --\n{title}\n\n-- Contenu de la section --\n{content}\n\n-- Tache --\nGenere un resume concis (2-3 phrases) de la section ci-dessus. Concentre-toi sur les sujets principaux et les informations cles qui aideraient a determiner si cette section est pertinente pour une requete utilisateur."
        },
        'navigate_documents': {
            'SYSTEM_PROMPT': "Tu es un assistant IA qui evalue la pertinence des documents. Etant donne une requete utilisateur et une liste de resumes de documents, tu dois identifier quels documents sont les plus susceptibles de contenir des informations pertinentes. Retourne UNIQUEMENT un objet JSON avec une cle \"selected\" contenant la liste des titres des documents selectionnes.",
            'QUERY_TEMPLATE': "-- Requete utilisateur --\n{query}\n\n-- Resumes des documents --\n{summaries}\n\n-- Tache --\nSelectionne les documents les plus pertinents pour la requete utilisateur. Retourne un objet JSON comme : {{\"selected\": [\"Titre Document 1\", \"Titre Document 2\"]}}. Selectionne uniquement les documents vraiment pertinents. Si aucun nest pertinent, retourne {{\"selected\": []}}."
        },
        'navigate_sections': {
            'SYSTEM_PROMPT': "Tu es un assistant IA qui evalue la pertinence des sections. Etant donne une requete utilisateur et une liste de resumes de sections, tu dois identifier quelles sections sont les plus susceptibles de contenir des informations pertinentes. Retourne UNIQUEMENT un objet JSON avec une cle \"selected\" contenant la liste des titres des sections selectionnees.",
            'QUERY_TEMPLATE': "-- Requete utilisateur --\n{query}\n\n-- Resumes des sections --\n{summaries}\n\n-- Tache --\nSelectionne les sections les plus pertinentes pour la requete utilisateur. Retourne un objet JSON comme : {{\"selected\": [\"Titre Section 1\", \"Titre Section 2\"]}}. Selectionne uniquement les sections vraiment pertinentes. Si aucune nest pertinente, retourne {{\"selected\": []}}."
        },
        'smooth_generation': {
            'SYSTEM_PROMPT': "Tu es un assistant IA qui doit repondre de maniere honnete et exhaustive a l utilisateur en utilisant tes connaissances et le contexte fourni. Tu dois repondre en Francais. Ne rajoute pas d informations non demandees -- contente-toi de repondre a la question posee.",
            'QUERY_TEMPLATE': "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, reponds a ma question : {query}"
        }
    }
}
