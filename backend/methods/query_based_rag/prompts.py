prompts = {
    "EN": {
        "qb_prompt": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to generate a list of relevant questions to ask to someone for assessing how well the person has understood the given text. Give the result in the form [question1?, question2?, ...] without adding anything. You must answer in English",
            "QUERY_TEMPLATE": '-- Informations --\nUnderstand the text as well as you can and concentrate on the relevant information (Dates, Concepts, People, Places, Useful links, Figures, ...)\n\n-- Instruction --\nGenerate relevant questions in a well-written english.\nThese questions have to be designed for assessing a reader\'s level of understanding of the given text.\nAsk questions with different levels of difficulty.\nIt is essential that the questions are understandable without having the original text in front of you.\n\n-- Output format --\nPut these questions in a python list.\nReturns this list, without any other text, so that it can be directly processed by a computer program.\n\n-- Example --\nText : "The company\'s first-half results were good. The financial presentation by the marketing team showed growth of over 20% in the airline sector, the highest increase in the last 5 years. BATO\'s CEO, Mr. Smith, consequently offered a bonus to all his teams."\n\nOutput : ["What\'s the name of BATO\'s CEO?", "What business sector has driven BATO\'s growth?", "How much is BATO\'s airline sector estimated to have grown?", "Why did BATO\'s employees receive a bonus?"]\n\n-- Your turn --\nText : "{query}"\n\nOutput : ',
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using your knowledge and the provided context. You must answer in English",
            "QUERY_TEMPLATE": "-- Context --\n\n{context}\n\nEUsing your knowledge and the provided context, answer my question : {query}",
        },
    },
    "FR": {
        "qb_prompt": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui génère uniquement une liste de questions pertinentes pour évaluer la compréhension d'un texte donné. Toujours répondre en Français. La sortie doit être strictement une **liste JSON valide**, par exemple [\"question1\", \"question2\", ...]. Ne mets jamais de bloc Markdown, de balises ```python```, ni de texte supplémentaire. La liste doit être directement exploitable avec json.loads().",
            "QUERY_TEMPLATE": "-- Informations --\nLis attentivement ce texte et concentre-toi sur les informations importantes (dates, concepts, personnes, lieux, liens, chiffres, etc.).\n\n-- Instructions --\nCrée des questions compréhensibles même sans le texte original.\nVarie le niveau de difficulté.\nNe renvoie que la liste JSON, par exemple [\"Question 1 ?\", \"Question 2 ?\", ...].\nNe mets pas de texte additionnel ni de bloc Markdown.\n\n-- Texte à analyser --\n\"{query}\"\n\n-- À ton tour --\nSortie :"

        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui génère uniquement une liste de questions en Français. Toujours renvoyer une **liste JSON valide** avec des guillemets doubles pour chaque string. Échappe tous les guillemets doubles à l'intérieur des questions. La sortie doit être directement exploitable avec json.loads().",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma question : {query}",
        },
    },
}
