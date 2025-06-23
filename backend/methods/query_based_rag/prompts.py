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
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit générer une liste de questions pertinentes à poser à quelqu'un pour évaluer à quel point cette personne a compris ou non le texte donné. Donne le résultat sous la forme [question1?, question2?,...] sans rien ajouter. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Informations --\nComprends au mieux ce texte et concentre toi sur les informations pertinentes (Dates, Concepts, Personnes, Lieux, Liens utiles, Chiffres, ...)\n\n-- Instructions --\nCréer des questions pertinentes dans un français formel.\nCes questions doivent permettre de juger le niveau de compréhension du texte d'un lecteur.\nFais des questions avec plusieurs niveaux de difficultés.\nIl faut absolument que les questions soient compréhensibles sans avoir le texte original sous les yeux.\n\n-- Structure de la sortie --\nRange ces questions dans une liste python.\nRenvoie cette liste, sans l'agrémenter d'aucun autre texte, pour qu'elle puisse diectement être traitée par un programme informatique.\n\n-- Exemple --\nTexte : \"Le bilan de l'entreprise a été bon sur ce premier semestre. La présentation financière réalisée par l'équipe marketing a notamment montré une croissance de plus de 20% du secteur aérien ce qui représente la plus forte augmentation sur ces 5 dernières années. Le PDG de l'entreprise BATO, M. Smith, a en conséquences offert une prime à l'ensemble de ses équipes.\"\n\nSortie : [\"Comment s'appelle le PDG de l'entreprise BATO ?\", \"Quel secteur d'activité a porté la croissance de l'entreprise BATO ?\", \"A combien est estimé la croissance du secteur aérien chez BATO ?\", \"Pourquoi les employés de BATO, ont-ils reçu une prime ?\"]\n\n-- A ton tour --\nTexte : \"{query}\"\n\nSortie : ",
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui doit répondre de manière honnête et exhaustive à l'utilisateur en utilisant tes connaissances et le contexte qui te sera fourni. Tu dois répondre en Français",
            "QUERY_TEMPLATE": "-- Contexte --\n\n{context}\n\nEn utilisant tes connaissances et le contexte fourni, réponds à ma question : {query}",
        },
    },
}
