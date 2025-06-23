PROMPTS = {
    "EN": {
        "extraction_text": {
            "SYSTEM_PROMPT": """You are an AI assistant whose job is to transform a text into a Python list of entities and relations. You must answer in English""",
            "QUERY_TEMPLATE": """-Goal-
                    Given a text document, identify all entities of those types from the text and all relationships among the identified entities.
                    Use {language} as output language.

                    -Steps-
                    1. Identify all entities. For each identified entity, extract the following information:
                    - entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
                    - entity_type: The type of the entity, like for example person, technology, concept, company, category, geo, event.
                    - entity_description: Comprehensive description of the entity's attributes and activities
                    Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

                    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
                    For each pair of related entities, extract the following information:
                    - source_entity: name of the source entity, as identified in step 1
                    - target_entity: name of the target entity, as identified in step 1
                    - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
                    - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity, this must be a number between 0 and 1
                    - relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
                    Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

                    3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
                    Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

                    4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

                    5. When finished, output {completion_delimiter}

                    #############################
                    -Real Data-
                    ######################

                    Text: {input_text}
                    ######################
                    Output:""",
        },
        "summarize_community": {
            "SYSTEM_PROMPT": """You're an AI assistant helping humans to interpret knowledge graphs. Knowledge graphs are graphs in which entities are linked together if they are in relationships. You must answer in English"

                            Each entity has been assigned to a community according to their proximity by a clustering algorithm. You will take as input all the descriptions of the entities and relationships in this group."\n\n"

                            # Goal #
                            Describe as well as possible the community thanks to relations and entities descriptions belonging to this community.

                            # Description structure #

                            Your description will consist of two sections, which will take the following names:
                            - Title: The name of the community that represents the key entities - the title should be short but evocative.
                            - Description:  A description of the community and its structure, including how the entities are related to each other and significant information associated with the entities. The SUMMARY section should be very complete, and should summarize exactly what's going on in the group. Relationships must be made explicit.\n\n

                            # Outputs #

                            Outputs a well-formatted json as follows:
                            {
                                "title": <title_section>,
                                "description": <description_section>
                            }""",
            "QUERY_TEMPLATE": """ -----
                                The given entities and relationships, written in {language}, I want yo to summuraize as a unique community are :

                                {list_descriptions}

                                -----
                                The title and description must be in {language}, do not give further explanation, output only the json.
                                """,
        },
        "DEFAULT_TUPLE_DELIMITER": "|",
        "DEFAULT_RECORD_DELIMITER": "##",
        "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
        "process_tickers": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "COMMUNITY_EVALUATOR": {
            "SYSTEM_PROMPT": """-- Informations --
                You will receive a list of communities desciriptions representing from a knowledge graph and a query from a user.

                -- Instructions --
                Your goal is to extract communities that seems relevant to answer the query.

                -- Example --
                Communities :

                {
                    "FinTech Frontier" : "An ecosystem exploring the intersection of finance and technology. Key themes include digital banking, blockchain innovations, robo-advisors, and payment automation reshaping how businesses and consumers interact with money.",
                    "Green Enterprise" : "A network of companies prioritizing sustainability and eco-friendly practices. Topics include renewable energy adoption, circular economy models, ESG reporting, and green supply chains.",
                    "AI Innovation Hub" : "A dynamic space focused on artificial intelligence applications across industries. Features include machine learning models, natural language processing, and AI ethics in business decision-making.",
                    "Retail Renaissance" : "A transformation of the retail sector driven by e-commerce, omnichannel strategies, consumer analytics, and immersive shopping experiences powered by technology."
                    }

                    Query :
                    What communities focus on technology transformation in finance and retail?

                    Output :
                    {
                    "FinTech Frontier" : "An ecosystem exploring the intersection of finance and technology. Key themes include digital banking, blockchain innovations, robo-advisors, and payment automation reshaping how businesses and consumers interact with money.",
                    "Retail Renaissance" : "A transformation of the retail sector driven by e-commerce, omnichannel strategies, consumer analytics, and immersive shopping experiences powered by technology."
                    }

                the format of your ouptut should respect the format of the example output. Adapt the answer to the given communities and the given query. Don't give further explanations.
                    """,
            "QUERY_TEMPLATE": """

                -- Your Turn --
                Communities :
                {context}

                Query :

                {query}

                Output :
                    """,
        },
        "smooth_generation": {
            "SYSTEM_PROMPT": "You are an AI assistant who has to answer honestly and exhaustive to the user using your knowledge and the provided context. You must answer in English",
            "QUERY_TEMPLATE": """-- Context --
                    {context}

                    Using your knowledge and the provided context, answer my question in {language} : {query}""",
        },
        "extraction_query": {
            "SYSTEM_PROMPT": """You are an AI assistant whose job is to transform a text into a Python list of entities
                            For each of them, write their name in a python list. Before and after the list, display the {end_delimiter} symbols
                            You must answer in English""",
            "QUERY_TEMPLATE": "Extract all the entities you read in this question: {query}.",
        },
        "local_search_contxt_template": """ --- CONTEXT ---

                Here is a list of entities and relations between them which may help to answer the query

                Entities :
                    {entities}

                Relations :
                    {relations}

                """,
        "unique_entity": "entity :{entity}, description : {entity_description}\n",
        "unique_relation": "source : {source}, target : {target}, description : {relation_description}\n"
    },
    "FR" : {
    "extraction_text": {
        "SYSTEM_PROMPT": """Vous êtes un assistant IA dont le travail est de transformer un texte en une liste Python d'entités et de relations. Vous devez répondre en Français.""",
        "QUERY_TEMPLATE": """-Objectif-
                Étant donné un document texte, identifiez toutes les entités de ces types à partir du texte et toutes les relations parmi les entités identifiées.
                Utilisez {language} comme langue de sortie.

                -Étapes-
                1. Identifiez toutes les entités. Pour chaque entité identifiée, extrayez les informations suivantes :
                - entity_name: Nom de l'entité, utilisez la même langue que le texte d'entrée. Si c'est en anglais, capitalisez le nom.
                - entity_type: Le type de l'entité, par exemple personne, technologie, concept, entreprise, catégorie, géo, événement.
                - entity_description: Description complète des attributs et activités de l'entité
                Formatez chaque entité comme suit : ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

                2. À partir des entités identifiées à l'étape 1, identifiez toutes les paires de (source_entity, target_entity) qui sont *clairement liées* entre elles.
                Pour chaque paire d'entités liées, extrayez les informations suivantes :
                - source_entity: nom de l'entité source, tel qu'identifié à l'étape 1
                - target_entity: nom de l'entité cible, tel qu'identifié à l'étape 1
                - relationship_description: explication de la raison pour laquelle vous pensez que l'entité source et l'entité cible sont liées entre elles
                - relationship_strength: un score numérique indiquant la force de la relation entre l'entité source et l'entité cible, ce doit être un nombre entre 0 et 1
                - relationship_keywords: un ou plusieurs mots-clés de haut niveau qui résument la nature globale de la relation, en se concentrant sur des concepts ou des thèmes plutôt que sur des détails spécifiques
                Formatez chaque relation comme suit : ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

                3. Identifiez des mots-clés de haut niveau qui résument les principaux concepts, thèmes ou sujets de l'ensemble du texte. Ceux-ci doivent capturer les idées globales présentes dans le document.
                Formatez les mots-clés de niveau de contenu comme suit : ("content_keywords"{tuple_delimiter}<high_level_keywords>)

                4. Retournez la sortie en {language} sous forme de liste unique de toutes les entités et relations identifiées aux étapes 1 et 2. Utilisez **{record_delimiter}** comme délimiteur de liste.

                5. Lorsque vous avez terminé, affichez {completion_delimiter}

                #############################
                -Données Réelles-
                ######################

                Texte: {input_text}
                ######################
                Sortie:""",
    },
    "summarize_community": {
        "SYSTEM_PROMPT": """Vous êtes un assistant IA aidant les humains à interpréter les graphes de connaissances. Les graphes de connaissances sont des graphes dans lesquels les entités sont liées ensemble si elles sont en relation. Vous devez répondre en Français."

                        Chaque entité a été assignée à une communauté selon leur proximité par un algorithme de clustering. Vous prendrez en entrée toutes les descriptions des entités et des relations dans ce groupe."\n\n"

                        # Objectif #
                        Décrivez le mieux possible la communauté grâce aux relations et aux descriptions des entités appartenant à cette communauté.

                        # Structure de la description #

                        Votre description consistera en deux sections, qui prendront les noms suivants :
                        - Titre : Le nom de la communauté qui représente les entités clés - le titre doit être court mais évocateur.
                        - Description : Une description de la communauté et de sa structure, incluant comment les entités sont liées entre elles et les informations significatives associées aux entités. La section RÉSUMÉ doit être très complète et doit résumer exactement ce qui se passe dans le groupe. Les relations doivent être explicites.\n\n

                        # Sorties #

                        Sorties un json bien formaté comme suit :
                        {
                            "title": <title_section>,
                            "description": <description_section>
                        }""",
        "QUERY_TEMPLATE": """ -----
                            Les entités et relations données, écrites en {language}, que je veux résumer en une communauté unique sont :

                            {list_descriptions}

                            -----
                            Le titre et la description doivent être en {language}, ne donnez pas d'explication supplémentaire, affichez uniquement le json.
                            """,
    },
    "DEFAULT_TUPLE_DELIMITER": "|",
    "DEFAULT_RECORD_DELIMITER": "##",
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
    "process_tickers": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "COMMUNITY_EVALUATOR": {
        "SYSTEM_PROMPT": """-- Informations --
            Vous allez recevoir une liste de descriptions de communautés représentant un graphe de connaissances et une requête d'un utilisateur.

            -- Instructions --
            Votre objectif est d'extraire les communautés qui semblent pertinentes pour répondre à la requête.

            -- Exemple --
            Communautés :

            {
                "FinTech Frontier" : "Un écosystème explorant l'intersection de la finance et de la technologie. Les thèmes clés incluent la banque numérique, les innovations blockchain, les robo-conseillers et l'automatisation des paiements qui transforment la manière dont les entreprises et les consommateurs interagissent avec l'argent.",
                "Green Enterprise" : "Un réseau d'entreprises priorisant la durabilité et les pratiques écologiques. Les sujets incluent l'adoption des énergies renouvelables, les modèles d'économie circulaire, les rapports ESG et les chaînes d'approvisionnement vertes.",
                "AI Innovation Hub" : "Un espace dynamique axé sur les applications de l'intelligence artificielle dans divers secteurs. Les caractéristiques incluent les modèles d'apprentissage automatique, le traitement du langage naturel et l'éthique de l'IA dans la prise de décision commerciale.",
                "Retail Renaissance" : "Une transformation du secteur de la vente au détail impulsée par le commerce électronique, les stratégies omnicanal, l'analyse des consommateurs et les expériences d'achat immersives alimentées par la technologie."
                }

                Requête :
                Quelles communautés se concentrent sur la transformation technologique dans la finance et le commerce de détail ?

                Sortie :
                {
                "FinTech Frontier" : "Un écosystème explorant l'intersection de la finance et de la technologie. Les thèmes clés incluent la banque numérique, les innovations blockchain, les robo-conseillers et l'automatisation des paiements qui transforment la manière dont les entreprises et les consommateurs interagissent avec l'argent.",
                "Retail Renaissance" : "Une transformation du secteur de la vente au détail impulsée par le commerce électronique, les stratégies omnicanal, l'analyse des consommateurs et les expériences d'achat immersives alimentées par la technologie."
                }

                le format de votre sortie doit respecter le format de la sortie de l'exemple. Adaptez la réponse aux communautés données et à la requête donnée. Ne donnez pas d'explications supplémentaires.
                    """,
        "QUERY_TEMPLATE": """

                -- Votre Tour --
                Communautés :
                {context}

                Requête :

                {query}

                Sortie :
                    """,
    },
    "smooth_generation": {
        "SYSTEM_PROMPT": "Vous êtes un assistant IA qui doit répondre honnêtement et de manière exhaustive à l'utilisateur en utilisant vos connaissances et le contexte fourni. Vous devez répondre en Français.",
        "QUERY_TEMPLATE": """-- Contexte --
                    {context}

                    En utilisant vos connaissances et le contexte fourni, répondez à ma question en {language} : {query}""",
    },
    "extraction_query": {
        "SYSTEM_PROMPT": """Vous êtes un assistant IA dont le travail est de transformer un texte en une liste Python d'entités.
                            Pour chacune d'elles, écrivez leur nom dans une liste Python. Avant et après la liste, affichez les symboles {end_delimiter}.
                            Vous devez répondre en Français.""",
        "QUERY_TEMPLATE": "Extrayez toutes les entités que vous lisez dans cette question : {query}.",
    },
    "local_search_contxt_template": """ --- CONTEXTE ---

                    Voici une liste d'entités et de relations entre elles qui peuvent aider à répondre à la requête.

                    Entités :
                        {entities}

                    Relations :
                        {relations}

                    """,
    "unique_entity": "entité : {entity}, description : {entity_description}\n",
    "unique_relation": "source : {source}, cible : {target}, description : {relation_description}\n"
}
}
