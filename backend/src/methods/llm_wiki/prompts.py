PROMPTS = {
    "EN": {
        "analyze_source": {
            "SYSTEM_PROMPT": "You are an AI assistant that analyzes documents to extract structured knowledge for a wiki. You must identify entities, concepts, facts, and connections. Be thorough and precise. Follow the wiki schema conventions provided.",
            "QUERY_TEMPLATE": """-- Wiki Schema --
{schema}

-- Existing Wiki Index --
{existing_index}

-- Source Document Name --
{source_name}

-- Source Document Content --
{content}

-- Task --
Analyze this document and extract structured knowledge. Return a JSON object with the following structure:
{{
    "entities": [
        {{"name": "Entity Name", "type": "person|organization|product|location|other", "description": "Brief description", "facts": ["fact1", "fact2"]}}
    ],
    "concepts": [
        {{"name": "Concept Name", "description": "Brief description", "facts": ["fact1", "fact2"]}}
    ],
    "key_facts": ["Important fact 1", "Important fact 2"],
    "source_summary": "2-3 sentence summary of the document",
    "contradictions_with_wiki": [
        {{"wiki_page": "page title", "existing_info": "what the wiki says", "new_info": "what the source says"}}
    ],
    "pages_to_update": [
        {{"title": "Existing Page Title", "page_type": "entity|concept", "new_facts": ["facts to merge"]}}
    ]
}}""",
        },
        "generate_pages": {
            "SYSTEM_PROMPT": "You are an AI assistant that generates wiki pages in markdown format. Each page must have YAML frontmatter and use wikilinks [[Page Name]] to reference other pages. Follow the wiki schema conventions provided. For existing pages, produce the COMPLETE merged page (old content integrated with new information).",
            "QUERY_TEMPLATE": """-- Wiki Schema --
{schema}

-- Source Name --
{source_name}

-- Analysis Result --
{analysis}

-- Contradictions Detected --
{contradictions}

-- Existing Pages to Update --
{existing_pages}

-- Task --
Generate the wiki pages based on the analysis. Return a JSON array of page objects:
[
    {{
        "action": "create|update",
        "page_type": "entity|concept|source|synthesis",
        "title": "Page Title",
        "slug": "page-slug",
        "frontmatter": {{
            "title": "Page Title",
            "type": "entity",
            "summary": "One-line description of this page (max 120 chars)",
            "sources": ["source_name.pdf"],
            "tags": ["tag1", "tag2"]
        }},
        "content": "Full markdown content with [[wikilinks]] and [standard links](path.md)...\\n\\nFor UPDATE actions, include ALL previous content merged with new information."
    }}
]

Always create a source page summarizing the document. Create entity and concept pages for each identified entity/concept. For existing pages, produce the COMPLETE merged version. Every page MUST have a "summary" in its frontmatter: a concise one-line description (max 120 characters) suitable for a wiki index listing.""",
        },
        "navigate_index": {
            "SYSTEM_PROMPT": "You are an AI assistant that selects relevant wiki pages for answering a query. Given a wiki index and a user query, identify which pages contain relevant information. Follow the wiki schema conventions provided. Return ONLY a JSON object with selected page paths.",
            "QUERY_TEMPLATE": """-- Wiki Schema --
{schema}

-- User Query --
{query}

-- Wiki Index --
{index}

-- Task --
Select the wiki pages that are most relevant to the user query. Return a JSON object like:
{{"selected": ["entities/company-name", "concepts/market-strategy", "sources/report-2025"]}}
Use the format "category/slug" (without .md extension). Select up to 10 most relevant pages. If none are relevant, return {{"selected": []}}.""",
        },
        "synthesize_answer": {
            "SYSTEM_PROMPT": "You are an AI assistant that answers questions using wiki page content. You must cite your sources using numbered references [1], [2], etc. corresponding to the provided pages. Follow the wiki schema conventions for tone and style. Be thorough and accurate.",
            "QUERY_TEMPLATE": """-- Wiki Schema --
{schema}

-- Wiki Pages --
{pages}

-- Task --
Using the wiki pages provided above, answer the following question. Cite your sources using numbered references [1], [2], etc. where each number corresponds to the page number above.

Question: {query}""",
        },
        "lint_report": {
            "SYSTEM_PROMPT": "You are an AI assistant that reviews wiki pages for quality issues. You must identify orphans, broken links, contradictions, stale content, and missing pages. Follow the wiki schema conventions provided. Return a structured analysis.",
            "QUERY_TEMPLATE": """-- Wiki Schema --
{schema}

-- Wiki Index --
{index}

-- All Pages Content --
{pages_content}

-- Wikilink Map --
{link_map}

-- Task --
Analyze the wiki for quality issues. Return a JSON object:
{{
    "orphans": ["page-slug1", "page-slug2"],
    "broken_links": [{{"from": "source-page", "link": "broken-link-target"}}],
    "missing_pages": ["mentioned-but-no-page1", "mentioned-but-no-page2"],
    "contradictions": [{{"page1": "slug1", "page2": "slug2", "description": "what contradicts"}}],
    "stale_pages": ["page-slug-that-needs-update"],
    "health_score": 85,
    "suggestions": ["suggestion1", "suggestion2"]
}}""",
        },
        "update_overview": {
            "SYSTEM_PROMPT": "You are an AI assistant that generates concise wiki overviews. Given all wiki pages, produce a synthesis that captures the key themes, entities, and relationships.",
            "QUERY_TEMPLATE": """-- All Wiki Pages --
{all_pages}

-- Task --
Generate a concise overview (200-400 words) for the wiki home page that synthesizes the main themes, key entities, and important relationships across all pages. Use wikilinks [[Page Name]] to reference key pages.""",
        },
        "crystallize": {
            "SYSTEM_PROMPT": "You are an AI assistant that evaluates whether a Q&A pair is worth saving as a wiki page. A good candidate contains non-trivial synthesis, useful analysis, a comparison, or a connection that would be valuable to preserve for future queries. Trivial or obvious answers should NOT be crystallized.",
            "QUERY_TEMPLATE": """-- Question --
{query}

-- Answer --
{answer}

-- Task --
Evaluate whether this Q&A pair is worth crystallizing into a wiki page. Return a JSON object:
{{
    "should_crystallize": true/false,
    "reason": "brief explanation",
    "suggested_title": "Suggested Page Title",
    "suggested_summary": "One-line summary (max 120 chars)",
    "suggested_tags": ["tag1", "tag2"],
    "page_content": "The markdown content for the wiki page if crystallizing. Must include [[wikilinks]] to relevant concepts. Only present if should_crystallize is true."
}}""",
        },
    },
    "FR": {
        "analyze_source": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui analyse des documents pour extraire des connaissances structurees pour un wiki. Tu dois identifier les entites, concepts, faits et connexions. Sois minutieux et precis. Suis les conventions du schema wiki fournies.",
            "QUERY_TEMPLATE": """-- Schema du Wiki --
{schema}

-- Index du Wiki Existant --
{existing_index}

-- Nom du Document Source --
{source_name}

-- Contenu du Document Source --
{content}

-- Tache --
Analyse ce document et extrais les connaissances structurees. Retourne un objet JSON avec la structure suivante :
{{
    "entities": [
        {{"name": "Nom de l'Entite", "type": "person|organization|product|location|other", "description": "Description breve", "facts": ["fait1", "fait2"]}}
    ],
    "concepts": [
        {{"name": "Nom du Concept", "description": "Description breve", "facts": ["fait1", "fait2"]}}
    ],
    "key_facts": ["Fait important 1", "Fait important 2"],
    "source_summary": "Resume du document en 2-3 phrases",
    "contradictions_with_wiki": [
        {{"wiki_page": "titre de la page", "existing_info": "ce que dit le wiki", "new_info": "ce que dit la source"}}
    ],
    "pages_to_update": [
        {{"title": "Titre de la Page Existante", "page_type": "entity|concept", "new_facts": ["faits a fusionner"]}}
    ]
}}""",
        },
        "generate_pages": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui genere des pages wiki en format markdown. Chaque page doit avoir un frontmatter YAML et utiliser les wikilinks [[Nom de Page]] pour referencer d'autres pages. Suis les conventions du schema wiki fournies. Pour les pages existantes, produis la page COMPLETE fusionnee (ancien contenu integre avec les nouvelles informations).",
            "QUERY_TEMPLATE": """-- Schema du Wiki --
{schema}

-- Nom de la Source --
{source_name}

-- Resultat de l'Analyse --
{analysis}

-- Contradictions Detectees --
{contradictions}

-- Pages Existantes a Mettre a Jour --
{existing_pages}

-- Tache --
Genere les pages wiki basees sur l'analyse. Retourne un tableau JSON d'objets page :
[
    {{
        "action": "create|update",
        "page_type": "entity|concept|source|synthesis",
        "title": "Titre de la Page",
        "slug": "slug-de-page",
        "frontmatter": {{
            "title": "Titre de la Page",
            "type": "entity",
            "summary": "Description en une ligne de cette page (max 120 caracteres)",
            "sources": ["nom_source.pdf"],
            "tags": ["tag1", "tag2"]
        }},
        "content": "Contenu markdown complet avec [[wikilinks]] et [liens standards](chemin.md)...\\n\\nPour les actions UPDATE, inclure TOUT le contenu precedent fusionne avec les nouvelles informations."
    }}
]

Cree toujours une page source resumant le document. Cree des pages entity et concept pour chaque entite/concept identifie. Pour les pages existantes, produis la version COMPLETE fusionnee. Chaque page DOIT avoir un champ \"summary\" dans son frontmatter : une description concise en une ligne (max 120 caracteres) adaptee pour un listing dans l'index du wiki.""",
        },
        "navigate_index": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui selectionne les pages wiki pertinentes pour repondre a une requete. Etant donne un index wiki et une requete utilisateur, identifie quelles pages contiennent des informations pertinentes. Suis les conventions du schema wiki fournies. Retourne UNIQUEMENT un objet JSON avec les chemins des pages selectionnees.",
            "QUERY_TEMPLATE": """-- Schema du Wiki --
{schema}

-- Requete Utilisateur --
{query}

-- Index du Wiki --
{index}

-- Tache --
Selectionne les pages wiki les plus pertinentes pour la requete utilisateur. Retourne un objet JSON comme :
{{"selected": ["entities/nom-entreprise", "concepts/strategie-marche", "sources/rapport-2025"]}}
Utilise le format "categorie/slug" (sans extension .md). Selectionne jusqu'a 10 pages les plus pertinentes. Si aucune n'est pertinente, retourne {{"selected": []}}.""",
        },
        "synthesize_answer": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui repond aux questions en utilisant le contenu des pages wiki. Tu dois citer tes sources avec des references numerotees [1], [2], etc. correspondant aux pages fournies. Suis les conventions du schema wiki pour le ton et le style. Sois exhaustif et precis.",
            "QUERY_TEMPLATE": """-- Schema du Wiki --
{schema}

-- Pages Wiki --
{pages}

-- Tache --
En utilisant les pages wiki fournies ci-dessus, reponds a la question suivante. Cite tes sources avec des references numerotees [1], [2], etc. ou chaque numero correspond au numero de page ci-dessus.

Question : {query}""",
        },
        "lint_report": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui examine les pages wiki pour detecter des problemes de qualite. Tu dois identifier les orphelins, liens casses, contradictions, contenu obsolete et pages manquantes. Suis les conventions du schema wiki fournies. Retourne une analyse structuree.",
            "QUERY_TEMPLATE": """-- Schema du Wiki --
{schema}

-- Index du Wiki --
{index}

-- Contenu de Toutes les Pages --
{pages_content}

-- Carte des Wikilinks --
{link_map}

-- Tache --
Analyse le wiki pour detecter les problemes de qualite. Retourne un objet JSON :
{{
    "orphans": ["slug-page1", "slug-page2"],
    "broken_links": [{{"from": "page-source", "link": "cible-lien-casse"}}],
    "missing_pages": ["mentionne-mais-pas-de-page1", "mentionne-mais-pas-de-page2"],
    "contradictions": [{{"page1": "slug1", "page2": "slug2", "description": "ce qui contredit"}}],
    "stale_pages": ["slug-page-a-mettre-a-jour"],
    "health_score": 85,
    "suggestions": ["suggestion1", "suggestion2"]
}}""",
        },
        "update_overview": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui genere des resumes synthetiques de wiki. Etant donne toutes les pages wiki, produis une synthese qui capture les themes cles, les entites et les relations importantes.",
            "QUERY_TEMPLATE": """-- Toutes les Pages Wiki --
{all_pages}

-- Tache --
Genere un resume concis (200-400 mots) pour la page d'accueil du wiki qui synthetise les themes principaux, les entites cles et les relations importantes a travers toutes les pages. Utilise les wikilinks [[Nom de Page]] pour referencer les pages cles.""",
        },
        "crystallize": {
            "SYSTEM_PROMPT": "Tu es un assistant IA qui evalue si une paire question/reponse merite d'etre sauvegardee comme page wiki. Un bon candidat contient une synthese non triviale, une analyse utile, une comparaison ou une connexion qu'il serait precieux de conserver pour de futures requetes. Les reponses triviales ou evidentes ne doivent PAS etre cristallisees.",
            "QUERY_TEMPLATE": """-- Question --
{query}

-- Reponse --
{answer}

-- Tache --
Evalue si cette paire Q/R merite d'etre cristallisee en page wiki. Retourne un objet JSON :
{{
    "should_crystallize": true/false,
    "reason": "explication breve",
    "suggested_title": "Titre de Page Suggere",
    "suggested_summary": "Resume en une ligne (max 120 caracteres)",
    "suggested_tags": ["tag1", "tag2"],
    "page_content": "Le contenu markdown pour la page wiki si cristallisation. Doit inclure des [[wikilinks]] vers les concepts pertinents. Uniquement present si should_crystallize est true."
}}""",
        },
    },
}
