"""
Configuration centralisée pour l'Agentic Router RAG.
Évite les valeurs en dur dans le code.
"""

class AgenticRouterConfig:
    """Configuration centralisée avec valeurs par défaut."""

    # Parallélisation
    MAX_WORKERS = 5  # Nombre max de workers pour ThreadPoolExecutor
    TIMEOUT_PER_QUERY = 30  # Timeout en secondes par requête

    # Chunking
    DEFAULT_NB_CHUNKS = 7  # Nombre de chunks par requête
    MAX_NB_CHUNKS = 20  # Maximum de chunks à récupérer
    CHUNK_SIMILARITY_THRESHOLD = 0.9  # Seuil de similarité pour la déduplication

    # Confiance
    DEFAULT_CONFIDENCE = 0.6  # Confiance par défaut
    MIN_CONFIDENCE = 0.0  # Confiance minimum
    MAX_CONFIDENCE = 1.0  # Confiance maximum
    CONFIDENCE_THRESHOLD = 0.7  # Seuil pour l'auto-correction

    # Itérations
    MAX_ITERATIONS = 3  # Nombre max d'itérations d'auto-correction

    # Logging
    ENABLE_LOGGING = True
    LOG_LEVEL = 'INFO'

    # Fallback
    ENABLE_FALLBACK = True
    FALLBACK_NB_CHUNKS = 5  # Nombre de chunks en cas de fallback

    @classmethod
    def get_max_workers(cls, nb_queries: int) -> int:
        """Retourne le nombre max de workers en fonction du nombre de requêtes."""
        return min(nb_queries, cls.MAX_WORKERS)

    @classmethod
    def get_nb_chunks(cls, requested: int = None) -> int:
        """Retourne le nombre de chunks à utiliser."""
        if requested is None:
            return cls.DEFAULT_NB_CHUNKS
        return max(1, min(requested, cls.MAX_NB_CHUNKS))

    @classmethod
    def validate_confidence(cls, confidence: float) -> float:
        """Valide et normalise un score de confiance."""
        return max(cls.MIN_CONFIDENCE, min(confidence, cls.MAX_CONFIDENCE))
