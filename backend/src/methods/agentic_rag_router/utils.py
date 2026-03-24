"""
Utilitaires pour le traitement parallèle et le parsing de confiance.
"""
import re
from typing import Dict, Any

class ConfidenceParser:
    """Utilitaire pour parser les scores de confiance depuis les réponses LLM."""

    @staticmethod
    def parse_confidence(text: str, default: float = 0.6) -> float:
        """
        Parse le score de confiance depuis le texte avec plusieurs patterns supportés.

        Args:
            text: Texte contenant potentiellement un score de confiance
            default: Valeur par défaut si aucun score n'est trouvé

        Returns:
            Score de confiance entre 0.0 et 1.0
        """
        patterns = [
            r'(?:confidence|score|qualité)\s*[:=]\s*([0-9.]+)',
            r'(?:CONFIANCE|CONFIANCE|SCORE)\s*[:=]\s*([0-9.]+)',
            r'([0-9.]+)\s*%\s*(?:confiance|sureté)',
            r'estimation\s*:\s*([0-9.]+)',
            r'fiabilité\s*:\s*([0-9.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Normaliser si c'est un pourcentage
                    if value > 1.0:
                        value = value / 100.0
                    return max(0.0, min(1.0, value))  # Clamp entre 0 et 1
                except (ValueError, IndexError):
                    continue

        return default

    @staticmethod
    def safe_parse_json(text: str) -> Dict[str, Any]:
        """
        Parse du JSON de manière robuste avec gestion des erreurs.

        Args:
            text: Texte contenant du JSON

        Returns:
            Dictionnaire parsé ou dictionnaire vide en cas d'erreur
        """
        try:
            # Essayer le parsing direct
            return json.loads(text)
        except json.JSONDecodeError:
            # Essayer d'extraire le JSON du texte
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def extract_number(text: str, default: float = 0.0) -> float:
        """
        Extrait un nombre d'un texte de manière robuste.

        Args:
            text: Texte contenant potentiellement un nombre
            default: Valeur par défaut si aucun nombre n'est trouvé

        Returns:
            Nombre extrait ou valeur par défaut
        """
        match = re.search(r'([0-9.]+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return default
