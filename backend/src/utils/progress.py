import logging
import os
from typing import Optional, Iterator, Any, Callable

logger = logging.getLogger(__name__)
LOG_INTERVAL = int(os.getenv('RAG_LOG_INTERVAL', '10'))

class ProgressBar:
    
    def __init__(self, iterable=None, total=None, desc=None, callback=None):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else None)
        self.desc = desc or "Processing"
        self.current = 0
        self.last_logged_percent = -1
        self._index = 0
        self.callback = callback
        if self.total:
            logger.info(f"[{self.desc}] Started - Total: {self.total}")
    
    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            raise TypeError("ProgressBar is not iterable without an iterable argument")
        self._index = 0
        for item in self.iterable:
            yield item
            self._index += 1
            self._log_progress()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        return False
    
    def __len__(self):
        return len(self.iterable) if self.iterable else self.total or 0
    
    def _log_progress(self):
        if self.total:
            self.current = self._index + 1
            percent = int((self.current / self.total) * 100)
            if percent - self.last_logged_percent >= LOG_INTERVAL:
                self.last_logged_percent = percent
                logger.info(f"[{self.desc}] {percent}% ({self.current}/{self.total})")
        if self.callback:
            progress = int((self.current / self.total) * 100) if self.total and self.current > 0 else 0
            self.callback(progress, f"{self.desc}: {self.current}/{self.total}")
    
    def set_description(self, desc):
        self.desc = desc
    
    def update(self, index=None, text=None):
        if index is not None:
            self._index = index
        self._log_progress()
    
    def message(self, text=None):
        if text:
            logger.info(f"[{self.desc}] {text}")
    
    def success(self, value):
        logger.info(f"[{self.desc}] ✓ {value}")
    
    def error(self, value):
        logger.error(f"[{self.desc}] ✗ {value}")
    
    def info(self, value):
        logger.info(f"[{self.desc}] ℹ {value}")
    
    def clear(self):
        if self.total and self.current >= self.total:
            logger.info(f"[{self.desc}] Completed")


class IndexationProgressTracker:
    def __init__(self, total: int, callback: Callable[[float, str], None], desc: str = "Indexation"):
        self.total = total
        self.callback = callback
        self.desc = desc
        self.current = 0
    
    def update(self, current: int, message: str = ""):
        self.current = current
        progress = float(current) / self.total * 100 if self.total > 0 else 0.0
        msg = f"{self.desc}: {current}/{self.total}" if not message else f"{self.desc}: {message}"
        self.callback(progress, msg)
    
    def increment(self, message: str = ""):
        self.current += 1
        progress = float(self.current) / self.total * 100 if self.total > 0 else 0.0
        msg = f"{self.desc}: {self.current}/{self.total}" if not message else f"{self.desc}: {message}"
        self.callback(progress, msg)


class TwoLevelProgressTracker:
    """
    Tracker pour progression à deux niveaux (global + sous-étape).
    Utilisé principalement pour GraphRAG.
    """
    def __init__(self, total_steps: int, callback: Callable):
        self.total_steps = total_steps
        self.callback = callback
        self.current_step = 0
        self.sub_total = 0
        self.sub_current = 0
    
    def set_sub_total(self, total: int):
        """Définit le total pour la sous-étape actuelle"""
        self.sub_total = total
        self.sub_current = 0
    
    def update_global(self, message: str):
        """Met à jour la progression globale uniquement"""
        progress = self.current_step * 100 / self.total_steps if self.total_steps > 0 else 100.0
        self.callback(progress, message, 0.0, "")
    
    def update_sub(self, sub_current: int, sub_message: str):
        """Met à jour la sous-progression (ex: documents dans extraction)"""
        self.sub_current = sub_current
        global_progress = self.current_step * 100 / self.total_steps
        sub_progress = self.sub_current * 100 / self.sub_total if self.sub_total > 0 else 0.0
        self.callback(global_progress, "", sub_progress, sub_message)
    
    def increment_sub(self, sub_message: str = ""):
        """Incrémente la sous-progression"""
        self.sub_current += 1
        self.update_sub(self.sub_current, sub_message)
    
    def complete_step(self, message: str = ""):
        """Termine l'étape globale actuelle"""
        self.current_step += 1
        self.sub_total = 0
        self.sub_current = 0
        progress = self.current_step * 100 / self.total_steps if self.total_steps > 0 else 100.0
        self.callback(progress, message, 0.0, "")
    
    def complete_all(self):
        """Termine toute l'indexation"""
        self.callback(100.0, "Indexation completed", 100.0, "")


tqdm = ProgressBar
