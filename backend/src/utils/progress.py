import logging
import os
from typing import Optional, Iterator, Any

logger = logging.getLogger(__name__)
LOG_INTERVAL = int(os.getenv('RAG_LOG_INTERVAL', '10'))

class ProgressBar:
    
    def __init__(self, iterable=None, total=None, desc=None):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else None)
        self.desc = desc or "Processing"
        self.current = 0
        self.last_logged_percent = -1
        self._index = 0
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


tqdm = ProgressBar
