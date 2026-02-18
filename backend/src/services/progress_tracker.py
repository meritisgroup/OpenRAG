import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class ProgressTracker:
    REPORT_PATH = 'data/report'
    
    def __init__(self, benchmark_id: str):
        self.benchmark_id = benchmark_id
        self.report_dir = os.path.join(self.REPORT_PATH, benchmark_id)
        self.progress_file = os.path.join(self.report_dir, 'progress.json')
        os.makedirs(self.report_dir, exist_ok=True)
    
    @classmethod
    def create_benchmark_id(cls) -> str:
        timestamp = datetime.now().strftime('%m-%d_%H-%M-%S')
        return timestamp
    
    @classmethod
    def get_report_dir(cls, benchmark_id: str) -> str:
        return os.path.join(cls.REPORT_PATH, benchmark_id)
    
    def initialize(self) -> None:
        self._write_progress({
            'status': 'pending',
            'progress': 0.0,
            'current_step': 'initializing',
            'error': None,
            'started_at': datetime.now().isoformat()
        })
    
    def update(self, progress: float, step: str, status: str = 'running') -> None:
        self._write_progress({
            'status': status,
            'progress': progress,
            'current_step': step,
            'error': None
        })
    
    def complete(self, results_summary: Optional[Dict[str, Any]] = None) -> None:
        data = {
            'status': 'completed',
            'progress': 100.0,
            'current_step': 'done',
            'error': None,
            'completed_at': datetime.now().isoformat()
        }
        if results_summary:
            data['results_summary'] = results_summary
        self._write_progress(data)
    
    def error(self, error_message: str) -> None:
        self._write_progress({
            'status': 'error',
            'progress': 0.0,
            'current_step': 'error',
            'error': error_message
        })
    
    def get_status(self) -> Dict[str, Any]:
        if not os.path.exists(self.progress_file):
            return {
                'status': 'not_found',
                'progress': 0.0,
                'current_step': '',
                'error': 'Benchmark not found'
            }
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_progress(self, data: Dict[str, Any]) -> None:
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def write_log(self, log_data: Dict[str, Any]) -> None:
        log_file = os.path.join(self.report_dir, 'logs.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
    
    def read_log(self) -> Dict[str, Any]:
        log_file = os.path.join(self.report_dir, 'logs.json')
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def update_log_step(self, step: str, value: float) -> None:
        logs = self.read_log()
        logs[step] = value
        self.write_log(logs)
