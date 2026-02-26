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
            status = json.load(f)
        
        progress = status.get('progress', 0.0)
        current_step = status.get('current_step', '')
        
        if 60 <= progress < 85 and current_step == 'Running evaluations':
            logs = self._read_logs_with_fallback()
            if logs:
                detailed_progress = self._calculate_detailed_progress(progress)
                if detailed_progress is not None:
                    status['progress'] = detailed_progress
                    status['detailed_info'] = self._get_detailed_status()
                    current_metric = self._find_current_metric(logs)
                    if current_metric:
                        status['current_step'] = f'Running {current_metric}'
        
        return status
    
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
    
    def _read_logs_with_fallback(self) -> Dict[str, Any]:
        try:
            return self.read_log()
        except Exception:
            return {}
    
    def _calculate_detailed_progress(self, base_progress: float = 60) -> Optional[float]:
        logs = self._read_logs_with_fallback()
        
        if not logs:
            return None
        
        metric_weights = {
            'nDCG score': 0.20,
            'Arena Battles': 0.30,
            'Ground Truth comparison': 0.25,
            'Context faithfulness': 0.15,
            'context relevance': 0.10
        }
        
        total_weighted_progress = 0.0
        
        for metric_name, weight in metric_weights.items():
            metric_progress = logs.get(metric_name, 0.0)
            if metric_progress > 0:
                total_weighted_progress += weight * (metric_progress / 100)
        
        eval_progress = base_progress + 25 * total_weighted_progress
        
        return eval_progress
    
    def _find_current_metric(self, logs: Dict[str, Any], completed_count: int = 0) -> Optional[str]:
        metric_order = ['nDCG score', 'Arena Battles', 'Ground Truth comparison', 
                       'Context faithfulness', 'context relevance']
        
        for metric_name in metric_order:
            progress = logs.get(metric_name, 0.0)
            if progress > 0 and progress < 100:
                return metric_name
            if progress >= 100:
                completed_count -= 1
        
        if completed_count >= 0 and completed_count < len(metric_order):
            return metric_order[completed_count] if completed_count < len(metric_order) else None
        
        return None
    
    def _get_detailed_status(self) -> Dict[str, Any]:
        logs = self._read_logs_with_fallback()
        
        return {
            'metrics': {
                metric_name: {
                    'progress': logs.get(metric_name, 0.0),
                    'status': 'completed' if logs.get(metric_name, 0.0) >= 100 
                             else 'running' if logs.get(metric_name, 0.0) > 0 
                             else 'pending'
                }
                for metric_name in ['nDCG score', 'Arena Battles', 'Ground Truth comparison',
                                   'Context faithfulness', 'context relevance']
            }
        }
