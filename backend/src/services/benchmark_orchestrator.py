import concurrent.futures
import os
import json
import pickle
import random
import re
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from pydantic import BaseModel

from factory import RAGFactory
from evaluation.agent_evaluator import DataFramePreparator, AgentEvaluator
from evaluation import end_to_end_evaluators
from database.utils import get_list_path_documents
from utils.agent import get_Agent
from utils.open_doc import Opener
from .progress_tracker import ProgressTracker
from .plot_generator import PlotGenerator


class BenchmarkOrchestrator:
    def __init__(self, benchmark_id: str, all_rags: Dict[str, str]):
        self.benchmark_id = benchmark_id
        self.tracker = ProgressTracker(benchmark_id)
        self.plot_generator = PlotGenerator(all_rags)
        self.report_dir = self.tracker.report_dir
        self.factory = RAGFactory()
    
    def run_full_benchmark(
        self,
        rag_names: List[str],
        databases: List[str],
        queries_doc_name: str,
        config: Dict[str, Any],
        models_infos: Dict[str, Any],
        benchmark_type: str = 'full_bench',
        reset_index: bool = False,
        reset_preprocess: bool = False
    ) -> Dict[str, Any]:
        try:
            self.tracker.initialize()
            self._init_logs()
            
            rag_agents = self._create_agents(
                rag_names, databases, config, models_infos
            )
            
            self._run_indexation(
                rag_agents, rag_names, reset_index, reset_preprocess
            )
            
            df, indexation_tokens = self._run_queries(
                rag_agents, rag_names, queries_doc_name, benchmark_type
            )
            
            if benchmark_type in ('ground_truth', 'full_bench'):
                results = self._run_evaluations(
                    df, rag_names, config, models_infos, benchmark_type
                )
            else:
                results = {
                    'type_bench': benchmark_type,
                    'df': df,
                    'indexations_tokens': indexation_tokens,
                    'databases': databases
                }
            
            plots = self._generate_plots(results, benchmark_type)
            results['plots'] = plots
            
            scores = self._extract_scores(results)
            results['scores'] = scores
            
            self._generate_pdf_report(results, plots)
            
            files = self._save_results(results, config)
            
            self.tracker.complete({
                'rag_names': rag_names,
                'databases': databases,
                'type': benchmark_type
            })
            
            return {
                'benchmark_id': self.benchmark_id,
                'status': 'completed',
                'scores': self._extract_scores(results),
                'files': files,
                'plots': plots
            }
            
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.tracker.error(error_msg)
            raise
    
    def _init_logs(self) -> None:
        self.tracker.write_log({
            'indexation': 0.0,
            'answers': 0.0,
            'Arena Battles': 0.0,
            'Ground Truth comparison': 0.0,
            'Context faithfulness': 0.0,
            'context relevance': 0.0,
            'nDCG score': 0.0
        })
    
    def _create_agents(
        self,
        rag_names: List[str],
        databases: List[str],
        config: Dict[str, Any],
        models_infos: Dict[str, Any]
    ) -> List[Any]:
        self.tracker.update(5, 'Creating agents', 'running')
        
        agents = []
        for i, rag_name in enumerate(rag_names):
            agent = self.factory.get_agent(
                rag_name=rag_name,
                config_server=config,
                models_infos=models_infos,
                databases_name=databases,
                custom=False
            )
            agents.append(agent)
            progress = 5 + (i + 1) / len(rag_names) * 5
            self.tracker.update(progress, f'Creating agent {rag_name}', 'running')
        
        return agents
    
    def _run_indexation(
        self,
        rag_agents: List[Any],
        rag_names: List[str],
        reset_index: bool,
        reset_preprocess: bool
    ) -> None:
        self.tracker.update(10, 'Indexation', 'running')
        
        current_reset_preprocess = reset_preprocess
        n = len(rag_agents)
        
        for i, (agent, rag_name) in enumerate(zip(rag_agents, rag_names)):
            agent.indexation_phase(
                reset_index=reset_index,
                reset_preprocess=current_reset_preprocess
            )
            if current_reset_preprocess:
                current_reset_preprocess = False
            
            progress = 10 + (i + 1) / n * 20
            self.tracker.update(progress, f'Indexing {rag_name}', 'running')
            self.tracker.update_log_step('indexation', int((i + 1) / n * 100))
    
    def _run_queries(
        self,
        rag_agents: List[Any],
        rag_names: List[str],
        queries_doc_name: str,
        benchmark_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.tracker.update(30, 'Running queries', 'running')
        
        queries_path = os.path.join('data', 'queries', queries_doc_name)
        log_file = os.path.join(self.report_dir, 'logs.json')
        
        dataframe_preparator = DataFramePreparator(
            rag_agents=rag_agents,
            rags_available=rag_names,
            input_path=queries_path
        )
        
        if benchmark_type == 'contexts':
            options = {'type_generation': 'no_generation'}
        else:
            options = {'type_generation': 'simple_generation'}
        
        n = len(rag_names)
        for i in range(n):
            progress = 30 + (i + 1) / n * 30
            self.tracker.update(progress, f'Generating answers ({i+1}/{n})', 'running')
        
        dataframe_preparator.run_all_queries(
            options_generation=options,
            log_file=log_file
        )
        
        df = dataframe_preparator.get_dataframe()
        indexation_tokens = dataframe_preparator.indexation_tokens
        
        self.tracker.update(60, 'Queries completed', 'running')
        
        return df, indexation_tokens
    
    def _run_evaluations(
        self,
        df: pd.DataFrame,
        rag_names: List[str],
        config: Dict[str, Any],
        models_infos: Dict[str, Any],
        benchmark_type: str
    ) -> Dict[str, Any]:
        self.tracker.update(60, 'Running evaluations', 'running')
        
        log_file = os.path.join(self.report_dir, 'logs.json')
        
        evaluation_agent = AgentEvaluator(
            dataframe=df,
            rags_available=rag_names,
            config_server=config,
            models_infos=models_infos
        )
        
        eval_type = 'all' if benchmark_type == 'full_bench' else 'ground_truth'
        
        self.tracker.update(65, 'Running evaluations', 'running')
        evals = evaluation_agent.get_evals(log_file=log_file, type=eval_type)
        self.tracker.update(85, 'Evaluations completed', 'running')
        
        results = {
            'type_bench': benchmark_type,
            'df': df,
            'evals': evals,
            'ground_truth_scores': evaluation_agent.ground_truth_comparator.all_scores_dict,
            'indexations_tokens': {},
            'databases': []
        }
        
        if benchmark_type == 'full_bench':
            results['arena_scores'] = evaluation_agent.arena.all_scores_dict
        
        return results
    
    def _generate_plots(
        self,
        results: Dict[str, Any],
        benchmark_type: str
    ) -> Dict[str, Any]:
        self.tracker.update(90, 'Generating plots', 'running')
        
        try:
            plots = self.plot_generator.generate_all_plots(results, benchmark_type)
        except Exception as e:
            print(f"Error generating plots: {e}")
            plots = {}
        
        return plots
    
    def _generate_pdf_report(
        self,
        results: Dict[str, Any],
        plots: Dict[str, Any]
    ) -> None:
        self.tracker.update(92, 'Generating PDF report', 'running')
        
        try:
            temp_evaluator = AgentEvaluator.__new__(AgentEvaluator)
            temp_evaluator.create_plot_report(plots, self.report_dir)
            pdf_path = os.path.join(self.report_dir, 'plot_report.pdf')
            if os.path.exists(pdf_path):
                print(f"PDF report generated successfully: {pdf_path}")
            else:
                print(f"Warning: PDF report was not created at {pdf_path}")
        except Exception as e:
            import traceback
            print(f"Error generating PDF report: {e}")
            print(traceback.format_exc())
    
    def _save_results(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        self.tracker.update(95, 'Saving results', 'running')
        
        results_file = os.path.join(self.report_dir, 'results_bench.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        df = results.get('df')
        if df is not None:
            csv_path = os.path.join(self.report_dir, 'bench_df.csv')
            df.to_csv(csv_path, index=False)
            
            df_to_save = df.copy()
            for rag in df.columns[2:]:
                df_to_save[rag] = df[rag].apply(
                    lambda d: d.get('ANSWER', '') if isinstance(d, dict) else ''
                )
            excel_path = os.path.join(self.report_dir, 'answers.xlsx')
            df_to_save.to_excel(excel_path, index=False, engine='openpyxl')
        
        config_path = os.path.join(self.report_dir, 'config_server.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        return {
            'results_pickle': results_file,
            'csv': os.path.join(self.report_dir, 'bench_df.csv'),
            'excel': os.path.join(self.report_dir, 'answers.xlsx'),
            'config': config_path,
            'pdf': os.path.join(self.report_dir, 'plot_report.pdf')
        }
    
    def _extract_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        scores = {
            'type': results.get('type_bench', 'unknown'),
            'ground_truth': results.get('ground_truth_scores', {}),
            'answers_scores': dict(end_to_end_evaluators.SCORES)
        }
        
        if 'arena_scores' in results:
            scores['arena'] = results['arena_scores']
        
        if 'evals' in results:
            evals = results['evals']
            if 'context_faithfulness_scores' in evals:
                scores['context_faithfulness'] = evals['context_faithfulness_scores']
            if 'context_relevance_scores' in evals:
                scores['context_relevance'] = evals['context_relevance_scores']
            if 'ndcg_scores' in evals:
                scores['ndcg'] = evals['ndcg_scores']
        
        return scores
    
    @staticmethod
    def generate_questions(
        databases: List[str],
        n_questions: int,
        config: Dict[str, Any],
        models_infos: Dict[str, Any]
    ) -> Tuple[List[str], List[str], str]:
        class QuestionOnPage(BaseModel):
            query: str
            answer: str
        
        question_generation_prompt = """
        You are an assistant specialized in generating questions to test RAG systems.
        Generate ONE clear question based on the content provided.
        Include sufficient context in the question.
        Provide the correct answer based solely on the content.
        """
        
        databases_path = [os.path.abspath(os.path.join('data', 'databases', db)) for db in databases]
        agent = get_Agent(config, models_infos=models_infos)
        
        all_files = []
        for db_path in databases_path:
            if not os.path.exists(db_path):
                print(f"Warning: Database path does not exist: {db_path}")
                continue
            files = get_list_path_documents(db_path)
            for f in files:
                abs_f = os.path.abspath(f)
                if os.path.exists(abs_f):
                    all_files.append(abs_f)
                else:
                    print(f"Warning: File listed but does not exist: {abs_f}")
        
        print(f"Found {len(all_files)} valid files in databases")
        if not all_files:
            raise ValueError(f"No documents found in databases: {databases}. Checked paths: {databases_path}")
        
        def get_random_context(databases_path, max_len=16000):
            file_path = random.choice(all_files)
            
            if file_path.endswith('.pdf'):
                old_path = Path(file_path)
                possible_paths = [
                    old_path.parent / 'md_without_images' / (old_path.stem + '.md'),
                    old_path.parent / 'pdf_text_extraction' / (old_path.stem + '.txt'),
                    old_path.with_name(old_path.stem + '.md'),
                    old_path.with_name(old_path.stem + '.txt'),
                ]
                for alt_path in possible_paths:
                    if alt_path.exists():
                        file_path = str(alt_path)
                        break
            
            content = Opener(save=False).open_doc(file_path)
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            if not sentences:
                return content[:max_len]
            
            index = random.randint(0, len(sentences) - 1)
            context = [sentences[index]]
            total_length = len(sentences[index])
            i_prev, i_next = index - 1, index + 1
            
            while total_length < max_len and (i_prev >= 0 or i_next < len(sentences)):
                if i_prev >= 0:
                    context.insert(0, sentences[i_prev])
                    total_length += len(sentences[i_prev])
                    i_prev -= 1
                if total_length >= max_len:
                    break
                if i_next < len(sentences):
                    context.append(sentences[i_next])
                    total_length += len(sentences[i_next])
                    i_next += 1
            
            return ' '.join(context)
        
        def generate_question():
            content = ''
            num_try = 0
            while len(content) < 100 and num_try < 3:
                content = get_random_context(databases_path)
                num_try += 1
            
            prompt = question_generation_prompt + f'\n\n<Content>\n{content}\n</Content>\n'
            response = agent.predict_json(
                prompt=prompt,
                model=config.get('model', ''),
                system_prompt=question_generation_prompt,
                json_format=QuestionOnPage
            )
            return response
        
        max_workers = config.get('max_workers', 1)
        if max_workers <= 1:
            max_workers = 1
        
        questions = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_question) for _ in range(n_questions)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    questions.append(result)
                except Exception as e:
                    error_msg = f"Error generating question: {e}"
                    print(error_msg)
                    errors.append(error_msg)
        
        if not questions and errors:
            raise RuntimeError(f"Failed to generate any questions. Errors: {'; '.join(errors)}")
        
        list_queries = [q.query for q in questions if q]
        list_answers = [q.answer for q in questions if q]
        
        file_path = './data/queries/generated_queries.xlsx'
        df = pd.DataFrame({'query': list_queries, 'answer': list_answers})
        
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_excel(file_path, index=False)
        
        return list_queries, list_answers, file_path
