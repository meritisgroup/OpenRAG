import json
import os
import pickle
import shutil
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from api.schemas.benchmark import (
    BenchmarkStartRequest, BenchmarkStatus, BenchmarkCompleteResult,
    GenerateQueriesRequest, GenerateQueriesResponse,
    BenchmarkReport, BenchmarkResult, BenchmarkRequest
)
from api.main import get_agent
from services.progress_tracker import ProgressTracker
from services.benchmark_orchestrator import BenchmarkOrchestrator
from services.plot_generator import PlotGenerator, _convert_to_serializable
from evaluation import end_to_end_evaluators
from evaluation.agent_evaluator import DataFramePreparator, AgentEvaluator
from factory.rag_registry import RAG_REGISTRY

router = APIRouter()

REPORT_PATH = 'data/report'


def _get_all_rags() -> dict:
    return {name: name for name in RAG_REGISTRY.keys()}


def _run_benchmark_task(
    benchmark_id: str,
    request: BenchmarkStartRequest
) -> None:
    try:
        all_rags = _get_all_rags()
        orchestrator = BenchmarkOrchestrator(benchmark_id, all_rags)
        orchestrator.run_full_benchmark(
            rag_names=request.rag_names,
            databases=request.databases,
            queries_doc_name=request.queries_doc_name,
            config=request.config,
            models_infos=request.models_infos,
            benchmark_type=request.benchmark_type,
            reset_index=request.reset_index,
            reset_preprocess=request.reset_preprocess
        )
    except Exception as e:
        tracker = ProgressTracker(benchmark_id)
        tracker.error(str(e))


@router.post("/start", response_model=BenchmarkStatus)
async def start_benchmark(
    request: BenchmarkStartRequest,
    background_tasks: BackgroundTasks
):
    if len(request.rag_names) < 1:
        raise HTTPException(status_code=400, detail="At least 1 RAG method required")
    
    benchmark_id = ProgressTracker.create_benchmark_id()
    tracker = ProgressTracker(benchmark_id)
    tracker.initialize()
    
    background_tasks.add_task(
        _run_benchmark_task,
        benchmark_id,
        request
    )
    
    return BenchmarkStatus(
        benchmark_id=benchmark_id,
        status='pending',
        progress=0.0,
        current_step='initializing'
    )


@router.get("/{benchmark_id}/status", response_model=BenchmarkStatus)
def get_benchmark_status(benchmark_id: str):
    tracker = ProgressTracker(benchmark_id)
    status = tracker.get_status()
    
    return BenchmarkStatus(
        benchmark_id=benchmark_id,
        status=status.get('status', 'not_found'),
        progress=status.get('progress', 0.0),
        current_step=status.get('current_step', ''),
        error=status.get('error'),
        started_at=status.get('started_at'),
        completed_at=status.get('completed_at')
    )


@router.get("/{benchmark_id}/result", response_model=BenchmarkCompleteResult)
def get_benchmark_result(benchmark_id: str):
    tracker = ProgressTracker(benchmark_id)
    status = tracker.get_status()
    
    if status.get('status') == 'not_found':
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    if status.get('status') == 'error':
        raise HTTPException(status_code=500, detail=status.get('error', 'Unknown error'))
    
    if status.get('status') not in ('completed', 'running'):
        progress_file = os.path.join(tracker.report_dir, 'progress.json')
        if not os.path.exists(progress_file):
            raise HTTPException(
                status_code=400, 
                detail=f"Benchmark not completed. Current status: {status.get('status')}"
            )
    
    results_file = os.path.join(tracker.report_dir, 'results_bench.pkl')
    if not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")
    
    scores = results.get('scores')
    if not scores:
        scores = _extract_scores_from_results(results)
    
    scores = _convert_to_serializable(scores)
    
    plots = results.get('plots', {})
    all_rags = _get_all_rags()
    plot_gen = PlotGenerator(all_rags)
    benchmark_type = results.get('type_bench', 'all')
    
    results_with_scores = results.copy()
    results_with_scores['scores'] = scores
    
    if not plots:
        try:
            plots = plot_gen.generate_all_plots(results_with_scores, benchmark_type)
        except Exception as e:
            print(f"Error generating plots: {e}")
            plots = {}
    else:
        missing_graphs = [k for k in ['context_graph', 'ground_truth_graph'] if k not in plots]
        
        if missing_graphs:
            try:
                new_plots = plot_gen.generate_all_plots(results_with_scores, benchmark_type)
                for k in missing_graphs:
                    if k in new_plots and new_plots[k]:
                        plots[k] = new_plots[k]
            except Exception as e:
                print(f"Error generating missing plots: {e}")
    
    plots = _convert_to_serializable(plots)
    
    files = {
        'results_pickle': os.path.join(tracker.report_dir, 'results_bench.pkl'),
        'csv': os.path.join(tracker.report_dir, 'bench_df.csv'),
        'excel': os.path.join(tracker.report_dir, 'answers.xlsx'),
        'config': os.path.join(tracker.report_dir, 'config_server.json'),
        'pdf': os.path.join(tracker.report_dir, 'plot_report.pdf')
    }
    
    databases = results.get('databases', [])
    df = results.get('df')
    rag_names = list(df.columns[2:]) if df is not None else []
    
    return BenchmarkCompleteResult(
        benchmark_id=benchmark_id,
        status='completed',
        scores=scores,
        files=files,
        plots=plots,
        databases=databases,
        rag_names=rag_names
    )


def _extract_scores_from_results(results: dict) -> dict:
    existing_scores = results.get('scores', {})
    
    if existing_scores:
        return existing_scores
    
    scores = {
        'type': results.get('type_bench', 'unknown'),
        'ground_truth': results.get('ground_truth_scores', {})
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


@router.get("/{benchmark_id}/download/{file_type}")
def download_benchmark_file(benchmark_id: str, file_type: str):
    tracker = ProgressTracker(benchmark_id)
    
    file_map = {
        'csv': 'bench_df.csv',
        'excel': 'answers.xlsx',
        'pdf': 'plot_report.pdf',
        'config': 'config_server.json',
        'results': 'results_bench.pkl'
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail=f"Unknown file type: {file_type}")
    
    file_path = os.path.join(tracker.report_dir, file_map[file_type])
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
    
    return FileResponse(
        file_path,
        filename=file_map[file_type],
        media_type='application/octet-stream'
    )


@router.post("/generate-queries", response_model=GenerateQueriesResponse)
def generate_queries(request: GenerateQueriesRequest):
    import traceback
    try:
        queries, answers, file_path = BenchmarkOrchestrator.generate_questions(
            databases=request.databases,
            n_questions=request.n_questions,
            config=request.config,
            models_infos=request.models_infos
        )
        
        return GenerateQueriesResponse(
            queries=queries,
            answers=answers,
            file_path=file_path
        )
    except Exception as e:
        print(f"Error generating queries: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
def list_reports():
    if not os.path.exists(REPORT_PATH):
        return {"reports": []}
    
    reports = []
    for foldername in os.listdir(REPORT_PATH):
        folder = os.path.join(REPORT_PATH, foldername)
        pkl_file = os.path.join(folder, 'results_bench.pkl')
        progress_file = os.path.join(folder, 'progress.json')
        
        if os.path.isdir(folder):
            status = 'unknown'
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    status = progress_data.get('status', 'unknown')
            
            if os.path.exists(pkl_file):
                try:
                    with open(pkl_file, 'rb') as f:
                        results = pickle.load(f)
                    reports.append(BenchmarkReport(
                        report_id=foldername,
                        created_at=foldername,
                        rag_names=list(results.get('df', {}).columns[2:]) if 'df' in results else [],
                        databases=results.get('databases', []),
                        type=results.get('type_bench', 'unknown')
                    ))
                except Exception as e:
                    print(f"Skipping incompatible report {foldername}: {e}")
                    reports.append(BenchmarkReport(
                        report_id=foldername,
                        created_at=foldername,
                        rag_names=[],
                        databases=[],
                        type='error'
                    ))
            elif status in ('running', 'pending'):
                reports.append(BenchmarkReport(
                    report_id=foldername,
                    created_at=foldername,
                    rag_names=[],
                    databases=[],
                    type=status
                ))
    
    return {"reports": reports}


@router.get("/report/{report_id}", response_model=BenchmarkResult)
def get_report(report_id: str):
    folder = os.path.join(REPORT_PATH, report_id)
    pkl_file = os.path.join(folder, 'results_bench.pkl')
    
    if not os.path.exists(pkl_file):
        raise HTTPException(status_code=404, detail="Report not found")
    
    try:
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load report: {str(e)}")
    
    return BenchmarkResult(
        benchmark_id=report_id,
        status="completed",
        rag_names=list(results.get('df', {}).columns[2:]) if 'df' in results else [],
        databases=results.get('databases', []),
        results={
            'type': results.get('type_bench', 'unknown'),
            'ground_truth_scores': results.get('ground_truth_scores', {}),
            'arena_scores': results.get('arena_scores', {})
        }
    )


@router.delete("/report/{report_id}")
def delete_report(report_id: str):
    folder = os.path.join(REPORT_PATH, report_id)
    
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Report not found")
    
    shutil.rmtree(folder)
    return {"status": "deleted", "report_id": report_id}


@router.post("/run")
def run_benchmark(request: BenchmarkRequest):
    rag_agents = []
    for session_id in request.session_ids:
        agent = get_agent(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent not found for session {session_id}")
        rag_agents.append(agent)
    
    timestamp = datetime.now().strftime('%m-%d_%H-%M-%S')
    benchmark_id = f"bench_{timestamp}"
    report_dir = os.path.join(REPORT_PATH, timestamp)
    os.makedirs(report_dir, exist_ok=True)
    
    log_file = os.path.join(report_dir, 'logs.json')
    with open(log_file, 'w') as f:
        json.dump({
            'indexation': 0.0,
            'answers': 0.0,
            'Arena Battles': 0.0,
            'Ground Truth comparison': 0.0
        }, f)
    
    queries_path = os.path.join('data', 'queries', request.queries_doc_name)
    
    dataframe_preparator = DataFramePreparator(
        rag_agents=rag_agents,
        rags_available=request.rag_names,
        input_path=queries_path
    )
    dataframe_preparator.run_all_queries(
        options_generation={'type_generation': 'simple_generation'},
        log_file=log_file
    )
    df = dataframe_preparator.get_dataframe()
    
    if request.benchmark_type == 'all':
        evaluation_agent = AgentEvaluator(
            dataframe=df,
            rags_available=request.rag_names,
            config_server=request.config,
            models_infos=request.models_infos
        )
        evals = evaluation_agent.get_evals(log_file=log_file)
        results = {
            'type_bench': 'all',
            'df': df,
            'evals': evals,
            'ground_truth_scores': evaluation_agent.ground_truth_comparator.all_scores_dict,
            'arena_scores': evaluation_agent.arena.all_scores_dict,
            'indexations_tokens': dataframe_preparator.indexation_tokens,
            'databases': request.databases
        }
    else:
        evaluation_agent = AgentEvaluator(
            dataframe=df,
            rags_available=request.rag_names,
            config_server=request.config,
            models_infos=request.models_infos
        )
        evals = evaluation_agent.get_evals(log_file=log_file, type='ground_truth')
        results = {
            'type_bench': 'ground_truth',
            'df': df,
            'evals': evals,
            'ground_truth_scores': evaluation_agent.ground_truth_comparator.all_scores_dict,
            'indexations_tokens': dataframe_preparator.indexation_tokens,
            'databases': request.databases
        }
    
    results_file = os.path.join(report_dir, 'results_bench.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    df.to_csv(os.path.join(report_dir, 'bench_df.csv'), index=False)
    
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        status="completed",
        rag_names=request.rag_names,
        databases=request.databases,
        results={
            'report_path': report_dir,
            'type': results['type_bench']
        }
    )
