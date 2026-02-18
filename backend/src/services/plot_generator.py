import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from math import sqrt
from typing import Dict, Any, Optional, List
import ast
import numpy as np


COLOR_DISCRETE_SEQUENCE = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
]


def _convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(item) for item in obj)
    return obj


class PlotGenerator:
    def __init__(self, all_rags: Dict[str, str]):
        self.all_rags = all_rags
    
    def generate_all_plots(self, results: Dict[str, Any], benchmark_type: str = 'all') -> Dict[str, Any]:
        plots = {}
        
        plots['token_graph'] = self.token_graph(
            results['df'], 
            results['indexations_tokens']
        )
        plots['time_graph'] = self.time_graph(results['df'])
        
        if benchmark_type == 'all':
            plots['context_graph'] = self.context_graph(
                results['evals'].get('context_faithfulness_scores', {}),
                results['evals'].get('context_relevance_scores', {}),
                results['evals'].get('ndcg_scores', {})
            )
        
        if benchmark_type in ('all', 'ground_truth'):
            plots['ground_truth_graph'] = self.ground_truth_graph(
                results.get('ground_truth_scores', {})
            )
        
        if benchmark_type == 'all':
            arena_graphs = self.arena_graphs(
                results.get('arena_scores', {})
            )
            plots['arena_graphs'] = arena_graphs
            plots['report_arena_graph'] = self.report_arena_graph(arena_graphs)
            
            impact = self.extract_impact(results['df'])
            energy = self.extract_energy(results['df'])
            plots['impact_graph'] = self.impact_graph(impact)
            plots['energy_graph'] = self.energy_graph(energy)
        
        return plots
    
    def token_graph(self, df: pd.DataFrame, indexation_tokens: Dict[str, Any]) -> Dict[str, Any]:
        all_queries = df
        tokens = {}
        list_rags = list(df.columns)[2:]
        
        for rag in list_rags:
            tokens[rag] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'embedding_tokens': indexation_tokens.get(rag, {}).get('embedding_tokens', 0),
                'indexation_input_tokens': indexation_tokens.get(rag, {}).get('input_tokens', 0),
                'indexation_output_tokens': indexation_tokens.get(rag, {}).get('output_tokens', 0)
            }
            for query in all_queries[rag]:
                q = self._parse_query(query)
                tokens[rag]['input_tokens'] += q.get('INPUT_TOKENS', 0)
                tokens[rag]['output_tokens'] += q.get('OUTPUT_TOKENS', 0)
        
        ticksval = []
        data = []
        for rag in tokens.keys():
            ticksval.append(rag)
            data.append({'RAG Method': rag, 'Token Type': 'Query Input Tokens', 'Nb Tokens': tokens[rag]['input_tokens']})
            data.append({'RAG Method': rag, 'Token Type': 'Query Output Tokens', 'Nb Tokens': tokens[rag]['output_tokens']})
            data.append({'RAG Method': rag, 'Token Type': 'Embedding Tokens', 'Nb Tokens': tokens[rag]['embedding_tokens']})
            data.append({'RAG Method': rag, 'Token Type': 'Indexation Input Tokens', 'Nb Tokens': tokens[rag]['indexation_input_tokens']})
            data.append({'RAG Method': rag, 'Token Type': 'Indexation Output Tokens', 'Nb Tokens': tokens[rag]['indexation_output_tokens']})
        
        tickstext = [self.all_rags.get(tick, tick) for tick in ticksval]
        
        fig = px.bar(
            data, x='RAG Method', y='Nb Tokens', color='Token Type',
            barmode='stack', title='Token Consumption',
            labels={'Nb Tokens': 'Nb Tokens', 'RAG Method': 'RAG Method'},
            color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
        )
        fig.update_layout(
            title=dict(text='Token Consumption', x=0.5, xanchor='center'),
            yaxis={'title': {'text': 'Nb Tokens'}},
            xaxis={'title': {'text': ''}, 'tickvals': ticksval, 'ticktext': tickstext},
            legend={'title': ''},
            legend_traceorder='reversed'
        )
        
        return fig.to_dict()
    
    def time_graph(self, df: pd.DataFrame) -> Dict[str, Any]:
        list_rags = list(df.columns)[2:]
        time_data = {}
        
        for rag in list_rags:
            query = df[rag].iloc[0] if len(df) > 0 else {}
            q = self._parse_query(query)
            time_data[rag] = q.get('TIME', 0)
        
        data = []
        for rag in time_data:
            data.append({
                'RAG Method': self.all_rags.get(rag, rag),
                'Answering Time': time_data[rag]
            })
        
        fig = px.bar(
            data, x='RAG Method', y='Answering Time',
            color='RAG Method', color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
        )
        fig.update_layout(
            title=dict(text='Answering Time comparison', x=0.5, xanchor='center'),
            legend={'title': {'text': ''}},
            yaxis={'title': {'text': 'Answering Time (s)'}}
        )
        
        return fig.to_dict()
    
    def context_graph(self, faithfulness: Dict[str, float], relevance: Dict[str, float], 
                      ndcg: Dict[str, float]) -> Dict[str, Any]:
        ticksval = []
        data = []
        
        for rag in relevance.keys():
            ticksval.append(rag)
            data.append({'RAG Method': rag, 'Score': relevance[rag], 'Metric': 'Context Relevance'})
        
        for rag in faithfulness.keys():
            data.append({'RAG Method': rag, 'Score': faithfulness[rag], 'Metric': 'Context Faithfulness'})
        
        for rag in ndcg.keys():
            data.append({'RAG Method': rag, 'Score': ndcg[rag], 'Metric': 'Context nDCG Score'})
        
        tickstext = [self.all_rags.get(tick, tick) for tick in ticksval]
        
        fig = px.bar(
            data, x='Score', y='RAG Method', color='Metric',
            barmode='group', title='Context Analysis',
            labels={'Score': 'Score', 'RAG Method': 'RAG Method'},
            orientation='h', color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
        )
        fig.update_layout(
            title=dict(text='Context Analysis', x=0.5, xanchor='center'),
            xaxis={'title': {'text': 'Score'}},
            yaxis={'title': {'text': ''}, 'tickvals': ticksval, 'ticktext': tickstext},
            legend_traceorder='reversed',
            legend={'title': ''},
            margin={'l': 150}
        )
        
        return fig.to_dict()
    
    def ground_truth_graph(self, ground_truth: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        data = []
        ticksval = []
        
        for rag in ground_truth.keys():
            ticksval.append(rag)
            for metric in ground_truth[rag]:
                data.append({
                    'RAG Method': rag,
                    'Score': ground_truth[rag][metric],
                    'Metric': metric
                })
        
        fig = px.bar(
            data, x='Score', y='RAG Method', color='Metric',
            barmode='group', title='Ground Truth Analysis',
            labels={'Score': 'Score', 'RAG Method': 'RAG Method'},
            orientation='h', color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
        )
        
        tickstext = [self.all_rags.get(tick, tick) for tick in ticksval]
        fig.update_layout(
            title=dict(text='Ground Truth Analysis', x=0.5, xanchor='center'),
            xaxis={'title': {'text': 'Score'}},
            yaxis={'title': {'text': ''}, 'tickvals': ticksval, 'ticktext': tickstext},
            legend={'title': {'text': ''}},
            legend_traceorder='reversed',
            margin={'l': 150}
        )
        
        return fig.to_dict()
    
    def arena_graphs(self, arena_matrix: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
        figures = {}
        
        for match in arena_matrix.keys():
            mid = match.find('_v_')
            rag1 = match[:mid]
            rag2 = match[mid + 3:]
            
            data = []
            for metric in arena_matrix[match].keys():
                scores = arena_matrix[match][metric]
                data.append({
                    'Metric': metric,
                    'RAG': self.all_rags.get(rag1, rag1),
                    'Score': scores[0] if len(scores) > 0 else 0
                })
                data.append({
                    'Metric': metric,
                    'RAG': self.all_rags.get(rag2, rag2),
                    'Score': scores[1] if len(scores) > 1 else 0
                })
            
            fig = px.bar(
                data, x='Score', y='Metric', color='RAG',
                barmode='stack', labels={'Score': '', 'Metric': ''},
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
            )
            fig.update_layout(xaxis=dict(range=[0, 100]), margin={'l': 100})
            figures[match] = fig.to_dict()
        
        return figures
    
    def report_arena_graph(self, arena_graphs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not arena_graphs:
            return {}
        
        n_cols = max(1, int((-1 + sqrt(1 + 8 * len(arena_graphs))) / 2))
        rag_list = []
        matches = list(arena_graphs.keys())
        
        for match in matches[0:n_cols]:
            mid = match.find('_v_')
            rag_b = match[mid + 3:]
            rag_list.append(self.all_rags.get(rag_b, rag_b))
        
        while len(rag_list) < n_cols ** 2:
            rag_list.append('')
        
        fig = make_subplots(rows=n_cols, cols=n_cols, subplot_titles=rag_list)
        
        row = 0
        col = 1
        prev_rag_a = ''
        y_titles = []
        yaxis = {}
        
        for match in matches:
            figure_data = arena_graphs[match]
            mid = match.find('_v_')
            rag_a = match[:mid]
            
            if rag_a != prev_rag_a:
                row += 1
                col = 1
                y_titles.append(self.all_rags.get(rag_a, rag_a))
            
            for trace in figure_data.get('data', []):
                fig.add_trace(trace, row=row, col=col)
            col += 1
            prev_rag_a = rag_a
        
        for i in range(n_cols):
            for j in range(n_cols):
                title_suffix = f'{n_cols * i + j + 1}'
                graph_title = y_titles[i] if j == 0 and i < len(y_titles) else ' '
                yaxis[f'yaxis{title_suffix}'] = {'showticklabels': False, 'title': graph_title}
        
        fig.update_layout(
            height=n_cols * 250, width=n_cols * 350,
            showlegend=False, barmode='stack', **yaxis,
            font={'size': 11, 'color': 'black', 'family': 'Arial, sans-serif'}
        )
        
        return fig.to_dict()
    
    def impact_graph(self, impacts: Dict[str, List]) -> Dict[str, Any]:
        data = []
        for rag in impacts.keys():
            data.append({
                'RAG Method': self.all_rags.get(rag, rag),
                'center': (impacts[rag][0] + impacts[rag][1]) / 2,
                'error': (impacts[rag][0] - impacts[rag][1]) / 2
            })
        
        fig = px.scatter(
            data, x='RAG Method', y='center', error_y='error',
            color='RAG Method', color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
        )
        fig.update_traces(marker=dict(size=16, symbol='square', line=dict(width=1, color='black')))
        fig.update_layout(
            yaxis_title='Greenhouse gas emissions (gCO2eq)',
            title=dict(text='Greenhouse gas emissions estimations', x=0.5, xanchor='center')
        )
        
        return fig.to_dict()
    
    def energy_graph(self, energies: Dict[str, List]) -> Dict[str, Any]:
        data = []
        for rag in energies.keys():
            data.append({
                'RAG Method': self.all_rags.get(rag, rag),
                'center': (energies[rag][0] + energies[rag][1]) / 2,
                'error': (energies[rag][0] - energies[rag][1]) / 2
            })
        
        fig = px.scatter(
            data, x='RAG Method', y='center', error_y='error',
            color='RAG Method', color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
        )
        fig.update_traces(marker=dict(size=16, symbol='square', line=dict(width=1, color='black')))
        fig.update_layout(
            yaxis_title='Power used (kWh)',
            title=dict(text='Energy consumption estimations', x=0.5, xanchor='center')
        )
        
        return fig.to_dict()
    
    def extract_impact(self, df: pd.DataFrame) -> Dict[str, List]:
        list_rags = list(df.columns)[2:]
        impacts = {}
        
        for rag in list_rags:
            impact = [0, 0, 'gCO2eq']
            for query in df[rag]:
                q = self._parse_query(query)
                q_impacts = q.get('IMPACTS', [0, 0])
                impact[0] += q_impacts[0] * 1000 if len(q_impacts) > 0 else 0
                impact[1] += q_impacts[1] * 1000 if len(q_impacts) > 1 else 0
            impacts[rag] = impact
        
        return impacts
    
    def extract_energy(self, df: pd.DataFrame) -> Dict[str, List]:
        list_rags = list(df.columns)[2:]
        energies = {}
        
        for rag in list_rags:
            energy = [0, 0, 'wH']
            for query in df[rag]:
                q = self._parse_query(query)
                q_energy = q.get('ENERGY', [0, 0])
                energy[0] += q_energy[0] * 1000 if len(q_energy) > 0 else 0
                energy[1] += q_energy[1] * 1000 if len(q_energy) > 1 else 0
            energies[rag] = energy
        
        return energies
    
    def _parse_query(self, query: Any) -> Dict[str, Any]:
        if isinstance(query, dict):
            return query
        if isinstance(query, str):
            try:
                return ast.literal_eval(query)
            except:
                return {}
        return {}
