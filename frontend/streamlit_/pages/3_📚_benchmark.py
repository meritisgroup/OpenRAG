import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_.services import BenchmarkService
from streamlit_.api_client import APIClient
from streamlit_.api_client.exceptions import APIError
from streamlit_.core.config import API_BASE_URL

st.markdown('# Benchmark Generation')
st.markdown('## Choose RAG techniques to benchmark:')

all_rags = list(st.session_state['all_rags'].keys())
nb_rags = len(all_rags)
rags_per_column = nb_rags // 3 if nb_rags % 3 == 0 else nb_rags // 3 + 1

col1, col2, col3 = st.columns(3)

with col1:
    for i in range(rags_per_column):
        disable = all_rags[i] == 'main' and st.session_state.hf_token in [None, '']
        st.session_state['benchmark']['rags'][all_rags[i]] = st.checkbox(
            label=st.session_state['all_rags'][all_rags[i]],
            value=st.session_state['benchmark']['rags'].get(all_rags[i], False),
            disabled=disable
        )

with col2:
    for i in range(rags_per_column, 2 * rags_per_column):
        if i < nb_rags:
            disable = all_rags[i] == 'main' and st.session_state.hf_token in [None, '']
            st.session_state['benchmark']['rags'][all_rags[i]] = st.checkbox(
                label=st.session_state['all_rags'][all_rags[i]],
                value=st.session_state['benchmark']['rags'].get(all_rags[i], False),
                disabled=disable
            )

with col3:
    for i in range(2 * rags_per_column, nb_rags):
        disable = all_rags[i] == 'main' and st.session_state.hf_token in [None, '']
        st.session_state['benchmark']['rags'][all_rags[i]] = st.checkbox(
            label=st.session_state['all_rags'][all_rags[i]],
            value=st.session_state['benchmark']['rags'].get(all_rags[i], False),
            disabled=disable
        )

st.markdown('## Import your list of queries')
left, right = st.columns([0.5, 0.15], vertical_alignment='bottom')

_client = APIClient(API_BASE_URL)
try:
    query_files = _client.list_query_files()
    list_queries = [q['filename'] for q in query_files]
except APIError as e:
    st.error(f"Error loading query files: {e}")
    list_queries = []

st.session_state['benchmark']['queries_doc_name'] = left.selectbox(
    label='Select your list of queries',
    options=list_queries,
    index=list_queries.index(st.session_state['benchmark'].get('queries_doc_name', list_queries[0])) if st.session_state['benchmark'].get('queries_doc_name') in list_queries else 0,
    label_visibility='collapsed'
)

if right.button(label='Delete Query Doc', type='primary', use_container_width=True):
    try:
        _client.delete_query_file(st.session_state['benchmark']['queries_doc_name'])
        st.rerun()
    except APIError as e:
        st.error(f"Error deleting query file: {e}")

if st.session_state['benchmark']['queries_doc_name'] is not None:
    try:
        query_data = _client.get_query_file(st.session_state['benchmark']['queries_doc_name'])
        queries_list = query_data.get('queries', [])
        if queries_list:
            st.session_state['benchmark']['queries'] = pd.DataFrame(queries_list)
            st.write(st.session_state['benchmark']['queries'])
    except APIError as e:
        st.error(f"Error loading query file: {e}")


def set_bool():
    st.session_state['benchmark']['load'] = True


uploaded_files = st.file_uploader(
    'Only `Excel` files are supported',
    type=['xls', 'xlsx', 'xlsm', 'odt', 'xlsb'],
    accept_multiple_files=False,
    on_change=set_bool
)

if st.session_state['benchmark'].get('load') and uploaded_files:
    try:
        _client.upload_query_file(uploaded_files.name, uploaded_files.getvalue())
        st.session_state['benchmark']['load'] = False
        st.rerun()
    except APIError as e:
        st.error(f"Error uploading query file: {e}")

st.markdown('## Choose database to perform benchmark on')
if 'benchmark_database' not in st.session_state or st.session_state['benchmark_database'] is None:
    st.session_state['benchmark_database'] = []

if len(st.session_state.get('all_databases', [])) == 0:
    st.warning('No database available, please create one in the databases page')
else:
    st.multiselect(label='Choose database', options=st.session_state['all_databases'], key='benchmark_database')

st.markdown('## Generate queries')
left, right = st.columns([0.5, 0.15], vertical_alignment='bottom')
st.session_state['benchmark']['number_of_questions'] = left.number_input(label='Number of queries', min_value=1, step=1, value=5)

if right.button(label='Generate queries', type='primary', use_container_width=True):
    with st.spinner('Generating queries...'):
        try:
            result = BenchmarkService.generate_queries(
                databases=st.session_state['benchmark_database'],
                n_questions=st.session_state['benchmark']['number_of_questions'],
                config=st.session_state['config_server'],
                models_infos=st.session_state['models_infos']
            )
            st.success(f"Generated {len(result['queries'])} queries to: {result['file_path']}")
            st.rerun()
        except APIError as e:
            st.error(f"Error generating queries: {e}")

if 'all_system_prompt' not in st.session_state:
    st.session_state['all_system_prompt'] = st.session_state['config_server']['all_system_prompt']
if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = st.session_state['config_server']['local_params']['generation_system_prompt_name']
if 'system_prompt_selected' not in st.session_state:
    st.session_state['system_prompt_selected'] = 'default'


def force_system_prompt():
    system_prompt_selected = st.session_state['system_prompt_selected']
    if system_prompt_selected != 'default':
        st.session_state['config_server']['local_params']['forced_system_prompt'] = True
        st.session_state['config_server']['local_params']['generation_system_prompt_name'] = system_prompt_selected
    else:
        st.session_state['config_server']['local_params']['forced_system_prompt'] = False
        st.session_state['config_server']['local_params']['generation_system_prompt_name'] = 'default'


st.markdown('## Choose the prompt to perform benchmark on')
system_prompt_selected = st.selectbox(
    label='**Choose system prompt**',
    options=st.session_state['all_system_prompt'],
    key='system_prompt_selected',
    on_change=force_system_prompt
)

reset_index = st.checkbox(label='Reset indexing', value=False)
reset_preprocess = st.checkbox(label='Reset preprocessing', value=False)

if 'benchmark_clicked' not in st.session_state:
    st.session_state['benchmark_clicked'] = False
    st.session_state['plot_to_display'] = False


def handle_click():
    st.session_state['benchmark_clicked'] = True


st.markdown('## Benchmark already done')


def get_saved_benchmarks():
    try:
        reports = BenchmarkService.list_reports()
        folders = [r['report_id'] for r in reports 
                   if r.get('type') in ('all', 'ground_truth', 'full_bench')]
        return ['None'] + sorted(folders, reverse=True)
    except APIError as e:
        st.error(f"Error loading benchmark reports: {e}")
        return ['None']


def show_already_done_benchmark():
    selected = st.session_state.get('benchmark_done')
    if not selected or selected == 'None':
        if 'benchmark_result' in st.session_state:
            del st.session_state['benchmark_result']
        return
    
    try:
        result = BenchmarkService.get_result(str(selected))
        st.session_state['benchmark_result'] = result
        st.session_state['benchmark_database'] = result.get('databases', [])
    except APIError as e:
        st.error(f"Error loading benchmark: {e}")


benchmark_already_done = get_saved_benchmarks()
selected_benchmark = st.selectbox(
    label='**Choose benchmark done**',
    options=benchmark_already_done,
    key='benchmark_done',
    on_change=show_already_done_benchmark
)

col1, col2, col3, col4, col5 = st.columns([0.2, 0.2, 0.2, 0.25, 0.15])

if col1.button('Generate Contexts', on_click=handle_click, disabled=st.session_state['benchmark_clicked'], use_container_width=True, type='primary'):
    if not st.session_state.get('benchmark_database'):
        st.session_state['benchmark_clicked'] = False
        st.error('Choose a database')
    else:
        rag_to_run = [rag for rag in st.session_state['benchmark']['rags'].keys() if st.session_state['benchmark']['rags'][rag]]
        if len(rag_to_run) < 1:
            st.error('Choose at least 1 RAG method')
            st.session_state['benchmark_clicked'] = False
        else:
            with st.spinner('Running benchmark...'):
                try:
                    result = BenchmarkService.run_benchmark_sync(
                        rag_names=rag_to_run,
                        databases=st.session_state['benchmark_database'],
                        queries_doc_name=st.session_state['benchmark']['queries_doc_name'],
                        config=st.session_state['config_server'],
                        models_infos=st.session_state['models_infos'],
                        benchmark_type='contexts',
                        reset_index=reset_index,
                        reset_preprocess=reset_preprocess
                    )
                    st.session_state['benchmark_result'] = result
                    st.session_state['benchmark_clicked'] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state['benchmark_clicked'] = False

if col2.button('Generate Answers', on_click=handle_click, disabled=st.session_state['benchmark_clicked'], use_container_width=True, type='primary'):
    if not st.session_state.get('benchmark_database'):
        st.session_state['benchmark_clicked'] = False
        st.error('Choose a database')
    else:
        rag_to_run = [rag for rag in st.session_state['benchmark']['rags'].keys() if st.session_state['benchmark']['rags'][rag]]
        if len(rag_to_run) < 1:
            st.error('Choose at least 1 RAG method')
            st.session_state['benchmark_clicked'] = False
        else:
            with st.spinner('Running benchmark...'):
                try:
                    result = BenchmarkService.run_benchmark_sync(
                        rag_names=rag_to_run,
                        databases=st.session_state['benchmark_database'],
                        queries_doc_name=st.session_state['benchmark']['queries_doc_name'],
                        config=st.session_state['config_server'],
                        models_infos=st.session_state['models_infos'],
                        benchmark_type='answers',
                        reset_index=reset_index,
                        reset_preprocess=reset_preprocess
                    )
                    st.session_state['benchmark_result'] = result
                    st.session_state['benchmark_clicked'] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state['benchmark_clicked'] = False

if col3.button('Generate ground truth', on_click=handle_click, disabled=st.session_state['benchmark_clicked'], use_container_width=True, type='primary'):
    if not st.session_state.get('benchmark_database'):
        st.session_state['benchmark_clicked'] = False
        st.error('Choose a database')
    else:
        rag_to_run = [rag for rag in st.session_state['benchmark']['rags'].keys() if st.session_state['benchmark']['rags'][rag]]
        if len(rag_to_run) < 2:
            st.error('Choose at least 2 RAG methods for ground truth comparison')
            st.session_state['benchmark_clicked'] = False
        else:
            progress_bar = st.progress(0, text="Starting benchmark...")
            status_text = st.empty()
            
            def update_progress(status):
                progress = status.get('progress', 0)
                step = status.get('current_step', '')
                progress_bar.progress(int(progress), text=f"{step} ({progress:.0f}%)")
                status_text.text(f"Current step: {step}")
            
            try:
                result = BenchmarkService.run_benchmark_sync(
                    rag_names=rag_to_run,
                    databases=st.session_state['benchmark_database'],
                    queries_doc_name=st.session_state['benchmark']['queries_doc_name'],
                    config=st.session_state['config_server'],
                    models_infos=st.session_state['models_infos'],
                    benchmark_type='ground_truth',
                    reset_index=reset_index,
                    reset_preprocess=reset_preprocess,
                    progress_callback=update_progress
                )
                st.session_state['benchmark_result'] = result
                st.session_state['benchmark_clicked'] = False
                progress_bar.progress(100, text="Completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state['benchmark_clicked'] = False

if col4.button('Generate Full Benchmark', on_click=handle_click, disabled=st.session_state['benchmark_clicked'], use_container_width=True, type='primary'):
    if not st.session_state.get('benchmark_database'):
        st.session_state['benchmark_clicked'] = False
        st.error('Choose a database')
    else:
        rag_to_run = [rag for rag in st.session_state['benchmark']['rags'].keys() if st.session_state['benchmark']['rags'][rag]]
        if len(rag_to_run) < 2:
            st.error('Choose at least 2 RAG methods for full benchmark')
            st.session_state['benchmark_clicked'] = False
        else:
            progress_bar = st.progress(0, text="Starting benchmark...")
            status_text = st.empty()
            
            def update_progress(status):
                progress = status.get('progress', 0)
                step = status.get('current_step', '')
                progress_bar.progress(int(progress), text=f"{step} ({progress:.0f}%)")
                status_text.text(f"Current step: {step}")
            
            try:
                result = BenchmarkService.run_benchmark_sync(
                    rag_names=rag_to_run,
                    databases=st.session_state['benchmark_database'],
                    queries_doc_name=st.session_state['benchmark']['queries_doc_name'],
                    config=st.session_state['config_server'],
                    models_infos=st.session_state['models_infos'],
                    benchmark_type='full_bench',
                    reset_index=reset_index,
                    reset_preprocess=reset_preprocess,
                    progress_callback=update_progress
                )
                st.session_state['benchmark_result'] = result
                st.session_state['benchmark_clicked'] = False
                progress_bar.progress(100, text="Completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state['benchmark_clicked'] = False


def display_plots(plots: dict):
    if not plots:
        return
    
    if 'ground_truth_graph' in plots:
        fig = go.Figure(plots['ground_truth_graph'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'context_graph' in plots:
        fig = go.Figure(plots['context_graph'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'token_graph' in plots:
        fig = go.Figure(plots['token_graph'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'time_graph' in plots:
        fig = go.Figure(plots['time_graph'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'impact_graph' in plots:
        fig = go.Figure(plots['impact_graph'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'energy_graph' in plots:
        fig = go.Figure(plots['energy_graph'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'arena_graphs' in plots:
        arena_graphs = plots['arena_graphs']
        if arena_graphs:
            matches = list(arena_graphs.keys())
            selected_match = st.selectbox(
                label='**Choose arena match to analyse**',
                options=matches,
                format_func=lambda m: m.replace('_v_', ' vs ')
            )
            if selected_match in arena_graphs:
                fig = go.Figure(arena_graphs[selected_match])
                st.plotly_chart(fig, use_container_width=True)


if st.session_state.get('benchmark_result'):
    result = st.session_state['benchmark_result']
    
    databases = st.session_state.get('benchmark_database', [])
    markdown_text = '**Benchmark runned on the following database:**\n'
    for db in databases:
        markdown_text += f'- {db}\n'
    st.markdown(markdown_text)
    st.success('***Benchmark terminé !***')
    
    plots = result.get('plots', {})
    if plots:
        display_plots(plots)
    
    files = result.get('files', {})
    benchmark_id = result.get('benchmark_id', '')
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    if benchmark_id:
        try:
            if 'pdf' in files:
                pdf_data = _client.download_benchmark_file(benchmark_id, 'pdf')
                col_dl1.download_button(
                    label='Download report',
                    data=pdf_data,
                    file_name='report.pdf',
                    type='primary',
                    use_container_width=True
                )
            
            if 'excel' in files:
                excel_data = _client.download_benchmark_file(benchmark_id, 'excel')
                col_dl2.download_button(
                    label='Download answers',
                    data=excel_data,
                    file_name='benchmark_answers.xlsx',
                    type='primary',
                    use_container_width=True
                )
            
            if 'csv' in files:
                csv_data = _client.download_benchmark_file(benchmark_id, 'csv')
                col_dl3.download_button(
                    label='Download CSV',
                    data=csv_data,
                    file_name='bench_df.csv',
                    type='primary',
                    use_container_width=True
                )
        except APIError as e:
            st.error(f"Error downloading files: {e}")
