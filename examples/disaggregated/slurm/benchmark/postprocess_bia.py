import glob
import os
import pandas as pd
import re
import argparse
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d
import numpy as np
import yaml

# mtp_accept_rate_1k = {1: 1.86, 2: 2.42, 3: 2.68}  # aa 1k/1k
# mtp_accept_rate_8k = {1: 1.9, 2: 2.5, 3: 2.8}  # aa from dlsim 8k/1k
mtp_accept_rate_1k = {1: 1.8, 2: 2.28, 3: 2.56}  # random 1k/1k
mtp_accept_rate_8k = {1: 1.84, 2: 2.38, 3: 2.76}  # random 8k/1k


def parse_median_tpot_from_bench_log(bench_log_path):
    """
    Parse the median TPOT value from 6_bench.log file.
    
    Args:
        bench_log_path: Path to 6_bench.log file
    
    Returns:
        float: Median TPOT in ms, or 0.0 if not found or invalid
    """
    if not os.path.exists(bench_log_path):
        return 0.0
    
    try:
        with open(bench_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for "Median TPOT (ms):" pattern
        # Example: "0: Median TPOT (ms):                        13.96"
        match = re.search(r'Median TPOT \(ms\):\s+([\d.]+)', content)
        if match:
            tpot_value = float(match.group(1))
            return tpot_value
        return 0.0
    except Exception as e:
        print(f"Error parsing {bench_log_path}: {e}")
        return 0.0


def is_job_completed(file_path, dir_prefix=None):
    """
    Check if a benchmark job completed successfully.
    
    A job is considered complete if:
    1. The job directory contains an 8_done_<slurmid>.txt file
    2. The 6_bench.log file has a valid (>0.0) median TPOT
    
    Args:
        file_path: Path to the log file (e.g., gen_server0.out or 3_output_GEN_0.log)
        dir_prefix: Path to the job directories folder (unused, kept for compatibility)
    
    Returns:
        bool: True if job completed successfully, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Get the job directory
    job_dir = os.path.dirname(file_path)
    
    # Check for 8_done_*.txt file
    done_files = glob.glob(os.path.join(job_dir, '8_done_*.txt'))
    if not done_files:
        print(f"Job not completed (no 8_done_*.txt): {job_dir}")
        return False
    
    # Check for valid median TPOT in 6_bench.log
    bench_log_path = os.path.join(job_dir, '6_bench.log')
    median_tpot = parse_median_tpot_from_bench_log(bench_log_path)
    
    if median_tpot <= 0.0:
        print(f"Job not completed (invalid TPOT={median_tpot}): {job_dir}")
        return False
    
    return True


def read_gen_config(file_path, check_completion=True, dir_prefix=None):
    """
    Read configuration from gen_config.yaml and folder name in the directory containing the given file.
    Returns a dict with normalized parameters or None if job incomplete.
    Raises an error if gen_config.yaml is not available.
    
    Maps:
    - enable_attention_dp: True -> "dep", False -> "tep"
    - tensor_parallel_size: TP size (num_gen_gpus)
    - moe_expert_parallel_size: EP size
    - concurrency: from folder name (batch size)
    
    Args:
        file_path: Path to 3_output_GEN_0.log file
        check_completion: If True, verify the job completed before processing (default: True)
        dir_prefix: Path to job directories folder (unused, kept for compatibility)
    """
    # Check if job completed successfully before processing
    if check_completion and not is_job_completed(file_path, dir_prefix=dir_prefix):
        return None
    
    # Get the directory containing the file
    file_dir = os.path.dirname(file_path)
    
    # Read from gen_config.yaml (required)
    gen_config_path = os.path.join(file_dir, 'gen_config.yaml')
    if not os.path.exists(gen_config_path):
        raise FileNotFoundError(f"gen_config.yaml not found at {gen_config_path}")
    
    with open(gen_config_path, 'r') as f:
        gen_config = yaml.safe_load(f)
    
    # Parse enable_attention_dp
    gen_enable_attention_dp = gen_config.get('enable_attention_dp', False)
    dp_type = 'dep' if gen_enable_attention_dp else 'tep'
    
    # Get parallelism sizes
    tp_size = int(gen_config.get('tensor_parallel_size', 1))
    cp_size = int(gen_config.get('context_parallel_size', 1))
    pp_size = int(gen_config.get('pipeline_parallel_size', 1))
    ep_size = int(gen_config.get('moe_expert_parallel_size', 1))
    
    # Validate: ep_size must equal cp_size * tp_size
    assert ep_size == cp_size * tp_size, \
        f"ep_size ({ep_size}) != cp_size ({cp_size}) * tp_size ({tp_size}) in {file_dir}"
    
    # Validate: pp_size must be 1
    assert pp_size == 1, \
        f"pp_size ({pp_size}) != 1 in {file_dir}"
    
    # Total number of GPUs = tp_size * cp_size * pp_size
    total_gpus = tp_size * cp_size * pp_size
    
    # num_gen_gpus is the total GPU count (used for per-GPU calculations)
    num_gen_gpus = total_gpus
    
    # Parse concurrency from folder name
    # Pattern: YYYYMMDD_HHMMSS_ISL-OSL_ctx#_gen#_tp#_pp#_ep#_cp#_batch#[_attnDP]
    folder_name = os.path.basename(file_dir)
    batch_match = re.search(r'batch(\d+)', folder_name)
    if not batch_match:
        raise ValueError(f"Could not parse batch size from folder name: {folder_name}")
    concurrency = int(batch_match.group(1))
    
    # MTP size (not typically in folder name for these jobs, default to 0)
    mtp_num = 0
    
    return {
        'dp_type': dp_type,
        'num_gen_gpus': num_gen_gpus,
        'concurrency': concurrency,
        'eplb': 0,
        'mtp_num': mtp_num,
        'job_id': 0,
        'gen_tp_size': tp_size,
        'gen_cp_size': cp_size,
        'gen_pp_size': pp_size,
        'gen_moe_ep_size': ep_size,
        'total_gpus': total_gpus,
        'gen_enable_attention_dp': gen_enable_attention_dp
    }


def create_visualization(df, output_path, enable_mtp, title_prefix="Combined"):
    """
    Create Plotly visualization for the processed data with two separate charts
    """
    # Extract num_gen_gpus from name column
    df['num_gen_gpus'] = df['name'].str.extract(r'ep(\d+)_').astype(int)

    # Group by num_gen_gpus and concurrency, keep the row with maximum throughput_per_user
    df_grouped = df.loc[df.groupby(['num_gen_gpus', 'concurrency'])[
        'throughput_per_user'].idxmax()].copy()

    # Sort by num_gen_gpus and throughput_per_user for better visualization
    df_grouped = df_grouped.sort_values(['num_gen_gpus', 'throughput_per_user'])

    # Create figure
    fig = go.Figure()

    # Get unique rank numbers for different colors
    unique_ranks = sorted(df_grouped['num_gen_gpus'].unique())
    colors = px.colors.qualitative.Set1[:len(unique_ranks)]

    # --- Data Preparation ---
    # Prepare data for all charts to be used in the update buttons
    first_chart_data = {'y': [], 'customdata': [], 'hovertemplate': []}
    second_chart_data = {'y': [], 'customdata': [], 'hovertemplate': []}
    third_chart_data = {'y': [], 'customdata': [], 'hovertemplate': []}

    for num_gen_gpus in unique_ranks:
        rank_data = df_grouped[df_grouped['num_gen_gpus'] == num_gen_gpus]

        # Common customdata for all charts
        customdata = list(zip(
            rank_data['name'], rank_data['concurrency'], rank_data['total_tput_per_gpu'],
            rank_data['output_throughput'], rank_data['elapsed_time_avg'],
            rank_data['ctx/gen inst ratio'], rank_data['gen/ctx inst ratio'], rank_data['ctx/gen gpu ratio']
        ))

        # Data for the first chart
        first_chart_data['y'].append(rank_data['output_tput_per_gpu'])
        first_chart_data['customdata'].append(customdata)
        first_chart_data['hovertemplate'].append(
            '<b>Name: %{customdata[0]}</b><br>' +
            'Concurrency: %{customdata[1]}<br>' +
            'Throughput per User: %{x:.4f}<br>' +
            'Output Throughput per GPU: %{y:.4f}<br>' +
            'Total Throughput per GPU: %{customdata[2]:.4f}<br>' +
            'Output Throughput: %{customdata[3]:.4f}<br>' +
            'Elapsed Time Avg: %{customdata[4]:.4f}s<br>' +
            'Ctx/Gen Inst Ratio: %{customdata[5]:.4f}<br>' +
            'Gen/Ctx Inst Ratio: %{customdata[6]:.4f}<br>' +
            'Ctx/Gen GPU Ratio: %{customdata[7]:.4f}<br>' +
            '<extra></extra>'
        )

        # Data for the second chart
        second_chart_data['y'].append(
            rank_data['output_throughput_per_gen_gpu'])
        second_chart_data['customdata'].append(customdata)
        second_chart_data['hovertemplate'].append(
            '<b>Name: %{customdata[0]}</b><br>' +
            'Concurrency: %{customdata[1]}<br>' +
            'Throughput per User: %{x:.4f}<br>' +
            'Output Throughput per Gen GPU: %{y:.4f}<br>' +
            'Total Throughput per GPU: %{customdata[2]:.4f}<br>' +
            'Output Throughput: %{customdata[3]:.4f}<br>' +
            'Elapsed Time Avg: %{customdata[4]:.4f}s<br>' +
            'Ctx/Gen Inst Ratio: %{customdata[5]:.4f}<br>' +
            'Gen/Ctx Inst Ratio: %{customdata[6]:.4f}<br>' +
            'Ctx/Gen GPU Ratio: %{customdata[7]:.4f}<br>' +
            '<extra></extra>'
        )

        # Data for the third chart
        third_chart_data['y'].append(rank_data['total_tput_per_gpu'])
        third_chart_data['customdata'].append(customdata)
        third_chart_data['hovertemplate'].append(
            '<b>Name: %{customdata[0]}</b><br>' +
            'Concurrency: %{customdata[1]}<br>' +
            'Throughput per User: %{x:.4f}<br>' +
            'Total Throughput per Total GPU: %{y:.4f}<br>' +
            'Total Throughput per GPU: %{customdata[2]:.4f}<br>' +
            'Output Throughput: %{customdata[3]:.4f}<br>' +
            'Elapsed Time Avg: %{customdata[4]:.4f}s<br>' +
            'Ctx/Gen Ratio: %{customdata[5]:.4f}<br>' +
            'Gen/Ctx Inst Ratio: %{customdata[6]:.4f}<br>' +
            'Ctx/Gen GPU Ratio: %{customdata[7]:.4f}<br>' +
            '<extra></extra>'
        )

    # Create initial traces for the first chart
    for i, num_gen_gpus in enumerate(unique_ranks):
        rank_data = df_grouped[df_grouped['num_gen_gpus'] == num_gen_gpus]
        fig.add_trace(go.Scatter(
            x=rank_data['throughput_per_user'],
            y=first_chart_data['y'][i],
            customdata=first_chart_data['customdata'][i],
            hovertemplate=first_chart_data['hovertemplate'][i],
            mode='markers+lines',
            name=f'Rank {num_gen_gpus}',
            line=dict(color=colors[i], width=2, shape='spline'),
            marker=dict(size=8, color=colors[i],
                        line=dict(width=1, color='white'))
        ))

    # --- Helper function to calculate annotations and shapes ---
    def get_max_ratio_layout(df_grouped, y_column_name):
        shapes = []
        annotations = []
        unique_ranks_calc = sorted(df_grouped['num_gen_gpus'].unique())

        if len(unique_ranks_calc) <= 1:
            return shapes, annotations

        # 定义目标rank组合：上面的点选择16/32，下面的点选择4/8
        target_upper_ranks = [16, 32]
        target_lower_ranks = [4, 8]

        # 检查是否有目标rank存在
        available_upper_ranks = [
            r for r in target_upper_ranks if r in unique_ranks_calc]
        available_lower_ranks = [
            r for r in target_lower_ranks if r in unique_ranks_calc]

        if not available_upper_ranks or not available_lower_ranks:
            return shapes, annotations

        interpolation_functions = {}
        x_ranges = {}
        for num_gen_gpus in unique_ranks_calc:
            rank_data = df_grouped[df_grouped['num_gen_gpus'] == num_gen_gpus]
            if len(rank_data) >= 2:
                x_values = rank_data['throughput_per_user'].values
                y_values = rank_data[y_column_name].values
                sorted_indices = np.argsort(x_values)
                x_sorted, y_sorted = x_values[sorted_indices], y_values[sorted_indices]
                try:
                    interp_func = interp1d(
                        x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
                    interpolation_functions[num_gen_gpus] = interp_func
                    x_ranges[num_gen_gpus] = (x_sorted.min(), x_sorted.max())
                except Exception as e:
                    print(
                        f"Debug: Failed to create interpolation for rank {num_gen_gpus}: {e}")

        if not x_ranges:
            return shapes, annotations

        # 只保留目标rank的插值函数
        target_ranks = available_upper_ranks + available_lower_ranks
        interpolation_functions = {
            k: v for k, v in interpolation_functions.items() if k in target_ranks}

        if len(interpolation_functions) < 2:
            return shapes, annotations

        # 寻找全局最佳的ratio组合
        global_max_ratio = 0
        best_params = None

        for upper_rank_target in available_upper_ranks:
            # 对每个upper_rank，遍历所有可能的lower_rank组合
            for lower_rank_target in available_lower_ranks:
                if upper_rank_target not in interpolation_functions or lower_rank_target not in interpolation_functions:
                    continue

                # 计算当前rank组合的x范围交集
                upper_x_range = x_ranges[upper_rank_target]
                lower_x_range = x_ranges[lower_rank_target]

                # 计算交集
                x_min = max(upper_x_range[0], lower_x_range[0])
                x_max = min(upper_x_range[1], lower_x_range[1])

                print(
                    f"Debug: Rank{upper_rank_target}/Rank{lower_rank_target} x range: [{x_min:.4f}, {x_max:.4f}]")

                if x_min >= x_max:
                    print(
                        f"Debug: No overlap for Rank{upper_rank_target}/Rank{lower_rank_target}")
                    continue

                x_samples = np.linspace(x_min, x_max, 100)

                for x_val in x_samples:
                    # 获取upper_rank的y值
                    upper_y_val = interpolation_functions[upper_rank_target](
                        x_val)
                    if np.isnan(upper_y_val) or upper_y_val <= 0:
                        continue

                    # 获取lower_rank的y值
                    lower_y_val = interpolation_functions[lower_rank_target](
                        x_val)
                    if np.isnan(lower_y_val) or lower_y_val <= 0:
                        continue
                    # 计算ratio
                    ratio = upper_y_val / lower_y_val
                    if ratio > global_max_ratio:
                        global_max_ratio = ratio
                        best_params = {
                            'x': x_val, 'y_top': upper_y_val, 'y_bottom': lower_y_val,
                            'rank_top': upper_rank_target, 'rank_bottom': lower_rank_target,
                            'max_ratio': ratio
                        }

        # 只添加全局最佳的辅助线
        if best_params:
            p = best_params
            print(
                f"Global max ratio: {p['max_ratio']:.2f}x (Rank {p['rank_top']} vs Rank {p['rank_bottom']}) with y-values {p['y_top']:.4f} / {p['y_bottom']:.4f}")

            color = 'red'
            shapes = [
                {'type': 'line', 'x0': p['x'], 'x1': p['x'], 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {
                    'color': color, 'width': 2, 'dash': 'dash'}},
                {'type': 'line', 'x0': 0, 'x1': 1, 'y0': p['y_top'], 'y1': p['y_top'], 'xref': 'paper', 'yref': 'y', 'line': {
                    'color': color, 'width': 1, 'dash': 'dot'}},
                {'type': 'line', 'x0': 0, 'x1': 1, 'y0': p['y_bottom'], 'y1': p['y_bottom'], 'xref': 'paper', 'yref': 'y', 'line': {
                    'color': color, 'width': 1, 'dash': 'dot'}}
            ]
            annotations = [{
                'x': p['x'], 'y': (p['y_top'] + p['y_bottom']) / 2,
                'text': f"Speedup: {p['max_ratio']:.2f}x<br>Rank {p['rank_top']}/Rank {p['rank_bottom']}<br>Fitted values: {p['y_top']:.4f}/{p['y_bottom']:.4f}<br>x = {p['x']:.4f}",
                'showarrow': True, 'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 2, 'arrowcolor': color,
                'ax': 40, 'ay': -80, 'bgcolor': 'rgba(255,255,255,0.8)', 'bordercolor': color, 'borderwidth': 1
            }]
        return shapes, annotations

    # --- Calculate layouts for all charts ---
    first_chart_shapes, first_chart_annotations = get_max_ratio_layout(
        df_grouped, 'output_tput_per_gpu')
    second_chart_shapes, second_chart_annotations = get_max_ratio_layout(
        df_grouped, 'output_throughput_per_gen_gpu')
    third_chart_shapes, third_chart_annotations = get_max_ratio_layout(
        df_grouped, 'total_tput_per_gpu')

    # --- Define full y-axis objects for robust updates ---
    base_yaxis_config = {
        'showgrid': True,
        'gridwidth': 1,
        'gridcolor': 'lightgray',
        'autorange': True
    }

    first_yaxis = {**base_yaxis_config,
                   'title': {'text': "Output Throughput per Total GPU"}}
    second_yaxis = {**base_yaxis_config,
                    'title': {'text': "Output Throughput per Gen GPU"}}
    third_yaxis = {**base_yaxis_config,
                   'title': {'text': "Total Throughput per Total GPU"}}

    # --- Create buttons ---
    buttons = [
        dict(
            label="out_tput_per_total_gpu",
            method="update",
            args=[
                {
                    "y": first_chart_data['y'],
                    "customdata": first_chart_data['customdata'],
                    "hovertemplate": first_chart_data['hovertemplate']
                },
                {
                    "title": {'text': f'Throughput Analysis - {title_prefix} (Output Throughput per Total GPU)'},
                    "yaxis": first_yaxis,
                    "shapes": first_chart_shapes,
                    "annotations": first_chart_annotations
                }
            ]
        ),
        dict(
            label="out_tput_per_gen_gpu",
            method="update",
            args=[
                {
                    "y": second_chart_data['y'],
                    "customdata": second_chart_data['customdata'],
                    "hovertemplate": second_chart_data['hovertemplate']
                },
                {
                    "title": {'text': f'Throughput Analysis - {title_prefix} (Output Throughput per Gen GPU)'},
                    "yaxis": second_yaxis,
                    "shapes": second_chart_shapes,
                    "annotations": second_chart_annotations
                }
            ]
        ),
        dict(
            label="total_tput_per_total_gpu",
            method="update",
            args=[
                {
                    "y": third_chart_data['y'],
                    "customdata": third_chart_data['customdata'],
                    "hovertemplate": third_chart_data['hovertemplate']
                },
                {
                    "title": {'text': f'Throughput Analysis - {title_prefix} (Total Throughput per Total GPU)'},
                    "yaxis": third_yaxis,
                    "shapes": third_chart_shapes,
                    "annotations": third_chart_annotations
                }
            ]
        )
    ]

    # --- Final Layout ---
    fig.update_layout(
        title={
            'text': f'Throughput Analysis - {title_prefix} (Output Throughput per Total GPU)'},
        xaxis_title='Throughput per User',
        yaxis=first_yaxis,
        hovermode='closest',
        template='plotly_white',
        legend=dict(title="EP rank number", yanchor="top",
                    y=0.99, xanchor="left", x=0.99),
        width=1200,
        height=800,
        shapes=first_chart_shapes,
        annotations=first_chart_annotations,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.0,
            'yanchor': 'top',
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'black',
            'borderwidth': 1
        }]
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Save as HTML file
    html_file = os.path.join(output_path, "summary_throughput_analysis_mtp0.html" if not enable_mtp else "summary_throughput_analysis_mtp.html")
    fig.write_html(html_file)
    print(f"Visualization saved to {html_file}")

    return df_grouped


def process_files_multi(dir_prefixes, output_path, ctx_request_rate, ctx_gpus, isl, osl, enable_mtp=False):
    """
    Process benchmark files from multiple directories and combine results.
    
    Args:
        dir_prefixes: List of directory prefixes to search for benchmark files
        output_path: Output directory path for all generated artifacts
        ctx_request_rate: Context request rate
        ctx_gpus: Number of context GPUs
        isl: Input sequence length
        osl: Output sequence length
        enable_mtp: If True, process only data with mtp_num > 0
    
    Returns:
        Combined DataFrame with all summary data, or None if no valid data found
    """
    mtp_accept_rate = mtp_accept_rate_1k if isl == 1024 else mtp_accept_rate_8k
    print(f"Using mtp_accept_rate: {mtp_accept_rate}")
    
    all_summary_data = []
    incomplete_jobs = []  # Track incomplete jobs
    
    for dir_prefix in dir_prefixes:
        print(f"\n{'='*60}")
        print(f"Processing directory: {dir_prefix}")
        print(f"{'='*60}")
        
        pattern = f"{dir_prefix}*/3_output_GEN_0.log"
        files = glob.glob(pattern)
        print(f"Found {len(files)} files matching pattern {pattern}")
        
        for file in files:
            data = []
            
            # First check if job completed successfully
            if not is_job_completed(file, dir_prefix=dir_prefix):
                job_dir = os.path.dirname(file)
                incomplete_jobs.append(job_dir)
                continue
            
            # Read parameters from gen_config.yaml + folder name
            try:
                config = read_gen_config(file, check_completion=False, dir_prefix=dir_prefix)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {file}: {e}")
                continue
            
            if config is None:
                continue

            # Extract parameters from config
            dp_tep = config['dp_type']
            num_gen_gpus = config['num_gen_gpus']
            concurrency = config['concurrency']
            eplb_num = config['eplb']
            mtp_num = config['mtp_num']
            job_id = config['job_id']
            tp_size = config['gen_tp_size']
            
            # Determine dp_size based on dp_type
            # dp_size = tp_size when attnDP is enabled (DEP mode), else 1 for TEP
            if dp_tep == 'tep':
                dp_size = 1
            else:  # dep
                dp_size = tp_size
            name = f"tp{config['gen_tp_size']}_cp{config['gen_cp_size']}_ep{config['gen_moe_ep_size']}_attnDP{int(config['gen_enable_attention_dp'])}_mtp{mtp_num}"

            if mtp_num > 0 and not enable_mtp:
                continue
            elif mtp_num == 0 and enable_mtp:
                continue

            # Find and read job_*.txt file to get host information
            hosts = ""
            try:
                file_dir = file.rsplit('/', 2)[0]
                job_files = glob.glob(f"{file_dir}/job_*.txt")
                if job_files:
                    with open(job_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        hosts = f.readline().strip()
                    print(f"Found host information: {hosts} from {job_files[0]}")
                else:
                    print(f"No job_*.txt files found in {file_dir}")
            except Exception as e:
                print(f"Error reading job file for {file}: {e}")

            # Read and parse log file
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                log_pattern = r'iter = (\d+), global_rank = (\d+), rank = (\d+), currank_total_requests = (\d+)/(\d+), host_step_time = ([\d.]+)ms, prev_device_step_time = ([\d.]+)ms, timestamp = ([^,]+), num_scheduled_requests: (\d+), states = \{\'num_ctx_requests\': (\d+), \'num_ctx_tokens\': (\d+), \'num_generation_tokens\': (\d+)\}'

                matches = re.findall(log_pattern, content)

                if matches:
                    for match in matches:
                        iter_num = int(match[0])
                        global_rank = int(match[1])
                        rank = int(match[2])
                        current_requests = int(match[3])
                        total_requests = int(match[4])
                        host_step_time = float(match[5]) / 1000
                        prev_device_step_time = float(match[6]) / 1000
                        timestamp = match[7]
                        num_scheduled_requests = int(match[8])
                        num_ctx_requests = int(match[9])
                        num_ctx_tokens = int(match[10])
                        num_generation_tokens = int(match[11])
                        elapsed_time = prev_device_step_time

                        throughput_per_user = num_generation_tokens / \
                            elapsed_time if elapsed_time > 0 else 0

                        data.append({
                            'concurrency': concurrency,
                            'iter': iter_num,
                            'global_rank': global_rank,
                            'rank': rank,
                            'current_requests': current_requests,
                            'total_requests': total_requests,
                            'elapsed_time': elapsed_time,
                            'timestamp': timestamp,
                            'num_scheduled_requests': num_scheduled_requests,
                            'num_ctx_requests': num_ctx_requests,
                            'num_ctx_tokens': num_ctx_tokens,
                            'num_generation_tokens': num_generation_tokens,
                            'throughput_per_user': throughput_per_user,
                            'host_step_time': host_step_time,
                            'prev_device_step_time': prev_device_step_time,
                            'hosts': hosts
                        })
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
            
            if data:
                df = pd.DataFrame(data)
                df = df.sort_values(['concurrency', 'iter'])
                output_file = file.split('.')[0] + '.csv'
                tmp_file = file.split('.')[0] + '.tmp.csv'

                df = df[df['num_ctx_tokens'] == 0]
                df = df.groupby(['iter', 'global_rank']).last().reset_index()
                df.to_csv(tmp_file, index=False)

                df = df.iloc[50:-10]

                iters_before_filter = len(df)
                if dp_tep == 'tep':
                    df = df[df['num_scheduled_requests'] == int(concurrency)]
                    df = df[df['num_generation_tokens'] ==
                            int(concurrency * (mtp_num + 1))]
                elif dp_tep == 'dep':
                    df = df[df['num_scheduled_requests']
                            == int(concurrency / dp_size)]
                    df = df[df['num_generation_tokens'] == int(
                        concurrency / dp_size * (mtp_num + 1))]
                iters_after_filter = len(df)
                print(f"FILTERING ITERATIONS: {iters_before_filter} -> {iters_after_filter} (removed {iters_before_filter - iters_after_filter})")

                # Filter outliers using median ±20%
                if not df.empty:
                    elapsed_time_median = df['elapsed_time'].median()
                    lower_bound = elapsed_time_median * 0.8
                    upper_bound = elapsed_time_median * 1.2

                    df_filtered = df[(df['elapsed_time'] >= lower_bound) & (
                        df['elapsed_time'] <= upper_bound)]

                    if not df_filtered.empty:
                        df = df_filtered
                        print(
                            f"Filtered {len(df) - len(df_filtered)} outliers from {file}")

                df.to_csv(output_file, index=False)
                print(f"Data saved to {output_file}")
                print(f"Total records processed: {len(data)}")

                if df.empty:
                    print(f"No valid data found for {file}")
                else:
                    elapsed_time_avg = df['elapsed_time'].mean()
                    throughput_per_user = 1 / elapsed_time_avg if elapsed_time_avg > 0 else 0
                    throughput_per_user = throughput_per_user * \
                        mtp_accept_rate[mtp_num] if mtp_num > 0 else throughput_per_user
                    output_throughput = throughput_per_user * concurrency
                    output_throughput_per_gen_gpu = output_throughput / num_gen_gpus
                    gen_req_rate = output_throughput / osl
                    output_tput_per_gpu = output_throughput / \
                        (ctx_gpus * gen_req_rate/ctx_request_rate + num_gen_gpus)
                    total_tput_per_gpu = output_tput_per_gpu * (isl + osl) / osl

                    hosts_info = data[0]['hosts'] if data else ""

                    all_summary_data.append({
                        'name': name,
                        'concurrency': concurrency,
                        'throughput_per_user': throughput_per_user,
                        'output_tput_per_gpu': output_tput_per_gpu,
                        'total_tput_per_gpu': total_tput_per_gpu,
                        'output_throughput_per_gen_gpu': output_throughput_per_gen_gpu,
                        'output_throughput': output_throughput,
                        'elapsed_time_avg': elapsed_time_avg,
                        'ctx/gen inst ratio': gen_req_rate/ctx_request_rate,
                        'gen/ctx inst ratio': ctx_request_rate/gen_req_rate,
                        'ctx/gen gpu ratio': ctx_gpus * gen_req_rate/(ctx_request_rate * num_gen_gpus),
                        'gen_req_rate': gen_req_rate,
                        'number_iters': len(df),
                        'host_step_time_avg': df['host_step_time'].mean(),
                        'prev_device_step_time_avg': df['prev_device_step_time'].mean(),
                        'source_dir': dir_prefix,
                        'gen_tp_size': config['gen_tp_size'],
                        'gen_moe_ep_size': config['gen_moe_ep_size'],
                        'gen_enable_attention_dp': config['gen_enable_attention_dp'],
                    })

    if all_summary_data:
        # Create DataFrame and sort
        df = pd.DataFrame(all_summary_data)
        df = df.sort_values(['name', 'concurrency'])

        # Save as CSV
        output_file = os.path.join(output_path, "summary_iterlog_mtp0.csv" if not enable_mtp else "summary_iterlog_mtp.csv")
        df.to_csv(output_file, index=False)
        print(f"\nCombined data saved to {output_file}")
        print(f"Total records processed: {len(all_summary_data)}")

        # Create visualization
        df_grouped = create_visualization(df, output_path, enable_mtp)
        df_grouped = df_grouped.sort_values(['num_gen_gpus', 'concurrency'])

        # Save grouped data to CSV
        grouped_output_file = os.path.join(output_path, "summary_grouped_data_mtp0.csv" if not enable_mtp else "summary_grouped_data_mtp.csv")
        df_grouped.to_csv(grouped_output_file, index=False)
        print(f"Grouped data saved to {grouped_output_file}")

        # Extract frontier points
        df_sorted = df_grouped.sort_values(
            'throughput_per_user', ascending=False).reset_index(drop=True)

        frontier_indices = [0]
        max_output_tput = df_sorted.loc[0, 'output_tput_per_gpu']

        for i in range(1, len(df_sorted)):
            current_output_tput = df_sorted.loc[i, 'output_tput_per_gpu']
            if current_output_tput > max_output_tput:
                frontier_indices.append(i)
                max_output_tput = current_output_tput

        df_frontier = df_sorted.iloc[frontier_indices].copy()

        # Add parallelism label with ctx/gen and gen/ctx ratios as whole numbers
        def get_parallelism_label(row):
            gen_enable_attention_dp = row.get('gen_enable_attention_dp', False)
            gen_moe_ep_size = row.get('gen_moe_ep_size', 1)
            ctx_gen_ratio = row['ctx/gen inst ratio']
            gen_ctx_ratio = row['gen/ctx inst ratio']
            
            # Determine base label
            if gen_enable_attention_dp:
                if gen_moe_ep_size > 1:
                    base_label = 'dep'
                else:
                    base_label = 'dp'
            else:
                if gen_moe_ep_size > 1:
                    base_label = 'tep'
                else:
                    base_label = 'tp'
            
            # Convert ratios to whole number representation
            # If ctx/gen < 1, express as 1:N where N = round(gen/ctx)
            # If ctx/gen >= 1, express as N:1 where N = round(ctx/gen)
            if ctx_gen_ratio < 1:
                ctx_part = 1
                gen_part = int(np.ceil(gen_ctx_ratio))
            else:
                ctx_part = int(np.ceil(ctx_gen_ratio))
                gen_part = 1
            
            return f"{base_label}{ctx_part}:{gen_part}"
        
        df_frontier['parallelism_label'] = df_frontier.apply(get_parallelism_label, axis=1)

        # Save frontier data to CSV
        frontier_output_file = os.path.join(output_path, "summary_frontier_mtp0.csv" if not enable_mtp else "summary_frontier_mtp.csv")
        df_frontier.to_csv(frontier_output_file, index=False)
        print(
            f"Frontier data saved to {frontier_output_file} ({len(df_frontier)} points)")

        # Print incomplete jobs summary
        if incomplete_jobs:
            print(f"\n{'='*60}")
            print(f"WARNING: {len(incomplete_jobs)} incomplete job(s) found:")
            print(f"{'='*60}")
            for job_path in incomplete_jobs:
                print(f"  - {job_path}")
            print(f"{'='*60}\n")

        return df
    else:
        print("No valid data found to save")
        
        # Print incomplete jobs summary even when no valid data
        if incomplete_jobs:
            print(f"\n{'='*60}")
            print(f"WARNING: {len(incomplete_jobs)} incomplete job(s) found:")
            print(f"{'='*60}")
            for job_path in incomplete_jobs:
                print(f"  - {job_path}")
            print(f"{'='*60}\n")
        
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process benchmark files and aggregate data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python postprocess.py \\
    --dir_prefixes /path/to/results1/coreai_comparch_trtllm/ /path/to/results2/coreai_comparch_trtllm/ \\
    --output_path /path/to/combined_output/ \\
    --ctx_request_rate 1.0 \\
    --ctx_gpus 8 \\
    --isl 8192 \\
    --osl 1024
''')
    parser.add_argument(
        '--dir_prefixes', nargs='+', required=True,
        help='One or more directory prefixes to search for benchmark files')
    parser.add_argument(
        '--output_path', required=True,
        help='Output directory path for all generated artifacts')
    parser.add_argument('--ctx_request_rate', required=True,
                        help='Context request rate', type=float)
    parser.add_argument('--ctx_gpus', required=True,
                        help='Context GPUs', type=int)
    parser.add_argument(
        '--isl', required=True, help='Input sequence length', type=int)
    parser.add_argument(
        '--osl', required=True, help='Output sequence length', type=int)
    parser.add_argument('--mtp', action='store_true',
                        help='Process only data with mtp_num > 0 (default: process only mtp_num = 0)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    summary_df = process_files_multi(
        args.dir_prefixes, args.output_path, args.ctx_request_rate, 
        args.ctx_gpus, args.isl, args.osl, args.mtp)
