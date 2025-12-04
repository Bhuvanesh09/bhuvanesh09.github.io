import pandas as pd
import plotly.graph_objects as go
import os

# Configuration
DATA_PATH = 'Data/benchmark_results_llama3.1_a10-latency_test-vllm.csv'
OUTPUT_DIR = '../bhuv-webpage/static/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read Data
df = pd.read_csv(DATA_PATH)

# Common Layout Settings
layout_settings = dict(
    width=800,
    height=600,
    plot_bgcolor='#FAF8F1',
    paper_bgcolor='#FAF8F1',
    xaxis=dict(gridcolor='#E0E0E0', gridwidth=1),
    yaxis=dict(gridcolor='#E0E0E0', gridwidth=1)
)
color_scheme = ['#4A90C0', '#FFA54F', '#9370DB', '#66CDAA', '#CD5C5C']

# --- Plot 1: Input Token Dependency (Latency vs Sequence Length) ---
print("Generating Plot 1...")
df_sem5 = df[df['Semaphores'] == 5]
fig1 = go.Figure()

# Viridis color scale approximation for discrete values if needed, 
# but the notebook uses standard numerical color mapping for 'Output Length'.
# However, the notebook snippet shows:
# marker=dict(size=8, color=output_length, colorscale='Viridis', ...)
# inside a loop over Output Length. This creates a discrete color for each trace based on the value.

for i, output_len in enumerate(df_sem5['Output Length'].unique()):
    df_subset = df_sem5[df_sem5['Output Length'] == output_len]
    color = color_scheme[i % len(color_scheme)]
    fig1.add_trace(go.Scatter(
        x=df_subset['Sequence Length'],
        y=df_subset['Mean Latency'],
        mode='lines+markers',
        name=f'Output Length: {output_len}',
        line=dict(color=color),
        marker=dict(size=8, color=color)
    ))

fig1.update_layout(
    title='Mean Latency vs Number of Input Tokens for Semaphore 5',
    xaxis_title='Number of Input Tokens (Sequence Length)',
    yaxis_title='Mean Latency',
    legend_title='Output Length',
    **layout_settings
)
fig1.write_json(f"{OUTPUT_DIR}/input_dependency.json")


# --- Plot 2: Output Token Dependency (Latency vs Output Length) ---
print("Generating Plot 2...")
# Using Semaphore 13 as per notebook usage: plot_latency_graph(..., semaphore=13)
df_sem13 = df[df['Semaphores'] == 13]
fig2 = go.Figure()

# Color by Sequence Length
seq_lengths = sorted(df_sem13['Sequence Length'].unique())
for i, seq_len in enumerate(seq_lengths):
    sub = df_sem13[df_sem13['Sequence Length'] == seq_len]
    color = color_scheme[i % len(color_scheme)]
    fig2.add_trace(go.Scatter(
        x=sub['Output Length'],
        y=sub['Mean Latency'],
        mode='lines+markers',
        name=f'Sequence Length: {seq_len}',
        line=dict(color=color),
        marker=dict(size=8, color=color)
    ))

fig2.update_layout(
    title='Mean Latency vs Output Length',
    xaxis_title='Output Length',
    yaxis_title='Mean Latency',
    legend_title='Sequence Length',
    **layout_settings
)
fig2.write_json(f"{OUTPUT_DIR}/output_dependency.json")


# --- Plot 3: Semaphore Dependency (Latency vs Semaphores) ---
print("Generating Plot 3...")
# Filter for Sequence Length 512 and Semaphores < 37
df_512 = df[(df['Sequence Length'] == 512) & (df['Semaphores'] < 37)]
fig3 = go.Figure()

output_lengths = sorted(df_512['Output Length'].unique())
for i, output_len in enumerate(output_lengths):
    sub = df_512[df_512['Output Length'] == output_len]
    color = color_scheme[i % len(color_scheme)]
    fig3.add_trace(go.Scatter(
        x=sub['Semaphores'],
        y=sub['Mean Latency'],
        mode='lines+markers',
        name=f'Output Length: {output_len}',
        line=dict(color=color, width=2),
        marker=dict(size=8, color=color)
    ))

fig3.update_layout(
    title='Mean Latency vs Semaphore Count',
    xaxis_title='Semaphore Count',
    yaxis_title='Mean Latency',
    legend_title='Output Length',
    **layout_settings
)
# Specific axis tweak for Plot 3
fig3.update_xaxes(tickmode='linear', tick0=5, dtick=8)

fig3.write_json(f"{OUTPUT_DIR}/semaphore_dependency.json")

print("Done.")
