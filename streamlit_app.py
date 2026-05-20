import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


st.set_page_config(page_title="S-U-LM", layout="wide")

st.title("S-U-LM - Runs Explorer")

runs_dir = Path("./runs")
stage_order = [
    "use_case_extraction",
    "uml_generation",
    "uml_to_text",
    "semantic_comparison",
]
stage_labels = {
    "use_case_extraction": "Use Case Extraction",
    "uml_generation": "UML Generation",
    "uml_to_text": "UML -> Text",
    "semantic_comparison": "Semantic Comparison",
}
metric_suffixes = {
    "duration_seconds": "Duration (s)",
    "peak_gpu_mb": "Peak GPU/VRAM (MB)",
    "end_cpu_percent": "CPU % at end",
}

uploaded = st.sidebar.file_uploader("Upload runs CSV (or leave to pick latest)", type=["csv"])
if uploaded is None:
    csv_files = sorted(runs_dir.glob("run_*.csv"), reverse=True)
    if csv_files:
        csv_path = csv_files[0]
        st.sidebar.write(f"Using latest: {csv_path.name}")
        df = pd.read_csv(csv_path)
    else:
        st.sidebar.warning("No run CSV found in runs/")
        st.stop()
else:
    df = pd.read_csv(uploaded)

numeric_columns = [
    "semantic_similarity_score",
    "total_duration_seconds",
    "combo_total",
    "combo_index",
    "system_ram_gb",
    "total_vram_gb",
    "cpu_cores",
    "cpu_cores_logical",
    "cpu_freq_mhz",
    "distance",
]
for stage in stage_order:
    for suffix in metric_suffixes:
        numeric_columns.append(f"{stage}_{suffix}")

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Create a readable combination column if not present
if "combination" not in df.columns:
    use_col = "model_use_case" if "model_use_case" in df.columns else None
    uml_col = "model_uml" if "model_uml" in df.columns else None
    if use_col and uml_col:
        df["combination"] = df[use_col].astype(str) + " -> " + df[uml_col].astype(str)
    else:
        df["combination"] = df.index.astype(str)

st.sidebar.header("Filters")
all_combos = sorted(df["combination"].unique())
selected_combos = st.sidebar.multiselect("Combination", options=all_combos, default=all_combos)

score_col = "semantic_similarity_score"
if score_col in df.columns:
    min_score = float(df[score_col].min())
    max_score = float(df[score_col].max())
    smin, smax = st.sidebar.slider("Semantic score range", min_value=0.0, max_value=1.0, value=(min_score, max_score), step=0.01)
else:
    smin, smax = 0.0, 1.0

only_success = st.sidebar.checkbox("Only successful runs (all stages)", value=False)
memory_unit = st.sidebar.radio("Memory unit", options=["MB", "GB"], index=1)
time_unit = st.sidebar.radio("Time unit", options=["s", "min"], index=0)

filtered = df[df["combination"].isin(selected_combos)].copy()
if score_col in filtered.columns:
    filtered = filtered[filtered[score_col].between(smin, smax)]

if only_success:
    status_cols = [c for c in filtered.columns if c.endswith("status")]
    if status_cols:
        for sc in status_cols:
            filtered = filtered[filtered[sc] == "success"]

stage_options = [stage for stage in stage_order if f"{stage}_duration_seconds" in filtered.columns]
selected_stage = st.sidebar.selectbox(
    "Focus stage",
    options=stage_options if stage_options else stage_order,
    index=0 if stage_options else 0,
)

def stage_metric_column(stage_name: str, suffix: str) -> str:
    return f"{stage_name}_{suffix}"


def convert_memory_value(series: pd.Series) -> pd.Series:
    if memory_unit == "GB":
        return series / 1024.0
    return series


def convert_time_value(series: pd.Series) -> pd.Series:
    if time_unit == "min":
        return series / 60.0
    return series


memory_axis_label = f"Peak GPU/VRAM ({memory_unit})"
time_axis_label = f"Duration ({time_unit})"


def available_stage_metric_frame(dataframe: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows = []
    for stage_name in stage_order:
        column_name = stage_metric_column(stage_name, suffix)
        if column_name in dataframe.columns:
            rows.append(
                pd.DataFrame(
                    {
                        "stage": stage_name,
                        "stage_label": stage_labels.get(stage_name, stage_name),
                        "value": dataframe[column_name],
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["stage", "stage_label", "value"])
    return pd.concat(rows, ignore_index=True)


def stage_summary_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for stage_name in stage_order:
        duration_col = stage_metric_column(stage_name, "duration_seconds")
        gpu_col = stage_metric_column(stage_name, "peak_gpu_mb")
        cpu_col = stage_metric_column(stage_name, "end_cpu_percent")
        if duration_col in dataframe.columns:
            summary_rows.append(
                {
                    "stage": stage_labels.get(stage_name, stage_name),
                    "duration_s": dataframe[duration_col].mean(),
                    "gpu_mb": dataframe[gpu_col].mean() if gpu_col in dataframe.columns else None,
                    "cpu_percent": dataframe[cpu_col].mean() if cpu_col in dataframe.columns else None,
                }
            )
    return pd.DataFrame(summary_rows)

st.header("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Runs", len(filtered))
if score_col in filtered.columns and len(filtered) > 0:
    col2.metric("Avg Semantic", f"{filtered[score_col].mean():.3f}")
else:
    col2.metric("Avg Semantic", "N/A")
if "total_duration_seconds" in filtered.columns:
    avg_total_time = convert_time_value(filtered["total_duration_seconds"]).mean()
    col3.metric(f"Avg Total Time ({time_unit})", f"{avg_total_time:.2f}")
else:
    col3.metric(f"Avg Total Time ({time_unit})", "N/A")
if "total_vram_gb" in filtered.columns:
    col4.metric("VRAM (GB)", f"{filtered['total_vram_gb'].dropna().iloc[0]:.2f}" if len(filtered) else "N/A")
else:
    col4.metric("VRAM (GB)", "N/A")

st.markdown("---")

st.header("Table")
st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

st.markdown("---")
overview_tab, stages_tab, data_tab = st.tabs(["Overview", "Stages", "Data"])

with overview_tab:
    st.subheader("Semantic score by combination")
    if score_col in filtered.columns:
        agg = filtered.groupby("combination")[score_col].mean().reset_index().sort_values(score_col, ascending=False)
        fig = px.bar(agg, x=score_col, y="combination", orientation="h", height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No semantic score column available in CSV")

    st.subheader("Score vs total time")
    if score_col in filtered.columns and "total_duration_seconds" in filtered.columns:
        time_display = convert_time_value(filtered["total_duration_seconds"])
        scatter = px.scatter(
            filtered.assign(total_time_display=time_display),
            x="total_time_display",
            y=score_col,
            color="combination",
            hover_data=[c for c in ["model_use_case", "model_uml", "combo_index", "combo_total"] if c in filtered.columns],
            height=420,
        )
        scatter.update_layout(xaxis_title=f"Total time ({time_unit})")
        st.plotly_chart(scatter, use_container_width=True)
    else:
        st.info("Need semantic score and total duration columns for the scatter plot")

    st.subheader(f"Semantic score x consumption ({selected_stage})")
    selected_stage_gpu_col = stage_metric_column(selected_stage, "peak_gpu_mb")
    selected_stage_duration_col = stage_metric_column(selected_stage, "duration_seconds")
    if score_col in filtered.columns and selected_stage_gpu_col in filtered.columns:
        semantic_consumption = filtered.copy()
        semantic_consumption[memory_axis_label] = convert_memory_value(semantic_consumption[selected_stage_gpu_col])
        fig_semantic_consumption = px.scatter(
            semantic_consumption,
            x=memory_axis_label,
            y=score_col,
            color="combination",
            hover_data=[c for c in ["model_use_case", "model_uml", selected_stage_duration_col, "total_duration_seconds"] if c in semantic_consumption.columns],
            height=430,
            title=f"Semantic score vs {selected_stage} consumption",
        )
        fig_semantic_consumption.update_layout(xaxis_title=memory_axis_label, yaxis_title="Semantic score")
        st.plotly_chart(fig_semantic_consumption, use_container_width=True)
    else:
        st.info("Need semantic score and selected stage GPU/VRAM columns for this chart")

    st.subheader(f"Semantic score x consumption x time ({selected_stage})")
    if score_col in filtered.columns and selected_stage_gpu_col in filtered.columns and "total_duration_seconds" in filtered.columns:
        bubble = filtered.copy()
        bubble[memory_axis_label] = convert_memory_value(bubble[selected_stage_gpu_col])
        bubble[time_axis_label] = convert_time_value(bubble["total_duration_seconds"])
        fig_bubble = px.scatter(
            bubble,
            x=memory_axis_label,
            y=score_col,
            size=time_axis_label,
            color="combination",
            hover_data=[c for c in ["model_use_case", "model_uml", "combo_index", "combo_total"] if c in bubble.columns],
            height=460,
            title=f"Semantic score vs {selected_stage} consumption with total time as bubble size",
            size_max=32,
        )
        fig_bubble.update_layout(xaxis_title=memory_axis_label, yaxis_title="Semantic score")
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("Need semantic score, selected stage GPU/VRAM and total duration columns for the bubble chart")

    st.subheader("Use case x UML heatmap")
    if ("model_use_case" in filtered.columns) and ("model_uml" in filtered.columns) and (score_col in filtered.columns):
        pivot = filtered.pivot_table(index="model_use_case", columns="model_uml", values=score_col, aggfunc="mean")
        fig2 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Need model_use_case, model_uml and semantic score columns for heatmap")

with stages_tab:
    st.subheader("Average resource usage by stage")
    stage_summary = stage_summary_frame(filtered)
    if not stage_summary.empty:
        stage_summary_display = stage_summary.copy()
        stage_summary_display["duration_display"] = convert_time_value(stage_summary_display["duration_s"])
        stage_summary_display["gpu_display"] = convert_memory_value(stage_summary_display["gpu_mb"])
        c1, c2, c3 = st.columns(3)
        if stage_summary_display["duration_display"].notna().any():
            fig_duration = px.bar(stage_summary_display, x="stage", y="duration_display", title=f"Average duration per stage ({time_unit})", height=360)
            fig_duration.update_layout(yaxis_title=f"Duration ({time_unit})")
            c1.plotly_chart(fig_duration, use_container_width=True)
        else:
            c1.info("No stage duration columns available")

        if stage_summary_display["gpu_display"].notna().any():
            fig_gpu = px.bar(stage_summary_display, x="stage", y="gpu_display", title=f"Average peak GPU/VRAM per stage ({memory_unit})", height=360)
            fig_gpu.update_layout(yaxis_title=f"Peak GPU/VRAM ({memory_unit})")
            c2.plotly_chart(fig_gpu, use_container_width=True)
        else:
            c2.info("No stage GPU columns available")

        if stage_summary["cpu_percent"].notna().any():
            fig_cpu = px.bar(stage_summary, x="stage", y="cpu_percent", title="Average CPU end % per stage", height=360)
            c3.plotly_chart(fig_cpu, use_container_width=True)
        else:
            c3.info("No stage CPU columns available")
    else:
        st.info("No per-stage metrics found in the selected data")

    st.subheader(f"Selected stage: {stage_labels.get(selected_stage, selected_stage)}")
    stage_columns = [
        stage_metric_column(selected_stage, suffix)
        for suffix in metric_suffixes
        if stage_metric_column(selected_stage, suffix) in filtered.columns
    ]
    if stage_columns:
        stage_view = filtered[["combination"] + stage_columns].copy()
        stage_view = stage_view.rename(columns={
            stage_metric_column(selected_stage, "duration_seconds"): time_axis_label,
            stage_metric_column(selected_stage, "peak_gpu_mb"): memory_axis_label,
            stage_metric_column(selected_stage, "end_cpu_percent"): "CPU %",
        })
        if time_axis_label in stage_view.columns:
            stage_view[time_axis_label] = convert_time_value(stage_view[time_axis_label])
        if memory_axis_label in stage_view.columns:
            stage_view[memory_axis_label] = convert_memory_value(stage_view[memory_axis_label])
        melted = stage_view.melt(id_vars=["combination"], var_name="metric", value_name="value")
        melted["metric"] = melted["metric"].map(
            {
                time_axis_label: f"Duration ({time_unit})",
                memory_axis_label: f"Peak GPU/VRAM ({memory_unit})",
                "CPU %": "CPU % at end",
            }
        )
        melted = melted.dropna(subset=["value"])
        fig_stage = px.bar(
            melted,
            x="combination",
            y="value",
            color="metric",
            barmode="group",
            height=460,
            title=f"{stage_labels.get(selected_stage, selected_stage)} metrics by combination",
        )
        fig_stage.update_layout(yaxis_title="Value")
        st.plotly_chart(fig_stage, use_container_width=True)
    else:
        st.info("Selected stage does not have metrics in this CSV")

with data_tab:
    st.subheader("Filtered rows")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.header("Download")
csv_data = filtered.to_csv(index=False)
st.download_button("Download filtered CSV", data=csv_data, file_name="filtered_runs.csv", mime="text/csv")
