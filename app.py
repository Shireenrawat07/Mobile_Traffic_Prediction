"""
Mobile Traffic Prediction — Federated Learning Dashboard
Run:  streamlit run app.py
"""

import os, json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FL Traffic Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] label {
    color: #8b949e !important; font-size: 0.75rem;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.main { background: #0d1117; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.metric-card {
    background: linear-gradient(135deg, #161b27 0%, #1c2333 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 1.4rem 1rem; text-align: center; transition: border-color .2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-label { font-size: 0.68rem; letter-spacing: .12em; text-transform: uppercase; color: #8b949e; margin-bottom: .4rem; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.55rem; font-weight: 700; color: #58a6ff; }
.metric-sub   { font-size: 0.65rem; color: #6e7681; margin-top: .2rem; }

.stTabs [data-baseweb="tab-list"] {
    background: #161b27; border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8b949e; font-size: .85rem; padding: .5rem 1.2rem; }
.stTabs [aria-selected="true"] { background: #21262d !important; color: #58a6ff !important; font-weight: 600; }

.sec { font-family: 'Space Mono', monospace; font-size: .78rem; letter-spacing: .15em;
       text-transform: uppercase; color: #58a6ff;
       border-bottom: 1px solid #21262d; padding-bottom: .4rem; margin-bottom: .9rem; }

.info  { background: #0d2233; border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0; padding: .75rem 1rem; margin: .4rem 0; font-size: .84rem; color: #c9d1d9; }
.warn  { background: #1c1a10; border-left: 3px solid #d29922; border-radius: 0 8px 8px 0; padding: .75rem 1rem; margin: .4rem 0; font-size: .84rem; color: #c9d1d9; }
.ok    { background: #0d2a1a; border-left: 3px solid #3fb950; border-radius: 0 8px 8px 0; padding: .75rem 1rem; margin: .4rem 0; font-size: .84rem; color: #c9d1d9; }

.stButton > button {
    background: linear-gradient(90deg, #1f6feb, #388bfd); color: white; border: none;
    border-radius: 8px; font-family: 'DM Sans', sans-serif; font-weight: 600;
    padding: .5rem 1.5rem; width: 100%; transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }
.pred-box {
    text-align: center; padding: 2rem;
    background: linear-gradient(135deg, #0d2a1a, #0a1f2e);
    border: 1px solid #3fb950; border-radius: 16px; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b27",
    font=dict(family="DM Sans", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#21262d"),
    legend=dict(bgcolor="#161b27", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=50, r=30, t=50, b=50),
)
STRAT_CLR = {"FedAvg": "#58a6ff", "FedProx": "#3fb950", "FedNova": "#d2a8ff"}

# ══════════════════════════════════════════════════════════════════
# ▶  DATA PATH HELPERS
#    All paths are built from the three sidebar keys:
#      strategy  → "FedAvg" | "FedProx" | "FedNova"
#      model     → "GRU"   | "LSTM"    | "RNN"    | "MLP"
#      alpha     → 0.1     | 0.5       | 1.0
# ══════════════════════════════════════════════════════════════════

def metrics_json_path(strategy: str, alpha: float) -> str:
    """results/{strategy_lower}_results/metrics_alpha_{alpha}.json"""
    return os.path.join("results",
                        f"{strategy.lower()}_results",
                        f"metrics_alpha_{alpha}.json")

def summary_csv_path() -> str:
    """Primary comparison CSV; tries several known filenames."""
    candidates = [
        os.path.join("results", "results_all_three.csv"),
        os.path.join("results", "fedavg_fedprox_fednova_results.csv"),
        os.path.join("results", "fedavg_fedprox_fednova_results.CSV"),
    ]
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


def model_results_csv_path(model: str) -> str | None:
    """
    Per-model results CSV.
    Naming convention: results/{MODEL}_MODEL_RESULTS.CSV
    e.g. results/GRU_MODEL_RESULTS.CSV
    """
    candidates = [
        os.path.join("results/Models_Results", f"{model.upper()}_MODEL_RESULTS.CSV"),
        os.path.join("results", f"{model.upper()}_MODEL_RESULTS.csv"),
        os.path.join("results", f"{model.lower()}_model_results.csv"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)

def plot_image_path(model: str) -> str:
    """plots/{model_lower}_vs_actual.png"""
    return os.path.join("plots", f"{model.lower()}_vs_actual.png")

def model_pth_path(strategy: str, alpha: float) -> str | None:
    """Find the saved .pth/.pt for the selected strategy + alpha."""
    a = str(alpha)
    candidates = [
        # Per-strategy folders
        os.path.join("results", f"{strategy.lower()}_results", f"model_alpha_{a}.pth"),
        os.path.join("results", f"{strategy.lower()}_results", f"model_alpha_{a}.pt"),
        # Legacy: fednova uses fednova_ prefix and .pt
        os.path.join("fednova_results", f"fednova_model_alpha_{a}.pt"),
        os.path.join("fedavg_results",  f"model_alpha_{a}.pth"),
        os.path.join("fedprox_results", f"model_alpha_{a}.pth"),
        # Root fallbacks
        f"global_model_{strategy.lower()}.pth",
        "global_model.pth",
    ]
    return next((p for p in candidates if os.path.exists(p)), None)

# ══════════════════════════════════════════════════════════════════
# ▶  METRICS LOADER  (never cached — must update on every change)
# ══════════════════════════════════════════════════════════════════

def load_metrics(strategy: str, alpha: float) -> dict | None:
    """
    Load results/{strategy_lower}_results/metrics_alpha_{alpha}.json
    Returns the raw dict (keyed by client name) or None if missing.
    """
    path = metrics_json_path(strategy, alpha)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def metrics_to_df(metrics: dict) -> pd.DataFrame:
    """Flatten {client: {MAE:…, RMSE:…, NRMSE:…}} → DataFrame."""
    rows = [{"client": k, **v} for k, v in metrics.items()]
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════
# ▶  SUMMARY CSV LOADER  (cached; filter applied at call site)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_summary_csv() -> pd.DataFrame | None:
    p = summary_csv_path()
    try:
        return pd.read_csv(p) if os.path.exists(p) else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_model_results_csv(model: str) -> pd.DataFrame | None:
    """Load results/Models_Results/{MODEL}_MODEL_RESULTS.CSV for the selected model."""
    p = model_results_csv_path(model)
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def filter_summary(df: pd.DataFrame, strategy: str,
                   model: str, alpha: float) -> pd.DataFrame:
    """
    Filter the summary CSV by strategy, model, and alpha.
    Handles flexible column naming (case-insensitive).
    """
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    # Detect column names
    col_map = {c.lower(): c for c in df.columns}

    strat_col = next((col_map[k] for k in col_map
                      if any(x in k for x in ["strategy", "method", "algorithm"])), None)
    model_col = next((col_map[k] for k in col_map
                      if any(x in k for x in ["model", "arch", "network"])), None)
    alpha_col = next((col_map[k] for k in col_map
                      if "alpha" in k or "hetero" in k), None)

    if strat_col:
        out = out[out[strat_col].str.lower() == strategy.lower()]
    if model_col:
        out = out[out[model_col].str.upper() == model.upper()]
    if alpha_col:
        try:
            out = out[out[alpha_col].astype(float) == float(alpha)]
        except Exception:
            pass
    return out

# ══════════════════════════════════════════════════════════════════
# ▶  PYTORCH MODEL LOADER
# ══════════════════════════════════════════════════════════════════
OUTPUT_SIZE  = 1
SEQUENCE_LEN = 10


def _detect_input_size(ckpt: dict, model_type: str) -> int:
    key_map = {
        "LSTM": ("lstm.weight_ih_l0", 1),
        "GRU":  ("gru.weight_ih_l0",  1),
        "RNN":  ("rnn.weight_ih_l0",  1),
    }
    if model_type in key_map:
        k, d = key_map[model_type]
        if k in ckpt:
            return ckpt[k].shape[d]
    if model_type == "CNN":
        cks = [k for k in ckpt if "conv_stack" in k and k.endswith(".weight") and ckpt[k].ndim == 3]
        return ckpt[cks[0]].shape[1] if cks else 1
    if model_type == "MLP":
        lks = [k for k in ckpt if k.startswith("fc") and k.endswith(".weight")]
        return ckpt[lks[0]].shape[1] if lks else 1
    return 1



def _safe_keys(ckpt: dict, prefix: str) -> list:
    return [k for k in ckpt if k.startswith(prefix)]


def _unwrap_checkpoint(raw) -> dict:
    """Handle raw state-dicts AND wrapped checkpoints like {"model_state_dict": {...}}."""
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint is not a dict (got {type(raw)}).")
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw                                         # already a raw state-dict
    for key in ("model_state_dict", "state_dict", "model", "net"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    for v in raw.values():
        if isinstance(v, dict) and any(isinstance(vv, torch.Tensor) for vv in v.values()):
            return v
    return raw


def _autodetect_arch(ckpt: dict) -> str:
    """
    Read checkpoint keys to determine which architecture was saved.
    The sidebar 'Model Architecture' selector controls plots/CSVs only —
    all per-strategy model files currently contain the same LSTM weights.
    """
    if _safe_keys(ckpt, "lstm.weight_ih_l"):   return "LSTM"
    if _safe_keys(ckpt, "gru.weight_ih_l"):    return "GRU"
    if _safe_keys(ckpt, "rnn.weight_ih_l"):    return "RNN"
    if [k for k in ckpt if "conv_stack" in k and k.endswith(".weight")]:  return "CNN"
    if [k for k in ckpt if k.startswith("fc") and k.endswith(".weight")]: return "MLP"
    return "LSTM"   # safe default


def _build_model(ckpt: dict, arch: str):
    """Build + load model. arch is the AUTO-DETECTED architecture string."""
    input_size = _detect_input_size(ckpt, arch)

    if arch == "LSTM":
        from models.lstm_model import TrafficPredictor
        keys = _safe_keys(ckpt, "lstm.weight_ih_l")
        hs = ckpt[keys[0]].shape[0] // 4
        nl = max(int(k.split("weight_ih_l")[1]) for k in keys) + 1
        m  = TrafficPredictor(input_size=input_size, hidden_size=hs,
                               num_layers=nl, output_size=OUTPUT_SIZE)

    elif arch == "GRU":
        from models.gru_model import TrafficPredictorGRU
        keys = _safe_keys(ckpt, "gru.weight_ih_l")
        hs = ckpt[keys[0]].shape[0] // 3
        nl = max(int(k.split("weight_ih_l")[1]) for k in keys) + 1
        m  = TrafficPredictorGRU(input_size=input_size, hidden_size=hs,
                                  num_layers=nl, output_size=OUTPUT_SIZE)

    elif arch == "RNN":
        from models.rnn_model import TrafficPredictorRNN
        keys = _safe_keys(ckpt, "rnn.weight_ih_l")
        hs    = ckpt[keys[0]].shape[0]
        nl    = max(int(k.split("weight_ih_l")[1].split("_")[0]) for k in keys) + 1
        bidir = any("_reverse" in k for k in ckpt)
        m = TrafficPredictorRNN(input_size=input_size, hidden_size=hs, num_layers=nl,
                                 output_size=OUTPUT_SIZE, bidirectional=bidir)

    elif arch == "CNN":
        from models.cnn_model import TrafficPredictorCNN
        ckeys = [k for k in ckpt if "conv_stack" in k
                 and k.endswith(".weight") and ckpt[k].ndim == 3]
        nc, ks, nl = (ckpt[ckeys[0]].shape[0], ckpt[ckeys[0]].shape[2], len(ckeys))                      if ckeys else (64, 3, 3)
        m = TrafficPredictorCNN(input_size=input_size, num_channels=nc,
                                 num_layers=nl, kernel_size=ks, output_size=OUTPUT_SIZE)

    elif arch == "MLP":
        try:
            from models.mlp_model import TrafficPredictorMLP
            lkeys = [k for k in ckpt if k.startswith("fc") and k.endswith(".weight")]
            out_features = ckpt[lkeys[-1]].shape[0] if lkeys else OUTPUT_SIZE
            m = TrafficPredictorMLP(input_size=input_size, output_size=out_features)
        except ImportError:
            raise ValueError("MLP model class not found in models/mlp_model.py")
    else:
        raise ValueError(f"Unknown arch: {arch}")

    m.load_state_dict(ckpt)
    m.eval()
    return m, arch   # always return (model, detected_arch)


@st.cache_resource(show_spinner=False)
def _cached_load_model(path: str):
    """
    Cached per file-path only.
    Architecture is auto-detected — the sidebar model selector is for plots/CSVs only.
    """
    raw  = torch.load(path, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(raw)
    arch = _autodetect_arch(ckpt)
    return _build_model(ckpt, arch)   # → (model, arch)


def load_pytorch_model(strategy: str, alpha: float):
    """Returns (model, path, detected_arch, error_str)."""
    path = model_pth_path(strategy, alpha)
    if path is None:
        return None, None, None, f"No model file found for {strategy} / α={alpha}."
    try:
        m, detected_arch = _cached_load_model(path)
        return m, path, detected_arch, None
    except Exception as e:
        return None, path, None, str(e)



@st.cache_resource(show_spinner=False)
def load_scaling_params():
    for p in ["scaling_params.pt", "scaling_params.pth"]:
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                pass
    return None


def scale_input(arr: np.ndarray, params) -> np.ndarray:
    if params is None:
        return arr
    try:
        tv = lambda v: v.item() if isinstance(v, torch.Tensor) else float(v)
        if isinstance(params, dict):
            min_k = sorted(k for k in params if "min" in k.lower())
            max_k = sorted(k for k in params if "max" in k.lower())
            if min_k and max_k:
                mins = np.array([tv(params[k]) for k in min_k])
                maxs = np.array([tv(params[k]) for k in max_k])
                rng  = maxs - mins; rng[rng == 0] = 1
                return (arr - mins) / rng
            mu_k  = sorted(k for k in params if "mean" in k.lower())
            std_k = sorted(k for k in params if "std"  in k.lower())
            if mu_k and std_k:
                means = np.array([tv(params[k]) for k in mu_k])
                stds  = np.array([tv(params[k]) for k in std_k]); stds[stds == 0] = 1
                return (arr - means) / stds
    except Exception:
        pass
    return arr

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Dashboard")
    st.markdown("---")
    st.markdown('<div class="sec">Configuration</div>', unsafe_allow_html=True)

    fl_strategy = st.selectbox("FL Strategy",       ["FedAvg", "FedProx", "FedNova"])
    model_type  = st.selectbox("Model Architecture", ["GRU", "LSTM", "RNN", "MLP"])
    alpha       = st.selectbox("Alpha (α)",          [0.1, 0.5, 1.0],
                                help="Data heterogeneity — changes results completely")

    st.markdown("---")
    st.markdown('<div class="sec">Status</div>', unsafe_allow_html=True)

    # Metrics JSON status
    json_path = metrics_json_path(fl_strategy, alpha)
    if os.path.exists(json_path):
        st.markdown(f'<div class="ok"> Metrics JSON found<br><small style="color:#6e7681">{json_path}</small></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">⚠️ JSON not found<br><small>{json_path}</small></div>',
                    unsafe_allow_html=True)

    # Plot image status
    img_path = plot_image_path(model_type)
    if os.path.exists(img_path):
        st.markdown(f'<div class="ok"> Plot found<br><small style="color:#6e7681">{img_path}</small></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">⚠️ Plot not found<br><small>{img_path}</small></div>',
                    unsafe_allow_html=True)

    # Model weights status — arch is AUTO-DETECTED from checkpoint keys
    model, loaded_path, detected_arch, err = load_pytorch_model(fl_strategy, alpha)
    if model:
        arch_note = f" (actual: {detected_arch})" if detected_arch != model_type else ""
        st.markdown(
            f'<div class="ok"> <b>{detected_arch}</b> loaded{arch_note}<br>'
            f'<small style="color:#6e7681">{loaded_path}</small></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">⚠️ {err}</div>', unsafe_allow_html=True)

    scaling = load_scaling_params()
    if scaling:
        st.markdown('<div class="ok"> scaling_params.pt loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn">⚠️ scaling_params.pt not found</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Mobile Traffic Prediction · FL Comparison")

# ══════════════════════════════════════════════════════════════════
# HEADER  (updates on every sidebar change)
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<h1 style='font-family:Space Mono,monospace;font-size:1.55rem;color:#e6edf3;margin-bottom:.2rem;'>
  Mobile Traffic Prediction
</h1>
<p style='color:#8b949e;font-size:.9rem;margin-top:0;'>
  {fl_strategy} &nbsp;·&nbsp; {model_type} &nbsp;·&nbsp; α = {alpha}
  &nbsp;<span style='color:#30363d'>|</span>&nbsp;
  <span style='color:#6e7681;font-size:.8rem;'>Metrics from:
    <code style='color:#58a6ff'>{json_path}</code>
  </span>
</p>
<hr style='border:none;border-top:1px solid #21262d;margin:.8rem 0 1.4rem;'>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD THIS SELECTION'S METRICS  (no cache — must reload on change)
# ══════════════════════════════════════════════════════════════════
raw_metrics = load_metrics(fl_strategy, alpha)   # dict | None
metrics_df  = metrics_to_df(raw_metrics) if raw_metrics else None

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  Metrics & Comparison",
    "📈  Predicted vs Actual",
    "⚡  Live Inference",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — Metrics & Comparison
# ──────────────────────────────────────────────────────────────────
with tab1:

    # ── Section A: Current selection summary cards ───────────────
    st.markdown(
        f'<div class="sec">{fl_strategy} · {model_type} · α={alpha} — Client Results</div>',
        unsafe_allow_html=True)

    if metrics_df is None:
        st.markdown(
            f'<div class="warn">No JSON found at <code>{json_path}</code>.<br>'
            f'Run the training script for this strategy/alpha combination first.</div>',
            unsafe_allow_html=True)
    else:
        metric_cols = [c for c in ["MAE", "RMSE", "NRMSE"] if c in metrics_df.columns]

        # Summary stat cards
        c1, c2, c3, c4 = st.columns(4)
        n_clients = metrics_df["client"].nunique() if "client" in metrics_df.columns else "—"
        vals = [
            ("Clients",   str(n_clients),                                     ""),
            ("Avg MAE",   f'{metrics_df["MAE"].mean():.4f}'   if "MAE"   in metrics_df.columns else "—", ""),
            ("Avg RMSE",  f'{metrics_df["RMSE"].mean():.4f}'  if "RMSE"  in metrics_df.columns else "—", ""),
            ("Avg NRMSE", f'{metrics_df["NRMSE"].mean():.4f}' if "NRMSE" in metrics_df.columns else "—", ""),
        ]
        for col, (lbl, val, sub) in zip([c1, c2, c3, c4], vals):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-client bar chart
        if metric_cols:
            fig = make_subplots(rows=1, cols=len(metric_cols), subplot_titles=metric_cols)
            palette = ["#58a6ff", "#3fb950", "#d2a8ff", "#ffa657", "#f85149", "#79c0ff"]
            for i, metric in enumerate(metric_cols, 1):
                for j, row in metrics_df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[str(row["client"])], y=[row[metric]],
                        name=str(row["client"]), showlegend=(i == 1),
                        marker_color=palette[j % len(palette)]),
                        row=1, col=i)
            fig.update_layout(
                barmode="group", height=360,
                title_text=f"{fl_strategy} · α={alpha} — per-client {', '.join(metric_cols)}",
                **PL)
            for ax in list(fig.layout):
                if ax.startswith(("xaxis", "yaxis")):
                    fig.layout[ax].update(gridcolor="#21262d", linecolor="#30363d")
            st.plotly_chart(fig, width="stretch")

        # Raw table
        st.markdown('<div class="sec" style="margin-top:1.2rem;">Raw JSON Data</div>',
                    unsafe_allow_html=True)
        st.dataframe(metrics_df, width="stretch")

    # ── Section B: Alpha sensitivity (all alphas, this strategy) ─
    st.markdown(
        f'<div class="sec" style="margin-top:2rem;">Alpha Sensitivity — {fl_strategy}</div>',
        unsafe_allow_html=True)

    all_alpha_rows = []
    for a in [0.1, 0.5, 1.0]:
        m = load_metrics(fl_strategy, a)
        if m:
            for client_key, metrics_val in m.items():
                all_alpha_rows.append({"client": client_key, "alpha": a, **metrics_val})

    if all_alpha_rows:
        alpha_df   = pd.DataFrame(all_alpha_rows)
        metric_opt = [c for c in ["MAE", "RMSE", "NRMSE"] if c in alpha_df.columns]
        if metric_opt:
            sel_metric = st.selectbox("Metric", metric_opt, key="alpha_metric")
            # Highlight selected alpha
            alpha_df["opacity"] = alpha_df["alpha"].apply(lambda a: 1.0 if a == alpha else 0.35)

            fig_a = go.Figure()
            for a_val in [0.1, 0.5, 1.0]:
                sub = alpha_df[alpha_df["alpha"] == a_val]
                opacity = 1.0 if a_val == alpha else 0.35
                fig_a.add_trace(go.Bar(
                    x=sub["client"].astype(str),
                    y=sub[sel_metric],
                    name=f"α={a_val}",
                    marker=dict(
                        color=STRAT_CLR.get(fl_strategy, "#58a6ff"),
                        opacity=opacity,
                    ),
                ))
            fig_a.update_layout(
                barmode="group", height=340,
                title_text=f"{fl_strategy} — {sel_metric} across all alphas (α={alpha} highlighted)",
                **PL)
            st.plotly_chart(fig_a, width="stretch")

            # Heatmap
            pivot = alpha_df.pivot_table(index="client", columns="alpha", values=sel_metric)
            fig_h = px.imshow(pivot, text_auto=".4f", color_continuous_scale="Blues",
                               title=f"{sel_metric} heatmap — {fl_strategy}")
            fig_h.update_layout(**PL, height=300)
            st.plotly_chart(fig_h, width="stretch")
    else:
        st.markdown(
            f'<div class="warn">No alpha data found for {fl_strategy}. '
            f'Expected JSONs in <code>results/{fl_strategy.lower()}_results/</code>.</div>',
            unsafe_allow_html=True)

    # ── Section C: Per-model results CSV ─────────────────────────
    st.markdown(
        f'<div class="sec" style="margin-top:2rem;">{model_type} Model — All Results</div>',
        unsafe_allow_html=True)

    model_csv_df = load_model_results_csv(model_type)
    model_csv_path_str = model_results_csv_path(model_type) or f"results/{model_type.upper()}_MODEL_RESULTS.CSV"

    if model_csv_df is not None:
        # Filter by alpha if the column exists
        col_map_m = {c.lower(): c for c in model_csv_df.columns}
        alpha_col_m = next((col_map_m[k] for k in col_map_m if "alpha" in k), None)
        if alpha_col_m:
            filt_m = model_csv_df[model_csv_df[alpha_col_m].astype(str) == str(alpha)]
            show_model_df = filt_m if not filt_m.empty else model_csv_df
        else:
            show_model_df = model_csv_df

        st.markdown(
            f'<div class="info">Loaded <code>{model_csv_path_str}</code>'
            + (f' · filtered to α={alpha}' if alpha_col_m else "")
            + '</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.dataframe(show_model_df, width="stretch")
        with col_right:
            num_m = show_model_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_m:
                sel_m2   = st.selectbox("Plot metric", num_m, key="model_metric")
                cat_m    = show_model_df.select_dtypes(exclude=[np.number]).columns.tolist()
                x_m      = cat_m[0] if cat_m else None
                clr_m    = next((c for c in show_model_df.columns
                                  if any(kw in c.lower() for kw in
                                         ["strategy","method","algorithm"])), None)
                fig_m = px.bar(show_model_df,
                                x=x_m or show_model_df.index, y=sel_m2,
                                color=clr_m, barmode="group",
                                color_discrete_map=STRAT_CLR,
                                title=f"{model_type} — {sel_m2}")
                fig_m.update_layout(height=340, **PL)
                st.plotly_chart(fig_m, width="stretch")
    else:
        st.markdown(
            f'<div class="warn">Not found: <code>{model_csv_path_str}</code></div>',
            unsafe_allow_html=True)

    # ── Section D: Cross-strategy comparison CSV ──────────────────
    st.markdown('<div class="sec" style="margin-top:2rem;">Cross-Strategy Comparison</div>',
                unsafe_allow_html=True)

    summary_df = load_summary_csv()
    if summary_df is None:
        st.markdown(
            f'<div class="warn">Summary CSV not found. Expected at '
            f'<code>{summary_csv_path()}</code>.</div>',
            unsafe_allow_html=True)
    else:
        filtered_summary = filter_summary(summary_df, fl_strategy, model_type, alpha)
        st.markdown(
            f'<div class="info">Loaded <code>{summary_csv_path()}</code> · '
            f'filtered by strategy=<b>{fl_strategy}</b>, model=<b>{model_type}</b>, α=<b>{alpha}</b></div>',
            unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            show_df = filtered_summary if not filtered_summary.empty else summary_df
            st.dataframe(show_df, width="stretch")
        with col_right:
            num_cols = show_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                sel_m2 = st.selectbox("Plot metric", num_cols, key="cmp_metric")
                cat_cols = show_df.select_dtypes(exclude=[np.number]).columns.tolist()
                x_col    = cat_cols[0] if cat_cols else None
                color_col = next((c for c in show_df.columns
                                   if any(kw in c.lower() for kw in ["strategy","method"])), None)
                fig_cmp = px.bar(show_df,
                                  x=x_col or show_df.index, y=sel_m2,
                                  color=color_col, barmode="group",
                                  color_discrete_map=STRAT_CLR,
                                  title=f"{sel_m2} — cross-strategy")
                fig_cmp.update_layout(height=350, **PL)
                st.plotly_chart(fig_cmp, width="stretch")

# ──────────────────────────────────────────────────────────────────
# TAB 2 — Predicted vs Actual  (image from /plots/)
# ──────────────────────────────────────────────────────────────────
with tab2:

    # ── Helper: safely load image or show warning ──────────────────
    def show_plot(path, caption="", use_full_width=True):
        if os.path.exists(path):
            st.image(path, caption=caption,
                     use_container_width=use_full_width)
        else:
            st.markdown(
                f'<div class="warn">⚠️ Not found: '
                f'<code>{path}</code></div>',
                unsafe_allow_html=True)

    # ── Three organised panels ────────────────────────────────────
    panel1, panel2, panel3 = st.tabs([
        "🏆  Final Results",
        "🤖  Model Comparison",
        "⚔️  Strategy Comparison",
    ])

    # ══════════════════════════════════════════════════════════════
    # PANEL 1 — FINAL RESULTS
    # ══════════════════════════════════════════════════════════════
    with panel1:
        st.markdown(
            '<div class="sec">Project Final Results — All Models</div>',
            unsafe_allow_html=True)

        # Hard metric cards from the bar chart values
        st.markdown("**Performance Summary (FedAvg global model)**")
        models_data = {
            "GRU":  {"MAE": 0.0135, "NRMSE": 0.0289, "color": "#3fb950"},
            "LSTM": {"MAE": 0.0137, "NRMSE": 0.0286, "color": "#58a6ff"},
            "CNN":  {"MAE": 0.0138, "NRMSE": 0.0294, "color": "#d2a8ff"},
            "RNN":  {"MAE": 0.0184, "NRMSE": 0.0340, "color": "#ffa657"},
            "MLP":  {"MAE": 0.0410, "NRMSE": 0.0466, "color": "#f85149"},
        }
        cols = st.columns(5)
        for col, (name, vals) in zip(cols, models_data.items()):
            border_color = "#3fb950" if name == model_type else "#30363d"
            selected_tag = " ✓" if name == model_type else ""
            col.markdown(f"""
            <div class="metric-card"
                 style="border-color:{border_color};border-width:2px;">
              <div class="metric-label"
                   style="color:{vals['color']};">{name}{selected_tag}</div>
              <div class="metric-value"
                   style="color:{vals['color']};font-size:1.2rem;">
                   {vals['MAE']:.4f}</div>
              <div class="metric-sub">MAE</div>
              <div style="font-family:Space Mono,monospace;font-size:.9rem;
                          color:{vals['color']};margin-top:.3rem;">
                   {vals['NRMSE']:.4f}</div>
              <div class="metric-sub">NRMSE</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Full-width model bar graph
        show_plot(
            os.path.join("plots", "models_bar_graph.png"),
            caption="Performance Comparison — All Deep Models (MAE & NRMSE)"
        )

        st.markdown('<div class="sec" style="margin-top:1.5rem;">'
                    'Selected Model — Predicted vs Actual</div>',
                    unsafe_allow_html=True)

        # Show the currently selected model's plot prominently
        selected_img = os.path.join(
            "plots", f"{model_type.lower()}_vs_actual.png")
        show_plot(
            selected_img,
            caption=f"{model_type} Model — Predicted vs Actual Traffic"
        )

        # Key finding callout
        st.markdown(f"""
        <div class="info" style="margin-top:1rem;">
          <b>Key Finding:</b> GRU achieves the lowest MAE (0.0135) and
          LSTM achieves the lowest NRMSE (0.0286), making them the
          top performers for federated mobile traffic prediction.
          MLP underperforms significantly due to its inability to
          model temporal dependencies in time-series data.
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # PANEL 2 — MODEL COMPARISON
    # ══════════════════════════════════════════════════════════════
    with panel2:
        st.markdown(
            '<div class="sec">Predicted vs Actual — All Models</div>',
            unsafe_allow_html=True)

        # Model selector to highlight one
        all_models = ["GRU", "LSTM", "CNN", "RNN", "MLP"]
        highlight = st.radio(
            "Highlight model",
            all_models,
            index=all_models.index(model_type)
                  if model_type in all_models else 0,
            horizontal=True,
            key="model_highlight_radio"
        )

        # Show highlighted model full width first
        st.markdown(f"**{highlight} — Full view**")
        show_plot(
            os.path.join("plots", f"{highlight.lower()}_vs_actual.png"),
            caption=f"{highlight}: MAE={models_data[highlight]['MAE']:.4f}  "
                    f"NRMSE={models_data[highlight]['NRMSE']:.4f}"
        )

        st.markdown('<div class="sec" style="margin-top:1.5rem;">'
                    'All Models — Side by Side</div>',
                    unsafe_allow_html=True)

        # 2-column grid for all models
        row1_models = ["GRU", "LSTM", "CNN"]
        row2_models = ["RNN", "MLP"]

        cols_r1 = st.columns(3)
        for col, name in zip(cols_r1, row1_models):
            with col:
                img_path = os.path.join(
                    "plots", f"{name.lower()}_vs_actual.png")
                border = ("3px solid " + models_data[name]["color"]
                          if name == highlight else "1px solid #30363d")
                st.markdown(
                    f'<div style="border:{border};border-radius:8px;'
                    f'padding:4px;margin-bottom:4px;">'
                    f'<p style="color:{models_data[name]["color"]};'
                    f'font-weight:700;margin:4px 0 2px 4px;">'
                    f'{name}  •  MAE {models_data[name]["MAE"]:.4f}  '
                    f'NRMSE {models_data[name]["NRMSE"]:.4f}</p></div>',
                    unsafe_allow_html=True)
                show_plot(img_path, caption="")

        cols_r2 = st.columns(2)
        for col, name in zip(cols_r2, row2_models):
            with col:
                img_path = os.path.join(
                    "plots", f"{name.lower()}_vs_actual.png")
                border = ("3px solid " + models_data[name]["color"]
                          if name == highlight else "1px solid #30363d")
                st.markdown(
                    f'<div style="border:{border};border-radius:8px;'
                    f'padding:4px;margin-bottom:4px;">'
                    f'<p style="color:{models_data[name]["color"]};'
                    f'font-weight:700;margin:4px 0 2px 4px;">'
                    f'{name}  •  MAE {models_data[name]["MAE"]:.4f}  '
                    f'NRMSE {models_data[name]["NRMSE"]:.4f}</p></div>',
                    unsafe_allow_html=True)
                show_plot(img_path, caption="")

    # ══════════════════════════════════════════════════════════════
    # PANEL 3 — STRATEGY COMPARISON
    # ══════════════════════════════════════════════════════════════
    with panel3:
        st.markdown(
            '<div class="sec">FL Strategy Comparison — RMSE across α</div>',
            unsafe_allow_html=True)

        # Strategy interpretation callout
        st.markdown("""
        <div class="info">
          These plots show RMSE at Dirichlet α = 0.1, 0.5, 1.0.
          Lower α = more heterogeneous (harder) data distribution.
          <b>FedAvg consistently outperforms FedProx and FedNova</b>
          on this dataset, suggesting the proximal constraint and
          normalisation introduce unnecessary complexity for this
          traffic prediction task.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main 3-strategy comparison — full width
        st.markdown("**All Three Strategies — RMSE vs α**")
        show_plot(
            os.path.join("plots", "rmse_fedavg_fednova_fedprox.png"),
            caption="RMSE Comparison: FedAvg vs FedProx vs FedNova "
                    "across Dirichlet α values"
        )

        st.markdown(
            '<div class="sec" style="margin-top:1.5rem;">'
            'Pairwise Strategy Comparisons</div>',
            unsafe_allow_html=True)

        # Side-by-side pairwise
        col_l, col_r = st.columns(2)
        with col_l:
            show_plot(
                os.path.join("plots", "rmse_fedavg_fedprox.png"),
                caption="FedAvg vs FedProx — RMSE across α"
            )
        with col_r:
            show_plot(
                os.path.join("plots", "rmse_fedavg_fednova.png"),
                caption="FedAvg vs FedNova — RMSE across α"
            )

        # Training convergence comparison
        st.markdown(
            '<div class="sec" style="margin-top:1.5rem;">'
            'Training Convergence — Loss per Round</div>',
            unsafe_allow_html=True)

        st.markdown("""
        <div class="info">
          These plots compare training loss convergence across 30
          communication rounds. All strategies converge rapidly in
          the first 5 rounds and plateau near round 10, confirming
          that 30 rounds is sufficient for this task.
        </div>""", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            show_plot(
                os.path.join("plots", "fedavg_median_comp.png"),
                caption="FedAvg vs MedianAvg — Loss per Round"
            )
        with col_b:
            show_plot(
                os.path.join("plots", "simple_fedavg_comp.png"),
                caption="FedAvg vs SimpleAvg — Loss per Round"
            )
        with col_c:
            show_plot(
                os.path.join("plots", "fedavg_median_simple.png"),
                caption="FedAvg vs SimpleAvg vs MedianAvg — Loss per Round"
            )

        # Strategy ranking callout
        st.markdown("""
        <div class="sec" style="margin-top:1.5rem;">
        Strategy Ranking Summary</div>""",
        unsafe_allow_html=True)

        ranking_cols = st.columns(3)
        ranking_data = [
            ("🥇 FedAvg",   "#58a6ff", "RMSE ≈ 0.036–0.045",
             "Best overall. Simple weighted averaging is most effective "
             "for homogeneous mobile traffic patterns."),
            ("🥈 FedProx",  "#ffa657", "RMSE ≈ 0.056–0.100",
             "Proximal term helps at α=1.0 (IID) but hurts at α=0.5. "
             "μ tuning required for this dataset."),
            ("🥉 FedNova",  "#d2a8ff", "RMSE ≈ 0.179–0.184",
             "Highest error. Normalisation may be over-correcting "
             "when clients have similar data volumes."),
        ]
        for col, (label, color, rmse, note) in zip(
                ranking_cols, ranking_data):
            col.markdown(f"""
            <div class="metric-card" style="border-color:{color};
                 border-width:2px;text-align:left;padding:1rem;">
              <div style="font-size:1.1rem;font-weight:700;
                   color:{color};margin-bottom:.4rem;">{label}</div>
              <div style="font-family:Space Mono,monospace;
                   font-size:.85rem;color:{color};
                   margin-bottom:.5rem;">{rmse}</div>
              <div style="font-size:.82rem;color:#c9d1d9;">
                   {note}</div>
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 — Live Inference
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec">Live Inference</div>', unsafe_allow_html=True)
 
    def run_inference(X_np: np.ndarray) -> float:
        X_scaled = scale_input(X_np, scaling)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(X_t)
        return out.squeeze().item()
 
    if model is None:
        st.markdown(
            f'<div class="warn">⚠️ No model loaded — {err}<br>'
            f'Expected path: <code>{model_pth_path(fl_strategy, alpha) or "not found"}</code></div>',
            unsafe_allow_html=True)
    else:
        raw_check  = torch.load(loaded_path, map_location="cpu", weights_only=False)
        ckpt_check = _unwrap_checkpoint(raw_check)
        n_features = _detect_input_size(ckpt_check, detected_arch)
        feat_names = [f"feature_{i + 1}" for i in range(n_features)]
 
        arch_note2 = f" (sidebar: {model_type})" if detected_arch != model_type else ""
        st.markdown(
            f'<div class="info">Arch: <b>{detected_arch}</b>{arch_note2} · '
            f'{fl_strategy} · α={alpha} · {n_features} input feature(s)<br>'
            f'<small style="color:#6e7681">Weights: {loaded_path}</small></div>',
            unsafe_allow_html=True)
 
        input_mode = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
 
        if input_mode == "Manual entry":
            st.markdown(
                f'<div class="info"><b>{SEQUENCE_LEN} timesteps × {n_features} feature(s)</b>. '
                f'Each row = one time step.</div>', unsafe_allow_html=True)
            default_df = pd.DataFrame(np.zeros((SEQUENCE_LEN, n_features)), columns=feat_names)
            edited = st.data_editor(default_df, width="stretch", num_rows="fixed")
 
            if st.button("▶  Run Prediction"):
                try:
                    pred = run_inference(edited.values.astype(np.float32))
                    st.markdown(f"""
                    <div class="pred-box">
                      <div class="metric-label">Predicted Traffic Volume</div>
                      <div style='font-family:Space Mono,monospace;font-size:3rem;
                                  color:#3fb950;font-weight:700;'>{pred:.4f}</div>
                      <div style='color:#6e7681;font-size:.8rem;margin-top:.4rem;'>
                        {model_type} · {fl_strategy} · α={alpha}
                      </div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Inference error: {e}")
 
        else:
            st.markdown(
                f'<div class="info">Upload a CSV with ≥ <b>{n_features}</b> numeric columns. '
                f'Last <b>{SEQUENCE_LEN}</b> rows = one window.</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
 
            if uploaded:
                try:
                    df_up    = pd.read_csv(uploaded)
                    num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
                    if len(num_cols) < n_features:
                        st.error(f"Need ≥ {n_features} numeric columns, found {len(num_cols)}.")
                    else:
                        feat_data = df_up[num_cols[:n_features]].values.astype(np.float32)
                        st.caption(f"{len(df_up)} rows × {n_features} features")
                        st.dataframe(df_up[num_cols[:n_features]].tail(SEQUENCE_LEN), width="stretch")
 
                        ca, cb = st.columns(2)
                        with ca:
                            if st.button(f"▶  Predict (last {SEQUENCE_LEN} rows)"):
                                pred = run_inference(feat_data[-SEQUENCE_LEN:])
                                st.markdown(f"""
                                <div class="pred-box">
                                  <div class="metric-label">Predicted Traffic Volume</div>
                                  <div style='font-family:Space Mono,monospace;font-size:2.5rem;
                                              color:#3fb950;font-weight:700;'>{pred:.4f}</div>
                                  <div style='color:#6e7681;font-size:.8rem;margin-top:.4rem;'>
                                    {model_type} · {fl_strategy} · α={alpha}
                                  </div>
                                </div>""", unsafe_allow_html=True)
                        with cb:
                            if st.button("⚡  Batch (sliding window)"):
                                if len(feat_data) < SEQUENCE_LEN:
                                    st.warning(f"Need ≥ {SEQUENCE_LEN} rows.")
                                else:
                                    preds_b = [run_inference(feat_data[i - SEQUENCE_LEN: i])
                                               for i in range(SEQUENCE_LEN, len(feat_data) + 1)]
                                    fig_b = go.Figure()
                                    fig_b.add_trace(go.Scatter(y=preds_b, mode="lines",
                                                                line=dict(color="#58a6ff", width=1.5),
                                                                name="Predicted"))
                                    fig_b.update_layout(height=300, title="Batch Predictions",
                                                         xaxis_title="Window", yaxis_title="Traffic", **PL)
                                    st.plotly_chart(fig_b, width="stretch")
                                    dl = pd.DataFrame({"prediction": preds_b})
                                    st.download_button("⬇ Download CSV", dl.to_csv(index=False),
                                                       "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")
 