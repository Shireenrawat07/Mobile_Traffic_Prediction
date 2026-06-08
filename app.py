"""
Mobile Traffic Prediction — Federated Learning Dashboard
Run:  streamlit run app.py
"""

import os, json, glob
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
.metric-label { font-size: 0.68rem; letter-spacing: .12em; text-transform: uppercase;
                color: #8b949e; margin-bottom: .4rem; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.55rem;
                font-weight: 700; color: #58a6ff; }
.metric-sub   { font-size: 0.65rem; color: #6e7681; margin-top: .2rem; }

.stTabs [data-baseweb="tab-list"] {
    background: #161b27; border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #8b949e; font-size: .85rem; padding: .5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important; color: #58a6ff !important; font-weight: 600;
}

.sec { font-family: 'Space Mono', monospace; font-size: .78rem; letter-spacing: .15em;
       text-transform: uppercase; color: #58a6ff;
       border-bottom: 1px solid #21262d; padding-bottom: .4rem; margin-bottom: .9rem; }

.info  { background: #0d2233; border-left: 3px solid #58a6ff;
         border-radius: 0 8px 8px 0; padding: .75rem 1rem;
         margin: .4rem 0; font-size: .84rem; color: #c9d1d9; }
.warn  { background: #1c1a10; border-left: 3px solid #d29922;
         border-radius: 0 8px 8px 0; padding: .75rem 1rem;
         margin: .4rem 0; font-size: .84rem; color: #c9d1d9; }
.ok    { background: #0d2a1a; border-left: 3px solid #3fb950;
         border-radius: 0 8px 8px 0; padding: .75rem 1rem;
         margin: .4rem 0; font-size: .84rem; color: #c9d1d9; }

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
STRAT_CLR  = {"FedAvg": "#58a6ff", "FedProx": "#3fb950", "FedNova": "#d2a8ff"}
MODEL_DATA = {
    "GRU":  {"MAE": 0.0135, "NRMSE": 0.0289, "color": "#3fb950"},
    "LSTM": {"MAE": 0.0137, "NRMSE": 0.0286, "color": "#58a6ff"},
    "CNN":  {"MAE": 0.0138, "NRMSE": 0.0294, "color": "#d2a8ff"},
    "RNN":  {"MAE": 0.0184, "NRMSE": 0.0340, "color": "#ffa657"},
    "MLP":  {"MAE": 0.0410, "NRMSE": 0.0466, "color": "#f85149"},
}

# ══════════════════════════════════════════════════════════════════
# PATH HELPERS
# ══════════════════════════════════════════════════════════════════

def metrics_json_path(strategy: str, alpha: float) -> str:
    return os.path.join("results",
                        f"{strategy.lower()}_results",
                        f"metrics_alpha_{alpha}.json")


def _detect_alphas(strategy: str) -> list:
    """Scan results folder for ALL available alpha values from JSON files."""
    folder = os.path.join("results", f"{strategy.lower()}_results")
    alphas = []
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.startswith("metrics_alpha_") and fname.endswith(".json"):
                try:
                    val = float(
                        fname.replace("metrics_alpha_", "").replace(".json", ""))
                    alphas.append(val)
                except ValueError:
                    pass
    return sorted(alphas) if alphas else [0.1, 0.5, 1.0]


def summary_csv_path() -> str:
    """Try all known summary CSV filenames."""
    candidates = [
        os.path.join("results", "results_all_four.csv"),
        os.path.join("results", "results_all_three.csv"),
        os.path.join("results", "fedavg_fedprox_fednova_results.csv"),
        os.path.join("results", "fedavg_results.csv"),
    ]
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


def model_results_csv_path(model: str) -> str | None:
    """results/Models_Results/{MODEL}_MODEL_RESULTS.CSV"""
    candidates = [
        os.path.join("results", "Models_Results",
                     f"{model.upper()}_MODEL_RESULTS.CSV"),
        os.path.join("results", "Models_Results",
                     f"{model.upper()}_MODEL_RESULTS.csv"),
        os.path.join("results", "Models_Results",
                     f"{model.lower()}_model_results.csv"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def model_pth_path(strategy: str, alpha: float, arch: str = "") -> str | None:
    """Search all known locations for a saved .pth / .pt model file."""
    a, sl, al = str(alpha), strategy.lower(), arch.lower() if arch else ""
    candidates = []
    if al:
        candidates += [
            os.path.join("results", f"{sl}_results",
                         f"{al}_model_alpha_{a}.pth"),
        ]
    candidates += [
        os.path.join("results", f"{sl}_results", f"model_alpha_{a}.pth"),
        os.path.join("results", f"{sl}_results", f"model_alpha_{a}.pt"),
        os.path.join("results", "Models_Results", f"{sl}_model_alpha_{a}.pth"),
        os.path.join("results", "Models_Results", "global_model.pth"),
        os.path.join("results", "fednova_results",
                     f"fednova_model_alpha_{a}.pt"),
        f"global_model_{sl}.pth",
        "global_model.pth",
    ]
    return next((p for p in candidates if p and os.path.exists(p)), None)


def plot_image_path(model: str) -> str:
    return os.path.join("plots", f"{model.lower()}_vs_actual.png")

# ══════════════════════════════════════════════════════════════════
# METRICS LOADER  (never cached — reloads on every sidebar change)
# ══════════════════════════════════════════════════════════════════

def load_metrics(strategy: str, alpha: float) -> dict | None:
    path = metrics_json_path(strategy, alpha)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def metrics_to_df(metrics: dict) -> pd.DataFrame:
    rows = [{"client": k, **v} for k, v in metrics.items()]
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════
# CSV LOADERS  (cached)
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
    p = model_results_csv_path(model)
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# SCALING PARAMS  — searches Dataset/ for .csv_scaler.pt files
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_scaling_params():
    search = [
        # Dataset folder — per-city scaler files
        *glob.glob(os.path.join("Dataset", "*.csv_scaler.pt")),
        *glob.glob(os.path.join("Dataset", "*.scaler.pt")),
        # Root-level fallbacks
        "scaling_params.pt",
        "scaling_params.pth",
        os.path.join("results", "scaling_params.pt"),
        os.path.join("utils", "scaling_params.pt"),
    ]
    for p in search:
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu",
                                  weights_only=False), p
            except Exception:
                pass
    return None, None


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
                stds  = np.array([tv(params[k]) for k in std_k])
                stds[stds == 0] = 1
                return (arr - means) / stds
    except Exception:
        pass
    return arr

# ══════════════════════════════════════════════════════════════════
# PYTORCH MODEL LOADER
# ══════════════════════════════════════════════════════════════════
OUTPUT_SIZE  = 1
SEQUENCE_LEN = 10


def _safe_keys(ckpt, prefix):
    return [k for k in ckpt if k.startswith(prefix)]


def _unwrap_checkpoint(raw):
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint type {type(raw).__name__}, expected dict.")
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw
    for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    for v in raw.values():
        if isinstance(v, dict) and any(isinstance(vv, torch.Tensor)
                                        for vv in v.values()):
            return v
    return raw


def _strip_prefix(ckpt):
    roots = ("lstm.", "gru.", "rnn.", "conv_stack.", "fc")
    if any(k.startswith(r) for k in ckpt for r in roots):
        return ckpt
    for prefix in ("model.", "module.", "net.", "backbone."):
        stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in ckpt.items()}
        if any(sk.startswith(r) for sk in stripped for r in roots):
            return stripped
    return ckpt


def _autodetect_arch(ckpt):
    if _safe_keys(ckpt, "lstm.weight_ih_l"):  return "LSTM"
    if _safe_keys(ckpt, "gru.weight_ih_l"):   return "GRU"
    if _safe_keys(ckpt, "rnn.weight_ih_l"):   return "RNN"
    if [k for k in ckpt if "conv_stack" in k and k.endswith(".weight")]:
        return "CNN"
    if [k for k in ckpt if k.startswith("fc") and k.endswith(".weight")]:
        return "MLP"
    return "LSTM"


def _detect_input_size(ckpt, arch):
    probe = {"LSTM": ("lstm.weight_ih_l0", 1),
             "GRU":  ("gru.weight_ih_l0",  1),
             "RNN":  ("rnn.weight_ih_l0",  1)}
    if arch in probe:
        k, d = probe[arch]
        if k in ckpt:
            return ckpt[k].shape[d]
    if arch == "CNN":
        cks = [k for k in ckpt if "conv_stack" in k
               and k.endswith(".weight") and ckpt[k].ndim == 3]
        return ckpt[cks[0]].shape[1] if cks else 1
    if arch == "MLP":
        lks = [k for k in ckpt if k.startswith("fc") and k.endswith(".weight")]
        return ckpt[lks[0]].shape[1] if lks else 1
    return 1


def _build_model(ckpt, arch):
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
        keys  = _safe_keys(ckpt, "rnn.weight_ih_l")
        hs    = ckpt[keys[0]].shape[0]
        nl    = max(int(k.split("weight_ih_l")[1].split("_")[0])
                    for k in keys) + 1
        bidir = any("_reverse" in k for k in ckpt)
        m = TrafficPredictorRNN(input_size=input_size, hidden_size=hs,
                                num_layers=nl, output_size=OUTPUT_SIZE,
                                bidirectional=bidir)
    elif arch == "CNN":
        from models.cnn_model import TrafficPredictorCNN
        ckeys = [k for k in ckpt if "conv_stack" in k
                 and k.endswith(".weight") and ckpt[k].ndim == 3]
        nc, ks, nl = (ckpt[ckeys[0]].shape[0], ckpt[ckeys[0]].shape[2],
                      len(ckeys)) if ckeys else (64, 3, 3)
        m = TrafficPredictorCNN(input_size=input_size, num_channels=nc,
                                num_layers=nl, kernel_size=ks,
                                output_size=OUTPUT_SIZE)
    elif arch == "MLP":
        try:
            from models.mlp_model import TrafficPredictorMLP
            lkeys = [k for k in ckpt if k.startswith("fc")
                     and k.endswith(".weight")]
            out_f = ckpt[lkeys[-1]].shape[0] if lkeys else OUTPUT_SIZE
            m = TrafficPredictorMLP(input_size=input_size, output_size=out_f)
        except ImportError:
            raise ValueError("mlp_model.py not found in models/")
    else:
        raise ValueError(f"Unknown arch: {arch}")
    m.load_state_dict(ckpt)
    m.eval()
    return m, arch


@st.cache_resource(show_spinner=False)
def _cached_load_model(path: str):
    raw  = torch.load(path, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(raw)
    ckpt = _strip_prefix(ckpt)
    arch = _autodetect_arch(ckpt)
    return _build_model(ckpt, arch)


def load_pytorch_model(strategy, alpha, arch=""):
    path = model_pth_path(strategy, alpha, arch)
    if path is None:
        return None, None, None, f"No .pth file found for {strategy} / α={alpha}."
    try:
        m, det_arch = _cached_load_model(path)
        return m, path, det_arch, None
    except Exception as e:
        return None, path, None, str(e)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 FL Dashboard")
    st.markdown("---")
    st.markdown('<div class="sec">Configuration</div>',
                unsafe_allow_html=True)

    fl_strategy = st.selectbox("FL Strategy",
                                ["FedAvg", "FedProx", "FedNova"])
    model_type  = st.selectbox("Model Architecture",
                                ["GRU", "LSTM", "RNN", "CNN", "MLP"])

    # Auto-detect ALL available alphas from JSON files in results folder
    _avail_alphas = _detect_alphas(fl_strategy)
    alpha = st.selectbox(
        "Alpha (α)", _avail_alphas,
        index=_avail_alphas.index(0.1) if 0.1 in _avail_alphas else 0,
        help=f"All available: {_avail_alphas}"
    )

    st.markdown("---")
    st.markdown('<div class="sec">Status</div>', unsafe_allow_html=True)

    # Metrics JSON
    json_path = metrics_json_path(fl_strategy, alpha)
    if os.path.exists(json_path):
        st.markdown(
            f'<div class="ok">✅ Metrics JSON found<br>'
            f'<small style="color:#6e7681">{json_path}</small></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warn">⚠️ JSON not found<br>'
            f'<small>{json_path}</small></div>',
            unsafe_allow_html=True)

    # Plot image
    img_path = plot_image_path(model_type)
    if os.path.exists(img_path):
        st.markdown(
            f'<div class="ok">✅ Plot found<br>'
            f'<small style="color:#6e7681">{img_path}</small></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warn">⚠️ Plot not found<br>'
            f'<small>{img_path}</small></div>',
            unsafe_allow_html=True)

    # Model weights
    model, loaded_path, detected_arch, err = load_pytorch_model(
        fl_strategy, alpha, model_type)
    if model:
        arch_note = (f" (actual: {detected_arch})"
                     if detected_arch != model_type else "")
        st.markdown(
            f'<div class="ok">✅ <b>{detected_arch}</b> loaded{arch_note}<br>'
            f'<small style="color:#6e7681">{loaded_path}</small></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warn">⚠️ No model file found<br>'
            f'<small style="color:#6e7681">'
            f'Metrics & Plots tabs work normally.<br>'
            f'Save a .pth to enable Live Inference.</small></div>',
            unsafe_allow_html=True)
        with st.expander("📁 Expected save path"):
            st.code(
                f"results/{fl_strategy.lower()}_results/"
                f"model_alpha_{alpha}.pth",
                language="bash")

    # Scaling params — searches Dataset/*.csv_scaler.pt
    scaling_data, scaling_path = load_scaling_params()
    if scaling_data:
        st.markdown(
            f'<div class="ok">✅ Scaler loaded<br>'
            f'<small style="color:#6e7681">{scaling_path}</small></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warn">⚠️ No scaler found<br>'
            '<small style="color:#6e7681">'
            'Checked Dataset/*.csv_scaler.pt<br>'
            'Only affects Live Inference scaling.</small></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Mobile Traffic Prediction · FL Comparison")

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<h1 style='font-family:Space Mono,monospace;font-size:1.55rem;
           color:#e6edf3;margin-bottom:.2rem;'>
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
# LOAD METRICS FOR CURRENT SELECTION
# ══════════════════════════════════════════════════════════════════
raw_metrics = load_metrics(fl_strategy, alpha)
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

    # ── Section A: Current selection ─────────────────────────────
    st.markdown(
        f'<div class="sec">{fl_strategy} · {model_type} · '
        f'α={alpha} — Client Results</div>',
        unsafe_allow_html=True)

    if metrics_df is None:
        st.markdown(
            f'<div class="warn">No JSON found at '
            f'<code>{json_path}</code>.<br>'
            f'Run the training script for this combination.</div>',
            unsafe_allow_html=True)
    else:
        metric_cols = [c for c in ["MAE", "RMSE", "NRMSE"]
                       if c in metrics_df.columns]

        c1, c2, c3, c4 = st.columns(4)
        n_clients = (metrics_df["client"].nunique()
                     if "client" in metrics_df.columns else "—")
        for col, (lbl, val) in zip([c1, c2, c3, c4], [
            ("Clients",   str(n_clients)),
            ("Avg MAE",   f'{metrics_df["MAE"].mean():.4f}'
                          if "MAE"   in metrics_df.columns else "—"),
            ("Avg RMSE",  f'{metrics_df["RMSE"].mean():.4f}'
                          if "RMSE"  in metrics_df.columns else "—"),
            ("Avg NRMSE", f'{metrics_df["NRMSE"].mean():.4f}'
                          if "NRMSE" in metrics_df.columns else "—"),
        ]):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if metric_cols:
            fig = make_subplots(rows=1, cols=len(metric_cols),
                                subplot_titles=metric_cols)
            palette = ["#58a6ff","#3fb950","#d2a8ff","#ffa657","#f85149","#79c0ff"]
            for i, metric in enumerate(metric_cols, 1):
                for j, row in metrics_df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[str(row["client"])], y=[row[metric]],
                        name=str(row["client"]), showlegend=(i == 1),
                        marker_color=palette[j % len(palette)]),
                        row=1, col=i)
            fig.update_layout(
                barmode="group", height=360,
                title_text=f"{fl_strategy} · α={alpha} — per-client metrics",
                **PL)
            for ax in list(fig.layout):
                if ax.startswith(("xaxis", "yaxis")):
                    fig.layout[ax].update(gridcolor="#21262d",
                                           linecolor="#30363d")
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="sec" style="margin-top:1.2rem;">'
                    'Raw JSON Data</div>', unsafe_allow_html=True)
        st.dataframe(metrics_df, width="stretch")

    # ── Section B: Alpha Sensitivity — ALL available alphas ───────
    st.markdown(
        f'<div class="sec" style="margin-top:2rem;">'
        f'Alpha Sensitivity — {fl_strategy} (all α values)</div>',
        unsafe_allow_html=True)

    all_alphas = _detect_alphas(fl_strategy)   # ← dynamic, not hardcoded
    all_alpha_rows = []
    for a in all_alphas:
        m = load_metrics(fl_strategy, a)
        if m:
            for client_key, metrics_val in m.items():
                all_alpha_rows.append(
                    {"client": client_key, "alpha": a, **metrics_val})

    if all_alpha_rows:
        alpha_df   = pd.DataFrame(all_alpha_rows)
        metric_opt = [c for c in ["MAE", "RMSE", "NRMSE"]
                      if c in alpha_df.columns]
        if metric_opt:
            sel_metric = st.selectbox("Metric", metric_opt,
                                       key="alpha_metric")

            # Bar chart — all alphas, selected alpha at full opacity
            fig_a = go.Figure()
            for a_val in all_alphas:
                sub     = alpha_df[alpha_df["alpha"] == a_val]
                opacity = 1.0 if a_val == alpha else 0.3
                fig_a.add_trace(go.Bar(
                    x=sub["client"].astype(str),
                    y=sub[sel_metric],
                    name=f"α={a_val}",
                    marker=dict(
                        color=STRAT_CLR.get(fl_strategy, "#58a6ff"),
                        opacity=opacity),
                ))
            fig_a.update_layout(
                barmode="group", height=380,
                title_text=(f"{fl_strategy} — {sel_metric} across all α "
                            f"(α={alpha} highlighted)"),
                **PL)
            st.plotly_chart(fig_a, width="stretch")

            # Line chart — mean per alpha across clients
            st.markdown("**Average across clients — trend by α**")
            mean_per_alpha = (alpha_df.groupby("alpha")[sel_metric]
                              .mean().reset_index())
            fig_line = px.line(
                mean_per_alpha, x="alpha", y=sel_metric,
                markers=True,
                title=f"{fl_strategy} — avg {sel_metric} vs α",
                labels={"alpha": "Alpha (α)", sel_metric: sel_metric},
                color_discrete_sequence=[STRAT_CLR.get(fl_strategy,"#58a6ff")],
            )
            fig_line.add_vline(
                x=alpha, line_dash="dash",
                line_color="#d29922",
                annotation_text=f"selected α={alpha}",
                annotation_position="top right")
            fig_line.update_traces(line_width=2.5, marker_size=9)
            fig_line.update_layout(height=320, **PL)
            st.plotly_chart(fig_line, width="stretch")

            # Heatmap
            pivot = alpha_df.pivot_table(index="client", columns="alpha",
                                          values=sel_metric)
            fig_h = px.imshow(pivot, text_auto=".4f",
                               color_continuous_scale="Blues",
                               title=f"{sel_metric} heatmap — {fl_strategy}")
            fig_h.update_layout(**PL, height=320)
            st.plotly_chart(fig_h, width="stretch")
    else:
        st.markdown(
            f'<div class="warn">No alpha data found for {fl_strategy}. '
            f'Expected JSONs in '
            f'<code>results/{fl_strategy.lower()}_results/</code>.</div>',
            unsafe_allow_html=True)

    # ── Section C: Per-model CSV ──────────────────────────────────
    st.markdown(
        f'<div class="sec" style="margin-top:2rem;">'
        f'{model_type} Model — Results</div>',
        unsafe_allow_html=True)

    model_csv_df      = load_model_results_csv(model_type)
    model_csv_path_str = (model_results_csv_path(model_type) or
                          f"results/Models_Results/{model_type.upper()}"
                          f"_MODEL_RESULTS.CSV")

    _SKIP = {"split","alpha","client","model","algorithm",
             "strategy","method","arch","round"}

    if model_csv_df is not None:
        priority = [c for c in model_csv_df.columns
                    if c.upper() in ("MAE","RMSE","NRMSE")]
        num_m    = priority or [
            c for c in model_csv_df.select_dtypes(
                include=[np.number]).columns
            if c.lower() not in _SKIP]

        col_l, col_r = st.columns([5, 5])
        with col_l:
            st.markdown(
                f'<div class="info">📂 <code>{model_csv_path_str}</code>'
                f'</div>', unsafe_allow_html=True)
            st.dataframe(model_csv_df, width="stretch")
        with col_r:
            if num_m:
                sel_m = st.selectbox("Metric", num_m, key="model_metric")
                alpha_col_m = next(
                    (c for c in model_csv_df.columns
                     if any(k in c.lower() for k in ["alpha","split"])), None)
                algo_col_m  = next(
                    (c for c in model_csv_df.columns
                     if any(k in c.lower() for k in
                            ["algorithm","strategy","method"])), None)

                if (alpha_col_m and
                        model_csv_df[alpha_col_m].nunique() > 1):
                    agg = (model_csv_df
                           .groupby([alpha_col_m] +
                                    ([algo_col_m] if algo_col_m else []))[sel_m]
                           .mean().reset_index())
                    fig_m = px.line(
                        agg.sort_values(alpha_col_m),
                        x=alpha_col_m, y=sel_m,
                        color=algo_col_m,
                        color_discrete_map=STRAT_CLR,
                        markers=True,
                        title=f"{model_type} — {sel_m} vs α",
                        labels={alpha_col_m: "Alpha (α)", sel_m: sel_m})
                    fig_m.update_traces(line_width=2.5, marker_size=9)
                else:
                    x_col_m = next(
                        (c for c in model_csv_df.columns
                         if c.lower() not in _SKIP
                         and model_csv_df[c].dtype == object), None)
                    fig_m = px.bar(
                        model_csv_df, x=sel_m,
                        y=x_col_m or model_csv_df.index.astype(str),
                        orientation="h",
                        title=f"{model_type} — {sel_m}",
                        text_auto=".4f")
                    fig_m.update_traces(
                        marker_color=STRAT_CLR.get(fl_strategy,"#58a6ff"),
                        textposition="outside")
                fig_m.update_layout(height=360, **PL)
                st.plotly_chart(fig_m, width="stretch")
    else:
        st.markdown(
            f'<div class="warn">Not found: '
            f'<code>{model_csv_path_str}</code></div>',
            unsafe_allow_html=True)

    # ── Section D: Cross-strategy comparison ─────────────────────
    st.markdown(
        '<div class="sec" style="margin-top:2rem;">'
        'Cross-Strategy Comparison</div>',
        unsafe_allow_html=True)

    summary_df = load_summary_csv()
    if summary_df is None:
        st.markdown(
            f'<div class="warn">Summary CSV not found. Tried: '
            f'<code>{summary_csv_path()}</code></div>',
            unsafe_allow_html=True)
    else:
        col_map_s    = {c.lower(): c for c in summary_df.columns}
        model_col_s  = next((col_map_s[k] for k in col_map_s
                             if any(kw in k for kw in ["model","arch"])), None)
        algo_col_s   = next((col_map_s[k] for k in col_map_s
                             if any(kw in k for kw in
                                    ["algorithm","strategy","method"])), None)
        client_col_s = next((col_map_s[k] for k in col_map_s
                             if "client" in k), None)
        alpha_col_s  = next((col_map_s[k] for k in col_map_s
                             if any(kw in k for kw in ["alpha","split"])), None)
        mc_s = [c for c in summary_df.columns
                if c.upper() in ("MAE","RMSE","NRMSE")] or [
            c for c in summary_df.select_dtypes(
                include=[np.number]).columns
            if c.lower() not in _SKIP]

        plot_df = summary_df.copy()
        if model_col_s:
            filt_m = plot_df[
                plot_df[model_col_s].str.upper() == model_type.upper()]
            if not filt_m.empty:
                plot_df = filt_m

        st.markdown(
            f'<div class="info">📂 <code>{summary_csv_path()}</code> · '
            f'model=<b>{model_type}</b> filter · {len(plot_df)} rows'
            f'</div>', unsafe_allow_html=True)

        if mc_s:
            sel_metric_s = st.selectbox("Metric to compare", mc_s,
                                         key="cmp_metric")
            ch1, ch2 = st.columns(2)

            with ch1:
                st.markdown("**Performance across α values**")
                if alpha_col_s and algo_col_s:
                    agg1 = (plot_df
                            .groupby([alpha_col_s, algo_col_s])[sel_metric_s]
                            .mean().reset_index())
                    fig1 = px.line(
                        agg1.sort_values(alpha_col_s),
                        x=alpha_col_s, y=sel_metric_s,
                        color=algo_col_s,
                        color_discrete_map=STRAT_CLR,
                        markers=True,
                        title=f"{sel_metric_s} vs α — all strategies",
                        labels={alpha_col_s: "Alpha (α)",
                                sel_metric_s: sel_metric_s})
                    fig1.update_traces(line_width=2.5, marker_size=9)
                    fig1.update_layout(height=340, **PL)
                    st.plotly_chart(fig1, width="stretch")
                else:
                    st.dataframe(plot_df, width="stretch")

            with ch2:
                st.markdown(f"**Per-client breakdown — α={alpha}**")
                sub = plot_df.copy()
                if alpha_col_s:
                    fa = sub[sub[alpha_col_s].astype(float) == float(alpha)]
                    if not fa.empty:
                        sub = fa
                if client_col_s and algo_col_s and len(sub) > 0:
                    fig2 = px.bar(
                        sub, x=client_col_s, y=sel_metric_s,
                        color=algo_col_s, barmode="group",
                        color_discrete_map=STRAT_CLR,
                        title=f"{sel_metric_s} per client — α={alpha}",
                        text_auto=".4f")
                    fig2.update_traces(textposition="outside",
                                        textfont_size=9)
                    fig2.update_layout(height=340, **PL)
                    st.plotly_chart(fig2, width="stretch")

        with st.expander("📋 Full data table"):
            st.dataframe(plot_df, width="stretch")

# ──────────────────────────────────────────────────────────────────
# TAB 2 — Predicted vs Actual  (all plots from plots/ folder)
# ──────────────────────────────────────────────────────────────────
with tab2:

    def show_plot(path, caption=""):
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.markdown(
                f'<div class="warn">⚠️ Not found: '
                f'<code>{path}</code></div>',
                unsafe_allow_html=True)

    def p(name):
        return os.path.join("plots", name)

    panel1, panel2, panel3 = st.tabs([
        "🏆  Final Results",
        "🤖  Model Comparison",
        "⚔️  Strategy Comparison",
    ])

    # ══════════════════════════════════════════
    # PANEL 1 — FINAL RESULTS
    # ══════════════════════════════════════════
    with panel1:
        st.markdown(
            '<div class="sec">Final Results — All Models & Strategies</div>',
            unsafe_allow_html=True)

        # Model metric cards
        cols = st.columns(5)
        for col, (name, vals) in zip(cols, MODEL_DATA.items()):
            border = "#3fb950" if name == model_type else "#30363d"
            col.markdown(f"""
            <div class="metric-card"
                 style="border-color:{border};border-width:2px;">
              <div class="metric-label"
                   style="color:{vals['color']};">
                   {name}{"  ✓" if name == model_type else ""}</div>
              <div class="metric-value"
                   style="color:{vals['color']};font-size:1.2rem;">
                   {vals['MAE']:.4f}</div>
              <div class="metric-sub">MAE</div>
              <div style="font-family:Space Mono,monospace;font-size:.85rem;
                          color:{vals['color']};margin-top:.3rem;">
                   {vals['NRMSE']:.4f}</div>
              <div class="metric-sub">NRMSE</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # four_algo_final_comparison
        st.markdown("**All Four Algorithms — Final Comparison**")
        show_plot(p("four_algo_final_comparison.png"),
                  "Final comparison across all FL algorithms")

        st.markdown("<br>", unsafe_allow_html=True)

        # models_bar_graph
        st.markdown("**Model Performance — MAE & NRMSE**")
        show_plot(p("models_bar_graph.png"),
                  "Performance Comparison — All Deep Models")

        st.markdown("<br>", unsafe_allow_html=True)

        # four_algo_split_vs_rmse
        st.markdown("**RMSE across α splits — All Algorithms**")
        show_plot(p("four_algo_split_vs_rmse.png"),
                  "All algorithms RMSE vs Dirichlet α")

        st.markdown("<br>", unsafe_allow_html=True)

        # bargraph_algo_comparison
        st.markdown("**Algorithm Comparison Bar Chart**")
        show_plot(p("bargraph_algo_comparison.png"),
                  "FL Algorithm comparison")

        st.markdown("<br>", unsafe_allow_html=True)

        # Selected model prediction
        st.markdown(
            f'<div class="sec">Selected Model ({model_type}) — '
            f'Predicted vs Actual</div>',
            unsafe_allow_html=True)
        show_plot(p(f"{model_type.lower()}_vs_actual.png"),
                  f"{model_type} — Predicted vs Actual")

        st.markdown("""
        <div class="info" style="margin-top:1rem;">
          <b>Key Findings:</b> GRU achieves best MAE (0.0135).
          LSTM achieves best NRMSE (0.0286). MLP performs worst,
          confirming temporal modelling is essential for traffic prediction.
          FedAvg outperforms FedProx and FedNova on this dataset.
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # PANEL 2 — MODEL COMPARISON
    # ══════════════════════════════════════════
    with panel2:
        st.markdown(
            '<div class="sec">Predicted vs Actual — All Models</div>',
            unsafe_allow_html=True)

        all_models = ["GRU", "LSTM", "CNN", "RNN", "MLP"]
        highlight  = st.radio(
            "Highlight model", all_models,
            index=all_models.index(model_type)
                  if model_type in all_models else 0,
            horizontal=True, key="model_highlight_radio")

        # Full width — highlighted model
        st.markdown(f"**{highlight} — Full View**")
        show_plot(p(f"{highlight.lower()}_vs_actual.png"),
                  f"{highlight}: MAE={MODEL_DATA[highlight]['MAE']:.4f}  "
                  f"NRMSE={MODEL_DATA[highlight]['NRMSE']:.4f}")

        # LSTM before FL training
        if os.path.exists(p("prediction_lstm_before.png")):
            st.markdown(
                '<div class="sec" style="margin-top:1.2rem;">'
                'LSTM — Before Federated Training</div>',
                unsafe_allow_html=True)
            st.markdown("""
            <div class="info">
              Predictions <b>before</b> FL training — shows how much
              federated learning improves accuracy.
            </div>""", unsafe_allow_html=True)
            show_plot(p("prediction_lstm_before.png"),
                      "LSTM before federated training")

        # Side by side grid
        st.markdown(
            '<div class="sec" style="margin-top:1.2rem;">'
            'All Models — Side by Side</div>',
            unsafe_allow_html=True)

        for row_models in [["GRU", "LSTM", "CNN"], ["RNN", "MLP"]]:
            cols = st.columns(len(row_models))
            for col, name in zip(cols, row_models):
                with col:
                    border = (f"3px solid {MODEL_DATA[name]['color']}"
                              if name == highlight
                              else "1px solid #30363d")
                    st.markdown(
                        f'<div style="border:{border};border-radius:8px;'
                        f'padding:4px;margin-bottom:6px;">'
                        f'<p style="color:{MODEL_DATA[name]["color"]};'
                        f'font-weight:700;margin:4px 0 2px 6px;">'
                        f'{name} · MAE {MODEL_DATA[name]["MAE"]:.4f} · '
                        f'NRMSE {MODEL_DATA[name]["NRMSE"]:.4f}</p>'
                        f'</div>', unsafe_allow_html=True)
                    show_plot(p(f"{name.lower()}_vs_actual.png"), "")

    # ══════════════════════════════════════════
    # PANEL 3 — STRATEGY COMPARISON
    # ══════════════════════════════════════════
    with panel3:
        st.markdown(
            '<div class="sec">FL Strategy Comparison</div>',
            unsafe_allow_html=True)

        st.markdown("""
        <div class="info">
          RMSE at Dirichlet α = 0.1, 0.5, 1.0. Lower α = harder data.
          <b>FedAvg consistently achieves the lowest RMSE</b>
          across all alpha values on this dataset.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # All four algorithms
        st.markdown("**All Four Algorithms — RMSE vs α**")
        show_plot(p("four_algo_split_vs_rmse.png"),
                  "RMSE across all FL strategies and alpha values")

        st.markdown("<br>", unsafe_allow_html=True)

        # Three strategies
        st.markdown("**FedAvg vs FedProx vs FedNova**")
        show_plot(p("rmse_fedavg_fednova_fedprox.png"),
                  "RMSE: FedAvg vs FedProx vs FedNova")

        st.markdown(
            '<div class="sec" style="margin-top:1.5rem;">'
            'Pairwise Comparisons</div>',
            unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            show_plot(p("rmse_fedavg_fedprox.png"),
                      "FedAvg vs FedProx — RMSE")
        with col_r:
            show_plot(p("rmse_fedavg_fednova.png"),
                      "FedAvg vs FedNova — RMSE")

        st.markdown(
            '<div class="sec" style="margin-top:1.5rem;">'
            'Training Convergence — Loss per Round</div>',
            unsafe_allow_html=True)
        st.markdown("""
        <div class="info">
          All strategies converge within 5 rounds and plateau near round 10.
        </div>""", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            show_plot(p("fedavg_median_comp.png"),
                      "FedAvg vs MedianAvg")
        with col_b:
            show_plot(p("simple_fedavg_comp.png"),
                      "FedAvg vs SimpleAvg")
        with col_c:
            show_plot(p("fedavg_median_simple.png"),
                      "FedAvg vs SimpleAvg vs MedianAvg")

        # Strategy ranking cards
        st.markdown(
            '<div class="sec" style="margin-top:1.5rem;">'
            'Strategy Ranking</div>',
            unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        for col, label, color, rmse, note in [
            (r1, "🥇 FedAvg",  "#58a6ff", "RMSE ≈ 0.036–0.045",
             "Best overall. Simple weighted averaging is most effective."),
            (r2, "🥈 FedProx", "#ffa657", "RMSE ≈ 0.056–0.100",
             "μ tuning required. Helps at α=1.0, hurts at α=0.5."),
            (r3, "🥉 FedNova", "#d2a8ff", "RMSE ≈ 0.179–0.184",
             "Normalisation overcorrects when client sizes are similar."),
        ]:
            col.markdown(f"""
            <div class="metric-card"
                 style="border-color:{color};border-width:2px;
                        text-align:left;padding:1rem;">
              <div style="font-size:1rem;font-weight:700;
                          color:{color};margin-bottom:.4rem;">{label}</div>
              <div style="font-family:Space Mono,monospace;
                          font-size:.8rem;color:{color};
                          margin-bottom:.5rem;">{rmse}</div>
              <div style="font-size:.82rem;color:#c9d1d9;">{note}</div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 3 — Live Inference
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec">Live Inference</div>',
                unsafe_allow_html=True)

    def run_inference(X_np: np.ndarray) -> float:
        X_scaled = scale_input(X_np, scaling_data)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(X_t)
        return out.squeeze().item()

    if model is None:
        st.markdown(
            f'<div class="warn">⚠️ No model loaded<br>'
            f'<small style="color:#6e7681">'
            f'Metrics &amp; Plots tabs work normally.<br>'
            f'To enable Live Inference, save a trained model to:<br>'
            f'<code>results/{fl_strategy.lower()}_results/'
            f'model_alpha_{alpha}.pth</code></small></div>',
            unsafe_allow_html=True)
    else:
        raw_check  = torch.load(loaded_path, map_location="cpu",
                                weights_only=False)
        ckpt_check = _unwrap_checkpoint(raw_check)
        ckpt_check = _strip_prefix(ckpt_check)
        n_features = _detect_input_size(ckpt_check, detected_arch)
        feat_names = [f"feature_{i+1}" for i in range(n_features)]

        arch_note2 = (f" (sidebar: {model_type})"
                      if detected_arch != model_type else "")
        st.markdown(
            f'<div class="info">Arch: <b>{detected_arch}</b>{arch_note2} · '
            f'{fl_strategy} · α={alpha} · {n_features} input feature(s)<br>'
            f'<small style="color:#6e7681">Weights: {loaded_path}</small>'
            f'</div>', unsafe_allow_html=True)

        input_mode = st.radio("Input method",
                               ["Manual entry", "Upload CSV"],
                               horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if input_mode == "Manual entry":
            st.markdown(
                f'<div class="info"><b>{SEQUENCE_LEN} timesteps × '
                f'{n_features} feature(s)</b>. '
                f'Each row = one time step.</div>',
                unsafe_allow_html=True)
            default_df = pd.DataFrame(
                np.zeros((SEQUENCE_LEN, n_features)), columns=feat_names)
            edited = st.data_editor(default_df, width="stretch",
                                     num_rows="fixed")

            if st.button("▶  Run Prediction"):
                try:
                    pred = run_inference(edited.values.astype(np.float32))
                    st.markdown(f"""
                    <div class="pred-box">
                      <div class="metric-label">Predicted Traffic Volume</div>
                      <div style='font-family:Space Mono,monospace;
                                  font-size:3rem;color:#3fb950;
                                  font-weight:700;'>{pred:.4f}</div>
                      <div style='color:#6e7681;font-size:.8rem;
                                  margin-top:.4rem;'>
                        {model_type} · {fl_strategy} · α={alpha}
                      </div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.markdown(
                f'<div class="info">Upload CSV with ≥ <b>{n_features}</b> '
                f'numeric columns. Last <b>{SEQUENCE_LEN}</b> rows = one '
                f'window.</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV", type=["csv"])

            if uploaded:
                try:
                    df_up    = pd.read_csv(uploaded)
                    num_cols = df_up.select_dtypes(
                        include=[np.number]).columns.tolist()
                    if len(num_cols) < n_features:
                        st.error(f"Need ≥ {n_features} numeric columns, "
                                 f"found {len(num_cols)}.")
                    else:
                        feat_data = df_up[num_cols[:n_features]].values.astype(
                            np.float32)
                        st.caption(f"{len(df_up)} rows × {n_features} features")
                        st.dataframe(
                            df_up[num_cols[:n_features]].tail(SEQUENCE_LEN),
                            width="stretch")

                        ca, cb = st.columns(2)
                        with ca:
                            if st.button(
                                    f"▶  Predict (last {SEQUENCE_LEN} rows)"):
                                pred = run_inference(feat_data[-SEQUENCE_LEN:])
                                st.markdown(f"""
                                <div class="pred-box">
                                  <div class="metric-label">
                                    Predicted Traffic Volume</div>
                                  <div style='font-family:Space Mono,monospace;
                                              font-size:2.5rem;color:#3fb950;
                                              font-weight:700;'>{pred:.4f}</div>
                                  <div style='color:#6e7681;font-size:.8rem;
                                              margin-top:.4rem;'>
                                    {model_type} · {fl_strategy} · α={alpha}
                                  </div>
                                </div>""", unsafe_allow_html=True)
                        with cb:
                            if st.button("⚡  Batch (sliding window)"):
                                if len(feat_data) < SEQUENCE_LEN:
                                    st.warning(
                                        f"Need ≥ {SEQUENCE_LEN} rows.")
                                else:
                                    preds_b = [
                                        run_inference(
                                            feat_data[i-SEQUENCE_LEN:i])
                                        for i in range(SEQUENCE_LEN,
                                                       len(feat_data)+1)]
                                    fig_b = go.Figure()
                                    fig_b.add_trace(go.Scatter(
                                        y=preds_b, mode="lines",
                                        line=dict(color="#58a6ff", width=1.5),
                                        name="Predicted"))
                                    fig_b.update_layout(
                                        height=300,
                                        title="Batch Predictions",
                                        xaxis_title="Window",
                                        yaxis_title="Traffic", **PL)
                                    st.plotly_chart(fig_b, width="stretch")
                                    dl = pd.DataFrame({"prediction": preds_b})
                                    st.download_button(
                                        "⬇ Download CSV",
                                        dl.to_csv(index=False),
                                        "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")