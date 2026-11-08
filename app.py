"""
Loan Risk Desk - evaluation console for five saved credit-default classifiers.
Data  : Statlog (German Credit Data), UCI Machine Learning Repository.
Note  : models are only loaded from /model and used for scoring; no re-training happens here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

HERE = Path(__file__).parent
LABEL_COL = "default"

REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
INK = "#8a8f98"          # neutral grey that stays readable in light and dark mode
ACCENT = "#6d5dfc"
WARM = "#ff9f43"
METRIC_HINTS = {
    "Accuracy": "share of all applicants classified correctly",
    "AUC": "how well risky applicants are ranked above safe ones",
    "Precision": "of those flagged as default, how many truly defaulted",
    "Recall": "of the real defaulters, how many were caught",
    "F1": "balance between precision and recall",
    "MCC": "overall correlation, robust to class imbalance",
}

st.set_page_config(page_title="Loan Risk Desk", page_icon="🏦", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{display:none;}}
    .block-container {{max-width: 1180px; padding-top: 1.2rem;}}
    .masthead {{border-left: 7px solid {ACCENT}; padding: .2rem 0 .2rem 1rem; margin-bottom: .6rem;}}
    .masthead h1 {{font-size: 2rem; margin: 0; letter-spacing: -.5px;}}
    .masthead span {{opacity: .65; font-size: .95rem;}}
    .tile {{border-radius: 16px; padding: .85rem 1rem; background: rgba(109,93,252,.10);
           border: 1px solid rgba(109,93,252,.28); height: 100%;}}
    .tile small {{text-transform: uppercase; letter-spacing: 1.2px; opacity: .65; font-size: .68rem;}}
    .tile b {{display:block; font-size: 1.75rem; line-height: 1.25;}}
    .tile em {{font-size: .72rem; opacity: .6; font-style: normal;}}
    .verdict {{border-radius: 12px; padding: .8rem 1.1rem; margin: .9rem 0;
              background: rgba(255,159,67,.12); border: 1px dashed rgba(255,159,67,.6);}}
    </style>
    <div class="masthead">
      <h1>🏦 Loan Risk Desk</h1>
      <span>Score loan applicants with five saved classifiers - upload a test file, choose a model, read the results.</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ utilities
@st.cache_resource
def fetch_model(fname):
    return joblib.load(HERE / "model" / fname)


@st.cache_data
def bundled_frame():
    return pd.read_csv(HERE / "test_data.csv")


def metric_bundle(truth, flag, prob):
    return {
        "Accuracy": accuracy_score(truth, flag),
        "AUC": roc_auc_score(truth, prob),
        "Precision": precision_score(truth, flag, zero_division=0),
        "Recall": recall_score(truth, flag, zero_division=0),
        "F1": f1_score(truth, flag, zero_division=0),
        "MCC": matthews_corrcoef(truth, flag),
    }


def decide(mdl, X, p, thr):
    """At the default 0.50 setting use the model's own predict(); otherwise cut the probabilities."""
    if abs(thr - 0.50) < 1e-9:
        return mdl.predict(X).astype(int)
    return (p >= thr).astype(int)


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_fig(w, h, **kw):
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    fig.patch.set_alpha(0)
    if not isinstance(ax, np.ndarray):
        ax.set_facecolor("none")
    return fig, ax


# ------------------------------------------------------------------ control bar
c_up, c_model, c_dl = st.columns([2.2, 1.6, 1])
with c_up:
    upload = st.file_uploader("Test data (CSV)", type="csv", label_visibility="collapsed",
                              help="Needs the 20 input columns and the 'default' column.")
with c_model:
    picked = st.selectbox("Model", list(REGISTRY), label_visibility="collapsed")
with c_dl:
    st.download_button("⬇ Sample CSV", (HERE / "test_data.csv").read_bytes(),
                       "test_data.csv", "text/csv", width="stretch")

with st.expander("Decision threshold (0.50 = each model's standard prediction)"):
    cutoff = st.slider("Flag an applicant as 'default' when probability is at least",
                       0.10, 0.90, 0.50, 0.05)

frame = pd.read_csv(upload) if upload is not None else bundled_frame()
origin = upload.name if upload is not None else "test_data.csv (bundled)"

model = fetch_model(REGISTRY[picked])
inputs = list(model.named_steps["prep"].feature_names_in_)
absent = [c for c in inputs + [LABEL_COL] if c not in frame.columns]
if absent:
    st.error("Your file lacks these columns: " + ", ".join(absent))
    st.stop()

features = frame[inputs]
truth = frame[LABEL_COL].astype(int)
if truth.nunique() < 2:
    st.error("The 'default' column must contain both 0 and 1 so AUC can be computed.")
    st.stop()

prob = model.predict_proba(features)[:, 1]
flag = decide(model, features, prob, cutoff)
scores = metric_bundle(truth, flag, prob)

st.caption(f"Source: **{origin}**  |  {len(frame)} applicants  |  {len(inputs)} features  |  "
           f"model: **{picked}**  |  threshold {cutoff:.2f}")

view = st.segmented_control("View", ["Scorecard", "Leaderboard", "Applicant lens"],
                            default="Scorecard", label_visibility="collapsed") or "Scorecard"

# ------------------------------------------------------------------ view 1
if view == "Scorecard":
    tiles = st.columns(6)
    for holder, (name, val) in zip(tiles, scores.items()):
        holder.markdown(f'<div class="tile"><small>{name}</small><b>{val:.3f}</b>'
                        f'<em>{METRIC_HINTS[name]}</em></div>', unsafe_allow_html=True)

    cm = confusion_matrix(truth, flag, labels=[0, 1])
    caught, missed = cm[1, 1], cm[1, 0]
    st.markdown(
        f'<div class="verdict">At this threshold the model catches <b>{caught}</b> of '
        f'<b>{caught + missed}</b> real defaulters and raises <b>{cm[0, 1]}</b> false alarms '
        f'among <b>{cm[0].sum()}</b> good applicants.</div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1.1, 1, 1])
    with left:
        st.markdown("##### Confusion matrix")
        fig, ax = new_fig(4, 3.6)
        pct = cm / cm.sum(axis=1, keepdims=True)
        cmap = LinearSegmentedColormap.from_list("indigo", ["#f4f2ff", ACCENT])
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}\n{pct[r, c]:.0%}", ha="center", va="center",
                        fontsize=13, color="white" if pct[r, c] > .55 else "#222")
        ax.set_xticks([0, 1], ["Predicted\nrepaid", "Predicted\ndefault"])
        ax.set_yticks([0, 1], ["Actual\nrepaid", "Actual\ndefault"])
        style_axes(ax)
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    with mid:
        st.markdown("##### ROC curve")
        fpr, tpr, _ = roc_curve(truth, prob)
        fig, ax = new_fig(4, 3.6)
        ax.fill_between(fpr, tpr, alpha=.18, color=ACCENT)
        ax.plot(fpr, tpr, color=ACCENT, lw=2.4)
        ax.plot([0, 1], [0, 1], ls=":", color=INK)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.text(.98, .06, f"AUC {scores['AUC']:.3f}", ha="right", color=INK, fontsize=11)
        style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("##### Score spread")
        fig, ax = new_fig(4, 3.6)
        bins = np.linspace(0, 1, 21)
        ax.hist(prob[truth == 0], bins=bins, alpha=.75, color=ACCENT, label="repaid")
        ax.hist(prob[truth == 1], bins=bins, alpha=.75, color=WARM, label="defaulted")
        ax.axvline(cutoff, color=INK, ls="--", lw=1.2)
        ax.set_xlabel("Predicted default probability")
        ax.set_ylabel("Applicants")
        ax.legend(frameon=False, labelcolor=INK)
        style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("##### Classification report")
    rep = pd.DataFrame(classification_report(
        truth, flag, target_names=["repaid (0)", "default (1)"], output_dict=True,
        zero_division=0)).T
    rep["support"] = rep["support"].astype(int)
    st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"})
                 .background_gradient(subset=["precision", "recall", "f1-score"], cmap="Purples", vmin=0, vmax=1),
                 width="stretch")

# ------------------------------------------------------------------ view 2
elif view == "Leaderboard":
    board = {}
    for nm, fname in REGISTRY.items():
        m = fetch_model(fname)
        p = m.predict_proba(features)[:, 1]
        board[nm] = metric_bundle(truth, decide(m, features, p, cutoff), p)
    table = pd.DataFrame(board).T
    table.index.name = "ML Model Name"

    rank_by = st.radio("Rank models by", list(table.columns), horizontal=True, index=5)
    ordered = table.sort_values(rank_by, ascending=False)
    ordered.insert(0, "Rank", range(1, len(ordered) + 1))
    st.dataframe(ordered.style.format({c: "{:.4f}" for c in table.columns})
                 .background_gradient(subset=list(table.columns), cmap="Purples", axis=0),
                 width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown(f"##### {rank_by} by model")
        fig, ax = new_fig(5.2, 3.4)
        srt = table[rank_by].sort_values()
        bars = ax.barh(srt.index, srt.values,
                       color=[WARM if n == srt.index[-1] else ACCENT for n in srt.index])
        for bar, v in zip(bars, srt.values):
            ax.text(v + .01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", color=INK)
        ax.set_xlim(0, max(1, srt.max() + .1))
        style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)
    with b:
        st.markdown("##### Shape of each model")
        axes_names = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
        ang = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
        ang += ang[:1]
        fig = plt.figure(figsize=(5.2, 3.8))
        fig.patch.set_alpha(0)
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor("none")
        for nm in table.index:
            vals = table.loc[nm, axes_names].tolist()
            vals += vals[:1]
            ax.plot(ang, vals, lw=2.4 if nm == picked else 1.1, label=nm)
        ax.set_xticks(ang[:-1], axes_names, color=INK)
        ax.set_ylim(0, 1)
        ax.tick_params(colors=INK, labelsize=7)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False, fontsize=8, labelcolor=INK)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------ view 3
else:
    idx = st.slider("Applicant row", 0, len(frame) - 1, 0)
    row = frame.iloc[idx]
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("##### Applicant profile")
        profile = pd.DataFrame({"value": row[inputs].astype(str)})
        st.dataframe(profile, width="stretch", height=430)
    with right:
        st.markdown("##### Default probability by model")
        single = features.iloc[[idx]]
        probs = {nm: float(fetch_model(f).predict_proba(single)[0, 1]) for nm, f in REGISTRY.items()}
        fig, ax = new_fig(5.4, 3.3)
        names = list(probs)
        vals = [probs[n] for n in names]
        ax.barh(names, vals, color=[WARM if v >= cutoff else ACCENT for v in vals])
        ax.axvline(cutoff, color=INK, ls="--", lw=1.2)
        for i, v in enumerate(vals):
            ax.text(min(v + .02, .9), i, f"{v:.0%}", va="center", color=INK)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)
        actual = "DEFAULTED" if int(row[LABEL_COL]) == 1 else "REPAID"
        votes = sum(v >= cutoff for v in vals)
        st.markdown(f'<div class="verdict">Recorded outcome: <b>{actual}</b>. '
                    f'{votes} of 5 models flag this applicant as a default risk.</div>',
                    unsafe_allow_html=True)

    st.markdown("##### All predictions for the selected model")
    out = frame.copy()
    out["predicted_default"] = flag
    out["default_probability"] = prob.round(3)
    only_wrong = st.toggle("Only misclassified rows")
    shown = out[out["predicted_default"] != out[LABEL_COL]] if only_wrong else out
    st.dataframe(shown, width="stretch", height=300)
    st.download_button("Download these predictions", shown.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
