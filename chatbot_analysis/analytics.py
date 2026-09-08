"""
analytics.py
============
End-to-end analytics pipeline for the Airport Multimodal Chatbot logs.

Covers every requirement of the BSBI 'Chatbot Analytics and Optimization'
assignment brief:

  Part 2 (Data Analytics)
    * Exploratory Data Analysis (statistical descriptors)
    * User segmentation & personalization      (Area 1)
    * Fallback rate + NLP-based fallback code  (Area 2)
    * Intent recognition accuracy + confusion matrix  (Area 3)
    * Intent co-occurrence matrix + Support/Confidence/Lift  (Area 4)
    * Response time, time-to-first-response, completion rate

  Part 3 (Metric Analysis)
    * LTV proxy by persona
    * Fallback strategy diagnostics (what's failing most)
    * CSAT + frustration analysis
    * Frustration-scenario extraction
    * Feeds the dashboard.py script

All charts saved as PNG (300 dpi), all tables as CSV.
"""
from __future__ import annotations
import json, re, math
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score)

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "outputs" / "logs"
OUT  = ROOT / "outputs" / "analytics"
FIG  = OUT / "figures"
TAB  = OUT / "tables"
for p in (OUT, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

# Consistent visual style for the report
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
turns    = pd.read_csv(LOGS / "conversation_logs.csv")
sessions = pd.read_csv(LOGS / "sessions.csv")
users    = pd.read_csv(LOGS / "users.csv")
turns["timestamp"] = pd.to_datetime(turns["timestamp"])
sessions["started_at"] = pd.to_datetime(sessions["started_at"])
print(f"Loaded {len(turns)} turns, {len(sessions)} sessions, {len(users)} users.")

# ---------------------------------------------------------------------------
# 1. EXPLORATORY DATA ANALYSIS  (Part 2 preamble)
# ---------------------------------------------------------------------------
eda = {
    "n_users": len(users),
    "n_sessions": len(sessions),
    "n_turns": len(turns),
    "avg_turns_per_session": round(len(turns)/len(sessions), 2),
    "unique_intents_observed": int(turns["true_intent"].nunique()),
    "unique_predicted_intents": int(turns["predicted_intent"].nunique()),
    "response_time_ms_mean": int(turns["response_time_ms"].mean()),
    "response_time_ms_median": int(turns["response_time_ms"].median()),
    "response_time_ms_p95": int(turns["response_time_ms"].quantile(0.95)),
    "time_to_first_response_ms_median": int(turns["time_to_first_response_ms"].median()),
    "csat_mean": round(turns["csat"].mean(), 2),
    "csat_median": int(turns["csat"].median()),
    "fallback_rate_%": round(turns["is_fallback"].mean()*100, 2),
    "intent_accuracy_%": round(turns["is_correct"].mean()*100, 2),
    "frustration_rate_%": round(turns["frustrated_signal"].mean()*100, 2),
    "session_completion_rate_%": round(sessions["completed"].mean()*100, 2),
    "session_abandon_rate_%":    round(sessions["abandoned"].mean()*100, 2),
}
pd.Series(eda, name="value").to_csv(TAB / "eda_summary.csv", header=True)
print("\n[EDA summary]")
for k, v in eda.items(): print(f"  {k:38s} {v}")

# Distribution charts (compact multi-panel EDA figure)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0,0].hist(turns["response_time_ms"], bins=40, color="#3b82f6", edgecolor="white")
axes[0,0].axvline(eda["response_time_ms_median"], color="crimson", linestyle="--",
                  label=f"median {eda['response_time_ms_median']} ms")
axes[0,0].set_title("Response time distribution")
axes[0,0].set_xlabel("ms"); axes[0,0].set_ylabel("turns"); axes[0,0].legend()

axes[0,1].bar(["1","2","3","4","5"],
              [(turns["csat"]==i).sum() for i in range(1,6)],
              color=["#dc2626","#f97316","#facc15","#84cc16","#16a34a"])
axes[0,1].set_title("CSAT rating distribution (1–5)")
axes[0,1].set_xlabel("CSAT"); axes[0,1].set_ylabel("turns")

top15 = turns["true_intent"].value_counts().head(15)
axes[1,0].barh(top15.index[::-1], top15.values[::-1], color="#0ea5e9")
axes[1,0].set_title("Top 15 requested intents")
axes[1,0].set_xlabel("turns")

mod_counts = turns["modality"].value_counts()
axes[1,1].pie(mod_counts.values, labels=mod_counts.index, autopct="%1.0f%%",
              colors=["#6366f1","#f59e0b","#10b981"], startangle=90)
axes[1,1].set_title("Modality mix")

fig.suptitle("Exploratory Data Analysis — Airport Chatbot Logs", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "01_eda_overview.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 2. USER SEGMENTATION & PERSONALIZATION  (Part 2 Area 1)
# ---------------------------------------------------------------------------
# By persona
persona_stats = (turns.groupby("persona")
    .agg(turns=("csat","size"),
         sessions=("session_id","nunique"),
         users=("user_id","nunique"),
         csat_mean=("csat","mean"),
         fallback_rate=("is_fallback","mean"),
         intent_accuracy=("is_correct","mean"),
         avg_rt_ms=("response_time_ms","mean"))
    .round(3))
persona_stats.to_csv(TAB / "segmentation_by_persona.csv")
print("\n[Segmentation — persona]"); print(persona_stats)

# By channel
channel_stats = (turns.groupby("channel")
    .agg(turns=("csat","size"),
         csat_mean=("csat","mean"),
         fallback_rate=("is_fallback","mean"),
         avg_rt_ms=("response_time_ms","mean"))
    .round(3))
channel_stats.to_csv(TAB / "segmentation_by_channel.csv")

# Persona × top intent (personalisation opportunity)
persona_intent = (turns.groupby(["persona","true_intent"]).size()
    .unstack(fill_value=0))
top_per_persona = (persona_intent
    .apply(lambda row: row.sort_values(ascending=False).head(5).index.tolist(), axis=1))
top_per_persona.to_csv(TAB / "top_intents_by_persona.csv", header=["top5_intents"])
print("\n[Top-5 intents per persona]"); print(top_per_persona)

# Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
persona_stats["csat_mean"].plot(kind="bar", ax=axes[0], color="#6366f1")
axes[0].set_title("Average CSAT by persona"); axes[0].set_ylim(0,5)
axes[0].set_ylabel("CSAT (1–5)"); axes[0].tick_params(axis="x", rotation=25)
persona_stats["fallback_rate"].mul(100).plot(kind="bar", ax=axes[1], color="#dc2626")
axes[1].set_title("Fallback rate by persona (%)"); axes[1].set_ylabel("%")
axes[1].tick_params(axis="x", rotation=25)
fig.suptitle("User Segmentation — Persona Performance", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "02_segmentation.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 3. FALLBACK RATE + NLP FALLBACK DETECTION  (Part 2 Area 2)
# ---------------------------------------------------------------------------
# The brief asks for CODE that detects fallback scenarios with NLP.
# We combine three signals: (a) confidence below threshold,
#                          (b) predicted intent == 'fallback' / 'out_of_scope',
#                          (c) linguistic fallback markers in the utterance.

FALLBACK_LINGUISTIC_PATTERNS = [
    r"\bi (don't|do not) understand\b",
    r"\bsorry[, ]+i\b",
    r"\bcan you rephrase\b",
    r"\bnot sure what you mean\b",
    r"\b(what|huh)\??$",
]
FALLBACK_REGEX = re.compile("|".join(FALLBACK_LINGUISTIC_PATTERNS), re.I)

def detect_fallback(row, threshold: float = 0.45) -> bool:
    """NLP + confidence hybrid fallback detector.

    A turn is treated as a fallback if ANY of:
      1. model confidence < threshold
      2. predicted intent explicitly says fallback / out_of_scope
      3. the bot response would contain a fallback linguistic marker
         (approximated here on the user utterance for offline analysis)
    """
    if row["confidence"] < threshold:
        return True
    if str(row["predicted_intent"]) in {"fallback", "out_of_scope"}:
        return True
    if bool(FALLBACK_REGEX.search(str(row["utterance"]))):
        return True
    return False

turns["nlp_fallback_flag"] = turns.apply(detect_fallback, axis=1)

fallback_report = {
    "confidence_based_flags": int((turns["confidence"] < 0.45).sum()),
    "predicted_intent_flags": int(turns["predicted_intent"].isin(["fallback","out_of_scope"]).sum()),
    "linguistic_marker_flags": int(turns["utterance"].str.contains(FALLBACK_REGEX).sum()),
    "combined_nlp_flags":     int(turns["nlp_fallback_flag"].sum()),
    "logged_is_fallback":     int(turns["is_fallback"].sum()),
    "overall_fallback_rate_%":round(turns["nlp_fallback_flag"].mean()*100, 2),
}
pd.Series(fallback_report, name="value").to_csv(TAB / "fallback_detection.csv", header=True)
print("\n[Fallback detection]"); print(fallback_report)

# Which intents fall back most often?
fb_by_intent = (turns.groupby("true_intent")["is_fallback"].mean()
    .sort_values(ascending=False).head(15)*100).round(1)
fb_by_intent.to_csv(TAB / "fallback_by_intent.csv", header=["fallback_rate_%"])

fig, ax = plt.subplots(figsize=(9, 5.5))
fb_by_intent[::-1].plot(kind="barh", ax=ax, color="#dc2626")
ax.set_title("Top 15 intents driving fallback (%)")
ax.set_xlabel("fallback rate (%)")
fig.tight_layout(); fig.savefig(FIG / "03_fallback_by_intent.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 4. INTENT RECOGNITION ACCURACY + CONFUSION MATRIX  (Part 2 Area 3)
# ---------------------------------------------------------------------------
# We exclude fallback predictions so the confusion matrix stays interpretable.
mask = ~turns["is_fallback"]
y_true = turns.loc[mask, "true_intent"].astype(str).values
y_pred = turns.loc[mask, "predicted_intent"].astype(str).values

# Focus the confusion matrix on the top-N intents (readability)
top_labels = list(pd.Series(y_true).value_counts().head(18).index)
mask_top = np.isin(y_true, top_labels) & np.isin(y_pred, top_labels)
cm = confusion_matrix(y_true[mask_top], y_pred[mask_top], labels=top_labels)

acc  = accuracy_score(y_true, y_pred)
f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
pd.Series({"accuracy_%": round(acc*100,2), "macro_f1": round(f1m,3),
           "weighted_f1": round(f1w,3)}, name="value").to_csv(
    TAB / "intent_accuracy_summary.csv", header=True)

report_df = pd.DataFrame(classification_report(y_true, y_pred,
    zero_division=0, output_dict=True)).T
report_df.to_csv(TAB / "intent_classification_report.csv")
print(f"\n[Intent metrics] accuracy={acc:.3f} macroF1={f1m:.3f} weightedF1={f1w:.3f}")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(top_labels))); ax.set_xticklabels(top_labels, rotation=45, ha="right")
ax.set_yticks(range(len(top_labels))); ax.set_yticklabels(top_labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Intent Confusion Matrix — Top {len(top_labels)} intents\n"
             f"accuracy={acc:.1%}  weighted-F1={f1w:.2f}")
for i in range(len(top_labels)):
    for j in range(len(top_labels)):
        v = cm[i, j]
        if v:
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > cm.max()*0.5 else "black", fontsize=8)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(FIG / "04_confusion_matrix.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 5. INTENT CO-OCCURRENCE + SUPPORT / CONFIDENCE / LIFT  (Part 2 Area 4)
# ---------------------------------------------------------------------------
# Mini Apriori on intent baskets = one basket per session.
sess_intents = (turns.groupby("session_id")["true_intent"]
    .apply(lambda s: sorted(set(s))))
N_SESSIONS = len(sess_intents)

# 1-item support
support1 = Counter()
for basket in sess_intents:
    for it in basket:
        support1[it] += 1
support1 = {k: v/N_SESSIONS for k, v in support1.items()}

# 2-item support
support2 = Counter()
for basket in sess_intents:
    for a, b in combinations(basket, 2):
        pair = tuple(sorted((a, b)))
        support2[pair] += 1
support2 = {k: v/N_SESSIONS for k, v in support2.items()}

MIN_SUPPORT = 0.02  # 2% of sessions
rules = []
for (a, b), sup in support2.items():
    if sup < MIN_SUPPORT: continue
    for ante, cons in [(a, b), (b, a)]:
        conf = sup / support1[ante] if support1[ante] else 0
        lift = conf / support1[cons] if support1[cons] else 0
        rules.append({"antecedent": ante, "consequent": cons,
                      "support": round(sup,3),
                      "confidence": round(conf,3),
                      "lift": round(lift,2)})

rules_df = (pd.DataFrame(rules)
    .sort_values(["lift","confidence"], ascending=False)
    .head(25))
rules_df.to_csv(TAB / "cooccurrence_rules.csv", index=False)
print("\n[Top co-occurrence rules by lift]"); print(rules_df.head(10).to_string(index=False))

# Heatmap of co-occurrence support for top intents
top_for_matrix = [k for k,_ in sorted(support1.items(), key=lambda x: -x[1])[:15]]
matrix = np.zeros((len(top_for_matrix), len(top_for_matrix)))
for i, a in enumerate(top_for_matrix):
    for j, b in enumerate(top_for_matrix):
        if i == j: matrix[i,j] = support1[a]
        else:
            pair = tuple(sorted((a, b)))
            matrix[i,j] = support2.get(pair, 0)

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(matrix, cmap="Purples")
ax.set_xticks(range(len(top_for_matrix))); ax.set_xticklabels(top_for_matrix, rotation=45, ha="right")
ax.set_yticks(range(len(top_for_matrix))); ax.set_yticklabels(top_for_matrix)
for i in range(len(top_for_matrix)):
    for j in range(len(top_for_matrix)):
        if matrix[i,j] > 0.01:
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()*0.5 else "black",
                    fontsize=7)
ax.set_title("Intent Co-Occurrence Support — Top 15 intents")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(FIG / "05_cooccurrence_matrix.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 6. LTV PROXY  (Part 3)
# ---------------------------------------------------------------------------
# LTV proxy = repeat sessions × avg satisfaction × completion probability.
# For a free B2C airport bot LTV is engagement, not revenue.
user_stats = (sessions.groupby("user_id")
    .agg(persona=("persona","first"),
         sessions=("session_id","nunique"),
         completed=("completed","sum"),
         csat=("avg_csat","mean"))
    .assign(ltv_score=lambda d: d["sessions"] * d["csat"] * (0.4 + 0.6*d["completed"]/d["sessions"])))
ltv_by_persona = (user_stats.groupby("persona")
    .agg(users=("sessions","size"),
         avg_sessions=("sessions","mean"),
         avg_csat=("csat","mean"),
         completion=("completed","sum"),
         avg_ltv=("ltv_score","mean"))
    .round(2)
    .sort_values("avg_ltv", ascending=False))
ltv_by_persona.to_csv(TAB / "ltv_by_persona.csv")
print("\n[LTV proxy by persona]"); print(ltv_by_persona)

fig, ax = plt.subplots(figsize=(8, 4.5))
ltv_by_persona["avg_ltv"].plot(kind="bar", ax=ax, color="#059669")
ax.set_title("LTV proxy score by persona (higher = more valuable)")
ax.set_ylabel("LTV score"); ax.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig(FIG / "06_ltv_by_persona.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 7. FRUSTRATION SCENARIO EXTRACTION  (Part 3)
# ---------------------------------------------------------------------------
# NLP frustration detection — regex on repetition and negative sentiment cues.
FRUSTRATION_REGEX = re.compile(
    r"\b(again|still|as i said|one more time|useless|not helping|"
    r"seriously|ugh|come on|why is this so hard|hello\?\?)", re.I)

turns["nlp_frustration"] = turns["utterance"].str.contains(FRUSTRATION_REGEX)

# Frustrated sessions
frust_sessions = (turns.groupby("session_id")
    .agg(any_frust=("nlp_frustration","any"),
         n_turns=("turn_index","size"),
         csat=("csat","mean"),
         fb_rate=("is_fallback","mean")))
frust_share = frust_sessions["any_frust"].mean()
print(f"\n[Frustration] {frust_share:.1%} of sessions contain a frustration marker")

# Pick worst-CSAT frustrated sessions but require at least 3 turns for richer narratives
rich_frust = frust_sessions[frust_sessions["any_frust"] & (frust_sessions["n_turns"] >= 3)]
worst = rich_frust.sort_values("csat").head(6).index.tolist()
# fallback: if none, take any worst-CSAT frustrated sessions
if not worst:
    worst = frust_sessions[frust_sessions["any_frust"]].sort_values("csat").head(6).index.tolist()
frust_examples = turns[turns["session_id"].isin(worst)][
    ["session_id","turn_index","persona","utterance","true_intent",
     "predicted_intent","is_fallback","csat"]]
frust_examples.to_csv(TAB / "frustration_examples.csv", index=False)
print("\n[Frustration example session — first 6 turns]")
print(frust_examples.head(6).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. FUNNEL — a classic session funnel analysis
# ---------------------------------------------------------------------------
funnel = {
    "1. Sessions started":       len(sessions),
    "2. Any intent understood":  int((sessions["any_fallback"] == False).sum()),
    "3. No frustration signals": int((sessions["any_frustration"] == False).sum()),
    "4. Session completed":      int(sessions["completed"].sum()),
}
pd.Series(funnel, name="sessions").to_csv(TAB / "funnel.csv", header=True)

fig, ax = plt.subplots(figsize=(8, 4))
ys = list(funnel.keys())[::-1]; xs = [funnel[k] for k in ys]
ax.barh(ys, xs, color=["#94a3b8","#38bdf8","#0ea5e9","#0369a1"])
for i, v in enumerate(xs):
    pct = v/len(sessions)*100
    ax.text(v+3, i, f"{v}  ({pct:.0f}%)", va="center")
ax.set_title("Session funnel (drop-off analysis)"); ax.set_xlabel("sessions")
fig.tight_layout(); fig.savefig(FIG / "07_funnel.png"); plt.close(fig)

print("\n[Funnel]"); print(pd.Series(funnel))

# ---------------------------------------------------------------------------
# 9. Save a machine-readable summary that dashboard.py + notebook will read
# ---------------------------------------------------------------------------
summary = {
    "eda": eda,
    "fallback_report": fallback_report,
    "intent_metrics": {"accuracy": round(acc,4), "macro_f1": round(f1m,4),
                       "weighted_f1": round(f1w,4)},
    "top_cooccurrence_rules": rules_df.head(10).to_dict(orient="records"),
    "ltv_by_persona": ltv_by_persona.reset_index().to_dict(orient="records"),
    "frustration_session_share_%": round(frust_share*100, 2),
    "funnel": funnel,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nAll outputs written under {OUT}/")
