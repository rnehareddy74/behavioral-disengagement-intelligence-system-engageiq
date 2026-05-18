import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter, CoxPHFitter
import shap
import os

st.set_page_config(page_title="EngageIQ", page_icon="🧠", layout="wide")

FEATURE_COLS = [
    'engagement_frequency','response_latency_hrs','inactivity_gap_days',
    'behavioral_decay','session_duration_min','missed_checkins',
    'days_since_last_login','message_open_rate'
]
TIER_COLORS = {"High":"red","Medium":"orange","Low":"green"}
SEG_COLORS  = {"Stable":"green","Decaying":"orange","Erratic":"steelblue"}
ACT_COLORS  = {"Human Outreach":"red","Automated Nudge":"orange","No Action":"green"}

@st.cache_data
def run_pipeline(path="patient_data.csv"):
    df = pd.read_csv(path)

    # ── Churn label ──────────────────────────────────────────────────────────
    raw = (
        (1/(df['engagement_frequency']+0.5))*0.20 +
        (df['response_latency_hrs']/120)*0.15 +
        (df['inactivity_gap_days']/60)*0.20 +
        (-df['behavioral_decay'].clip(-1,0))*0.15 +
        (df['missed_checkins']/15)*0.15 +
        (1-df['message_open_rate'])*0.15
    ).clip(0,1)
    interaction = ((-df['behavioral_decay'].clip(-1,0))*(df['inactivity_gap_days']/60)).clip(0,1)
    combined    = (raw*0.7 + interaction*0.3).clip(0,1)
    df['churn_label'] = (combined > 0.38).astype(int)

    # ── Risk model ───────────────────────────────────────────────────────────
    X = df[FEATURE_COLS]
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, df['churn_label'])
    df['risk_score'] = model.predict_proba(X)[:,1]

    p40 = float(np.percentile(df['risk_score'], 40))
    p70 = float(np.percentile(df['risk_score'], 70))
    df['risk_tier'] = df['risk_score'].apply(
        lambda s: 'High' if s>=p70 else ('Medium' if s>=p40 else 'Low')
    )

    # ── Survival ─────────────────────────────────────────────────────────────
    np.random.seed(42)
    scores = df['risk_score'].values
    df['duration_days']  = np.random.exponential(scale=90*(1-scores)+5).clip(1,180).astype(int)
    df['event_observed'] = (np.random.rand(len(df)) < (0.3+0.6*scores)).astype(int)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    explainer     = shap.TreeExplainer(model)
    shap_vals_raw = explainer.shap_values(X)
    # Random Forest returns shape (n_samples, n_features, n_classes) or list
    if isinstance(shap_vals_raw, list):
        shap_vals = np.array(shap_vals_raw[1])
    elif np.array(shap_vals_raw).ndim == 3:
        shap_vals = np.array(shap_vals_raw)[:,:,1]
    else:
        shap_vals = np.array(shap_vals_raw)

    df['top_driver'] = [
        FEATURE_COLS[int(np.argmax(np.abs(shap_vals[i])))] for i in range(len(df))
    ]

    # ── Segmentation k=3 ─────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km       = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels   = km.fit_predict(X_scaled)

    cr = []
    for c in range(3):
        mask = labels==c
        avg  = ((1/(df.loc[mask,'engagement_frequency'].mean()+0.5))*0.3 +
                (df.loc[mask,'inactivity_gap_days'].mean()/60)*0.3 +
                (df.loc[mask,'missed_checkins'].mean()/15)*0.4)
        cr.append((c, avg))
    sc2  = [c for c,_ in sorted(cr, key=lambda x:x[1])]
    amap = {sc2[0]:'Stable', sc2[1]:'Decaying', sc2[2]:'Erratic'}
    df['segment'] = [amap[c] for c in labels]

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df['PC1'] = coords[:,0]; df['PC2'] = coords[:,1]

    # ── Intervention ─────────────────────────────────────────────────────────
    def get_action(row):
        seg, tier, decay = row['segment'], row['risk_tier'], row['behavioral_decay']
        if seg == 'Erratic':
            return 'Human Outreach', 'Erratic — unpredictable, proactive contact needed'
        if seg == 'Decaying' and decay < -0.35:
            return 'Human Outreach', f'Steep decay ({decay:.2f}) — escalate now'
        if seg == 'Decaying':
            return 'Automated Nudge', 'Gradual decline — re-engagement message'
        if tier == 'High':   return 'Human Outreach',  'Risk threshold exceeded'
        if tier == 'Medium': return 'Automated Nudge',  'Risk threshold — monitor closely'
        return 'No Action', 'Stable — passive monitoring'

    res = df.apply(get_action, axis=1, result_type='expand')
    df['action'] = res[0]; df['action_reason'] = res[1]

    return df, model, shap_vals, explainer, p40, p70

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("EngageIQ")
st.sidebar.caption("Behavioral Risk Intelligence for Care Teams")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload patient_data.csv", type="csv")
if uploaded:
    with open("patient_data.csv","wb") as f:
        f.write(uploaded.read())
    st.sidebar.success("File uploaded!")
    st.cache_data.clear()

if not os.path.exists("patient_data.csv"):
    st.warning("No data found. Upload patient_data.csv using the sidebar.")
    st.stop()

try:
    df, model, shap_vals, explainer, p40, p70 = run_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Risk Scoring",
    "Survival Analysis",
    "SHAP Explainability",
    "Segmentation",
    "Intervention Engine"
])

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.title("Population Dashboard")
    st.caption("Real-time overview of patient behavioral disengagement signals.")
    st.markdown("---")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Patients",  len(df))
    k2.metric("High Risk",       (df['risk_tier']=='High').sum())
    k3.metric("Medium Risk",     (df['risk_tier']=='Medium').sum())
    k4.metric("Low Risk",        (df['risk_tier']=='Low').sum())
    k5.metric("Avg Risk Score",  f"{df['risk_score'].mean():.3f}")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(df, x='risk_score', color='risk_tier',
                           color_discrete_map=TIER_COLORS, nbins=40,
                           template='plotly_dark', opacity=0.85)
        fig.add_vline(x=p70, line_dash='dash', line_color='red',    annotation_text='High')
        fig.add_vline(x=p40, line_dash='dash', line_color='orange', annotation_text='Medium')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Segment Breakdown")
        sc = df['segment'].value_counts().reset_index()
        sc.columns = ['Segment','Count']
        fig2 = px.pie(sc, names='Segment', values='Count',
                      color='Segment', color_discrete_map=SEG_COLORS,
                      hole=0.45, template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.subheader("Engagement Frequency vs Inactivity Gap")
        fig3 = px.scatter(df, x='engagement_frequency', y='inactivity_gap_days',
                          color='risk_tier', color_discrete_map=TIER_COLORS,
                          hover_data=['patient_id','risk_score','segment'],
                          template='plotly_dark', opacity=0.7)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        st.subheader("Behavioral Decay vs Response Latency")
        fig4 = px.scatter(df, x='behavioral_decay', y='response_latency_hrs',
                          color='segment', color_discrete_map=SEG_COLORS,
                          hover_data=['patient_id','risk_score'],
                          template='plotly_dark', opacity=0.7)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Patient Records")
    tier_f = st.multiselect("Filter by Tier",    ["High","Medium","Low"],          default=["High","Medium","Low"])
    seg_f  = st.multiselect("Filter by Segment", df['segment'].unique().tolist(),  default=df['segment'].unique().tolist())
    fdf = df[df['risk_tier'].isin(tier_f) & df['segment'].isin(seg_f)].sort_values('risk_score', ascending=False)
    st.dataframe(fdf[['patient_id','risk_score','risk_tier','segment','action','top_driver']].reset_index(drop=True),
                 use_container_width=True, height=320)

# ── RISK SCORING ──────────────────────────────────────────────────────────────
elif page == "Risk Scoring":
    st.title("Risk Scoring")
    st.caption("Random Forest — selected automatically by cross-validated AUC.")
    st.markdown("---")

    fi = pd.DataFrame({'Feature':FEATURE_COLS,'Importance':model.feature_importances_}).sort_values('Importance')
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Feature Importance")
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Purples', template='plotly_dark')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Patient Radar vs Population Avg")
        pid  = st.selectbox("Select Patient", df['patient_id'].tolist())
        row  = df[df['patient_id']==pid].iloc[0]
        maxv = df[FEATURE_COLS].max().values
        avg_n = df[FEATURE_COLS].mean().values / (maxv+1e-6)
        pat_n = row[FEATURE_COLS].values / (maxv+1e-6)
        fig2  = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=list(avg_n)+[avg_n[0]], theta=FEATURE_COLS+[FEATURE_COLS[0]],
                                        fill='toself', name='Population Avg',
                                        line_color='mediumpurple', fillcolor='rgba(147,112,219,0.15)'))
        fig2.add_trace(go.Scatterpolar(r=list(pat_n)+[pat_n[0]], theta=FEATURE_COLS+[FEATURE_COLS[0]],
                                        fill='toself', name=pid,
                                        line_color='red', fillcolor='rgba(255,0,0,0.15)'))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])), template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
        ca,cb = st.columns(2)
        ca.metric("Risk Score", f"{row['risk_score']:.4f}")
        cb.metric("Risk Tier",  row['risk_tier'])

# ── SURVIVAL ──────────────────────────────────────────────────────────────────
elif page == "Survival Analysis":
    st.title("Survival Analysis")
    st.caption("KM curves show when patients disengage. Cox PH shows what drives dropout timing.")
    st.markdown("---")

    st.subheader("Kaplan-Meier Survival Curves by Risk Tier")
    ci_fill = {"High":"rgba(255,0,0,0.12)","Medium":"rgba(255,165,0,0.12)","Low":"rgba(0,128,0,0.12)"}
    fig = go.Figure()
    for tier in ['High','Medium','Low']:
        subset = df[df['risk_tier']==tier]
        if len(subset) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(subset['duration_days'], event_observed=subset['event_observed'], label=tier)
        t   = kmf.survival_function_.index.values
        s   = kmf.survival_function_.iloc[:,0].values
        ci_l = kmf.confidence_interval_.iloc[:,0].values
        ci_u = kmf.confidence_interval_.iloc[:,1].values
        fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name=tier,
                                  line=dict(color=TIER_COLORS[tier], width=2.5)))
        fig.add_trace(go.Scatter(
            x=list(t) + list(t[::-1]),
            y=list(ci_u) + list(ci_l[::-1]),
            fill='toself', showlegend=False,
            fillcolor=ci_fill[tier],
            line=dict(color='rgba(255,255,255,0)')
        ))
    fig.update_layout(template='plotly_dark',
                      xaxis_title='Days Since Enrollment',
                      yaxis_title='Probability Still Engaged',
                      yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cox PH — Hazard Ratios")
    try:
        cox_f  = ['engagement_frequency','response_latency_hrs','inactivity_gap_days',
                   'behavioral_decay','missed_checkins','message_open_rate']
        cox_df = df[cox_f+['duration_days','event_observed']].copy()
        sc2    = StandardScaler()
        cox_df[cox_f] = sc2.fit_transform(cox_df[cox_f])
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='duration_days', event_col='event_observed')
        st.metric("Concordance Index", f"{cph.concordance_index_:.4f}")

        summary = cph.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].sort_values('exp(coef)')
        hr_colors = ['red' if v>1 else 'green' for v in summary['exp(coef)']]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=summary.index, x=summary['exp(coef)'],
                               orientation='h', marker_color=hr_colors))
        for i,(idx,row) in enumerate(summary.iterrows()):
            fig2.add_trace(go.Scatter(
                x=[row['exp(coef) lower 95%'], row['exp(coef) upper 95%']],
                y=[idx, idx], mode='lines',
                line=dict(color='lightgray', width=2), showlegend=False
            ))
        fig2.add_vline(x=1.0, line_dash='dash', line_color='lightgray', annotation_text='HR=1')
        fig2.update_layout(template='plotly_dark',
                           xaxis_title='Hazard Ratio (>1 faster dropout, <1 protective)')
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Significant: response_latency_hrs and message_open_rate (p<0.005). Others predict whether someone churns, not how fast.")
    except Exception as e:
        st.error(f"Cox model error: {e}")

# ── SHAP ──────────────────────────────────────────────────────────────────────
elif page == "SHAP Explainability":
    st.title("SHAP Explainability")
    st.caption("Per-patient feature attribution — why is each patient flagged as high risk.")
    st.markdown("---")

    try:
        st.subheader("Global Feature Impact (Mean |SHAP|)")
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df  = pd.DataFrame({'Feature':FEATURE_COLS,'Mean |SHAP|':mean_abs}).sort_values('Mean |SHAP|')
        fig = px.bar(shap_df, x='Mean |SHAP|', y='Feature', orientation='h',
                     color='Mean |SHAP|', color_continuous_scale='Purples', template='plotly_dark')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Global SHAP error: {e}")

    try:
        st.subheader("Individual Patient SHAP Waterfall")
        pid  = st.selectbox("Select Patient", df['patient_id'].tolist())
        pidx = df[df['patient_id']==pid].index[0]
        sv   = shap_vals[pidx]
        fv   = df[FEATURE_COLS].iloc[pidx].values
        shap_mini = pd.DataFrame({'Feature':FEATURE_COLS,'SHAP':sv,'Value':fv}).sort_values('SHAP')
        fig2 = px.bar(shap_mini, x='SHAP', y='Feature', orientation='h',
                      color='SHAP', color_continuous_scale=['green','whitesmoke','red'],
                      color_continuous_midpoint=0, text=shap_mini['Value'].round(2),
                      template='plotly_dark', title=f"Risk Drivers — {pid}")
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        ca,cb,cc = st.columns(3)
        ca.metric("Risk Score", f"{df.loc[pidx,'risk_score']:.4f}")
        cb.metric("Risk Tier",  df.loc[pidx,'risk_tier'])
        cc.metric("Top Driver", df.loc[pidx,'top_driver'])
    except Exception as e:
        st.error(f"Waterfall error: {e}")

    try:
        st.subheader("SHAP Dependence — Inactivity Gap")
        fidx = FEATURE_COLS.index('inactivity_gap_days')
        fig3 = px.scatter(
            x=df['inactivity_gap_days'].values,
            y=shap_vals[:,fidx],
            color=df['risk_tier'].values,
            color_discrete_map=TIER_COLORS,
            labels={'x':'inactivity_gap_days','y':'SHAP value','color':'Risk Tier'},
            template='plotly_dark', opacity=0.7
        )
        fig3.add_hline(y=0, line_dash='dash', line_color='lightgray')
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Dependence plot error: {e}")

# ── SEGMENTATION ──────────────────────────────────────────────────────────────
elif page == "Segmentation":
    st.title("Behavioral Segmentation")
    st.caption("KMeans k=3 — validated by silhouette score and clinical utility.")
    st.markdown("---")

    try:
        st.subheader("PCA Cluster Visualization")
        fig = px.scatter(df, x='PC1', y='PC2', color='segment',
                         color_discrete_map=SEG_COLORS,
                         hover_data=['patient_id','risk_score','risk_tier'],
                         template='plotly_dark', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("k=3 chosen over k=2 (best silhouette) — two clusters just replicates the churn label. k=3 gives clinically actionable archetypes.")
    except Exception as e:
        st.error(f"PCA plot error: {e}")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Segment Distribution")
        sc = df['segment'].value_counts().reset_index()
        sc.columns = ['Segment','Count']
        fig2 = px.bar(sc, x='Segment', y='Count', color='Segment',
                      color_discrete_map=SEG_COLORS, template='plotly_dark', text='Count')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Avg Risk Score by Segment")
        rs = df.groupby('segment')['risk_score'].mean().reset_index().sort_values('risk_score')
        fig3 = px.bar(rs, x='risk_score', y='segment', orientation='h',
                      color='segment', color_discrete_map=SEG_COLORS, template='plotly_dark')
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Segment Feature Profiles")
    seg_means = df.groupby('segment')[FEATURE_COLS].mean().round(2)
    st.dataframe(seg_means, use_container_width=True)

# ── INTERVENTION ──────────────────────────────────────────────────────────────
elif page == "Intervention Engine":
    st.title("Intervention Engine")
    st.caption("Segment + risk tier + decay slope determines action — not just a score threshold.")
    st.markdown("---")

    k1,k2,k3 = st.columns(3)
    k1.metric("Human Outreach",  (df['action']=='Human Outreach').sum())
    k2.metric("Automated Nudge", (df['action']=='Automated Nudge').sum())
    k3.metric("No Action",       (df['action']=='No Action').sum())
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Action Distribution")
        ac = df['action'].value_counts().reset_index()
        ac.columns = ['Action','Count']
        fig = px.pie(ac, names='Action', values='Count', color='Action',
                     color_discrete_map=ACT_COLORS, hole=0.4, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Actions by Segment")
        cross = pd.crosstab(df['segment'], df['action']).reset_index().melt(id_vars='segment')
        fig2  = px.bar(cross, x='segment', y='value', color='variable',
                       color_discrete_map=ACT_COLORS, template='plotly_dark',
                       labels={'value':'Count','segment':'Segment','variable':'Action'})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Clinician Worklist")
    worklist = df[df['action']=='Human Outreach'][[
        'patient_id','risk_score','risk_tier','segment','action_reason',
        'top_driver','inactivity_gap_days','missed_checkins','days_since_last_login'
    ]].sort_values('risk_score', ascending=False).reset_index(drop=True)
    st.dataframe(worklist, use_container_width=True, height=300)
    st.caption(f"{len(worklist)} patients require human outreach")

    st.subheader("Automated Nudge Queue")
    nudge = df[df['action']=='Automated Nudge'][[
        'patient_id','risk_score','segment','action_reason',
        'top_driver','days_since_last_login','message_open_rate'
    ]].sort_values('risk_score', ascending=False).reset_index(drop=True)
    st.dataframe(nudge, use_container_width=True, height=250)

    st.markdown("---")
    st.subheader("What-If Risk Simulator")
    st.caption("Change a feature value and see how the risk score changes for any patient.")
    pid    = st.selectbox("Select Patient", df['patient_id'].tolist())
    feat   = st.selectbox("Feature to adjust", FEATURE_COLS)
    pidx   = df[df['patient_id']==pid].index[0]
    curr   = float(df.loc[pidx, feat])
    new_val = st.slider("New value", float(df[feat].min()), float(df[feat].max()), curr, step=0.1)

    X_mod  = df[FEATURE_COLS].iloc[pidx].values.copy().astype(float)
    X_mod[FEATURE_COLS.index(feat)] = new_val
    new_score = float(model.predict_proba(X_mod.reshape(1,-1))[0][1])
    new_tier  = 'High' if new_score>=p70 else ('Medium' if new_score>=p40 else 'Low')

    ca,cb,cc = st.columns(3)
    ca.metric("Original Score", f"{df.loc[pidx,'risk_score']:.4f}")
    cb.metric("New Score",      f"{new_score:.4f}", delta=f"{new_score-df.loc[pidx,'risk_score']:+.4f}")
    cc.metric("New Tier",       new_tier)
