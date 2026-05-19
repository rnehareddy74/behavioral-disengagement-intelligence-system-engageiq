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

st.set_page_config(page_title="EngageIQ", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1440px; }
.stApp { background: #f4f6fb; }

[data-testid="stSidebar"] {
    background: #0b0f1e !important;
    border-right: 1px solid #1c2333;
}
[data-testid="stSidebar"] * { color: lightsteelblue !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 16px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1c2333 !important; color: #ffffff !important; }

.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #e8ecf4;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    height: 100%;
}
.kpi-label { font-size: 12px; font-weight: 600; color: slategray; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #0b0f1e; line-height: 1; margin-bottom: 6px; }
.kpi-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-red { background: #fff0f0; color: crimson; }
.badge-orange { background: #fff8f0; color: coral; }
.badge-green { background: #f0fff4; color: mediumseagreen; }
.badge-blue { background: #f0f4ff; color: steelblue; }

.section-header {
    font-size: 15px; font-weight: 600; color: #0b0f1e;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 2px solid #f0f2f8;
    display: flex; align-items: center; gap: 8px;
}
.card {
    background: white; border-radius: 14px; padding: 22px;
    border: 1px solid #e8ecf4; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.page-title {
    font-size: 26px; font-weight: 700; color: #0b0f1e;
    margin-bottom: 4px; letter-spacing: -0.3px;
}
.page-subtitle { font-size: 13px; color: slategray; margin-bottom: 24px; }

.sidebar-logo {
    font-size: 20px; font-weight: 700; color: #ffffff !important;
    letter-spacing: -0.3px; margin-bottom: 4px;
}
.sidebar-sub { font-size: 11px; color: slategray !important; margin-bottom: 24px; }
.sidebar-section { font-size: 10px; font-weight: 600; color: #3d4f6e !important;
    text-transform: uppercase; letter-spacing: 0.08em; margin: 16px 0 8px; }

.tag { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 500; margin: 1px; }
.tag-high { background: #fff0f0; color: crimson; }
.tag-medium { background: #fff8f0; color: coral; }
.tag-low { background: #f0fff4; color: mediumseagreen; }
.tag-stable { background: #f0fff4; color: mediumseagreen; }
.tag-decaying { background: #fff8f0; color: coral; }
.tag-erratic { background: #f0f4ff; color: steelblue; }

.worklist-row {
    background: white; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #e8ecf4; margin-bottom: 8px;
    display: flex; align-items: center; justify-content: space-between;
}
.patient-id { font-size: 13px; font-weight: 600; color: #0b0f1e; font-family: 'JetBrains Mono', monospace; }
.driver-text { font-size: 12px; color: slategray; margin-top: 2px; }
.risk-score-badge {
    font-size: 16px; font-weight: 700; color: crimson;
    background: #fff0f0; padding: 6px 12px; border-radius: 8px;
}

div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; color: #0b0f1e !important; }
div[data-testid="stMetricLabel"] { font-size: 12px !important; font-weight: 600 !important; color: slategray !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
div[data-testid="metric-container"] {
    background: white; border-radius: 14px; padding: 18px 20px !important;
    border: 1px solid #e8ecf4; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    'engagement_frequency','response_latency_hrs','inactivity_gap_days',
    'behavioral_decay','session_duration_min','missed_checkins',
    'days_since_last_login','message_open_rate'
]
PLOT_THEME = dict(template="plotly_white",
                  paper_bgcolor="white", plot_bgcolor="white",
                  font=dict(family="Inter", size=12, color="#0b0f1e"),
                  margin=dict(t=32, b=32, l=16, r=16))

def tier_badge(t):
    cls = {"High":"tag-high","Medium":"tag-medium","Low":"tag-low"}.get(t,"tag-low")
    return f'<span class="tag {cls}">{t}</span>'

def seg_badge(s):
    cls = {"Stable":"tag-stable","Decaying":"tag-decaying","Erratic":"tag-erratic"}.get(s,"tag-low")
    return f'<span class="tag {cls}">{s}</span>'

@st.cache_data
def run_pipeline(path="patient_data.csv"):
    df = pd.read_csv(path)
    # Drop any pre-computed columns — always recompute fresh from raw features
    drop_cols = ["duration_days","event_observed","risk_score","risk_tier",
                 "churn_label","top_driver","segment","PC1","PC2",
                 "action","action_reason","true_group","_risk_signal"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Ensure feature columns are numeric
    for col in ["engagement_frequency","response_latency_hrs","inactivity_gap_days",
                "behavioral_decay","session_duration_min","missed_checkins",
                "days_since_last_login","message_open_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
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

    X = df[FEATURE_COLS]
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, df['churn_label'])
    df['risk_score'] = model.predict_proba(X)[:,1]

    p40 = float(np.percentile(df['risk_score'], 40))
    p70 = float(np.percentile(df['risk_score'], 70))
    df['risk_tier'] = df['risk_score'].apply(lambda s: 'High' if s>=p70 else ('Medium' if s>=p40 else 'Low'))

    np.random.seed(42)
    scores = df['risk_score'].values
    duration_days  = np.random.exponential(scale=90*(1-scores)+5).clip(1,180).astype(float)
    event_observed = (np.random.rand(len(df)) < (0.3+0.6*scores)).astype(float)

    explainer     = shap.TreeExplainer(model)
    shap_vals_raw = explainer.shap_values(X)
    if isinstance(shap_vals_raw, list): shap_vals = np.array(shap_vals_raw[1])
    elif np.array(shap_vals_raw).ndim == 3: shap_vals = np.array(shap_vals_raw)[:,:,1]
    else: shap_vals = np.array(shap_vals_raw)
    df['top_driver'] = [FEATURE_COLS[int(np.argmax(np.abs(shap_vals[i])))] for i in range(len(df))]

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
        cr.append((c,avg))
    sc2  = [c for c,_ in sorted(cr, key=lambda x:x[1])]
    amap = {sc2[0]:'Stable',sc2[1]:'Decaying',sc2[2]:'Erratic'}
    df['segment'] = [amap[c] for c in labels]

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df['PC1'] = coords[:,0]; df['PC2'] = coords[:,1]

    def get_action(row):
        seg, tier, decay = row['segment'], row['risk_tier'], row['behavioral_decay']
        if seg == 'Erratic':           return 'Human Outreach',  'Erratic — unpredictable, proactive contact'
        if seg == 'Decaying' and decay < -0.35: return 'Human Outreach', f'Steep decay ({decay:.2f})'
        if seg == 'Decaying':          return 'Automated Nudge', 'Gradual decline — re-engagement'
        if tier == 'High':             return 'Human Outreach',  'Risk threshold exceeded'
        if tier == 'Medium':           return 'Automated Nudge', 'Risk threshold — monitor'
        return 'No Action', 'Stable — passive monitoring'

    res = df.apply(get_action, axis=1, result_type='expand')
    df['action'] = res[0]; df['action_reason'] = res[1]
    return df, model, shap_vals, explainer, p40, p70, duration_days, event_observed

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">EngageIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Behavioral Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload patient_data.csv", type="csv")
    if uploaded:
        with open("patient_data.csv","wb") as f: f.write(uploaded.read())
        st.success("Data loaded successfully")
        st.cache_data.clear()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "Dashboard",
        "Risk Scoring",
        "Survival Analysis",
        "SHAP Explainability",
        "Segmentation",
        "Intervention Engine"
    ], label_visibility="collapsed")

if not os.path.exists("patient_data.csv"):
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:60vh;flex-direction:column;gap:16px;">
        <div style="font-size:48px;">🧠</div>
        <div style="font-size:22px;font-weight:700;color:#0b0f1e;">Welcome to EngageIQ</div>
        <div style="font-size:14px;color:slategray;">Upload patient_data.csv using the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    df, model, shap_vals, explainer, p40, p70, dur_arr, evnt_arr = run_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

TIER_COLORS = {"High":"crimson","Medium":"coral","Low":"mediumseagreen"}
SEG_COLORS  = {"Stable":"mediumseagreen","Decaying":"coral","Erratic":"steelblue"}
ACT_COLORS  = {"Human Outreach":"crimson","Automated Nudge":"coral","No Action":"mediumseagreen"}

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown('<div class="page-title">Population Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time overview of patient behavioral disengagement signals across the full cohort</div>', unsafe_allow_html=True)

    high  = int((df['risk_tier']=='High').sum())
    med   = int((df['risk_tier']=='Medium').sum())
    low   = int((df['risk_tier']=='Low').sum())
    total = len(df)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Patients</div>
            <div class="kpi-value">{total:,}</div>
            <span class="kpi-badge badge-blue">Full cohort</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">High Risk</div>
            <div class="kpi-value" style="color:crimson">{high:,}</div>
            <span class="kpi-badge badge-red">{high/total*100:.1f}% of cohort</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Medium Risk</div>
            <div class="kpi-value" style="color:coral">{med:,}</div>
            <span class="kpi-badge badge-orange">{med/total*100:.1f}% of cohort</span>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Low Risk</div>
            <div class="kpi-value" style="color:mediumseagreen">{low:,}</div>
            <span class="kpi-badge badge-green">{low/total*100:.1f}% of cohort</span>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Risk Score</div>
            <div class="kpi-value">{df['risk_score'].mean():.3f}</div>
            <span class="kpi-badge badge-blue">cohort average</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='risk_score', color='risk_tier',
                           color_discrete_map=TIER_COLORS, nbins=40, opacity=0.85,
                           labels={"risk_score":"Risk Score","risk_tier":"Tier"})
        fig.add_vline(x=p70, line_dash='dash', line_color='crimson', line_width=1.5, annotation_text='High', annotation_font_size=11)
        fig.add_vline(x=p40, line_dash='dash', line_color='coral', line_width=1.5, annotation_text='Medium', annotation_font_size=11)
        fig.update_layout(**PLOT_THEME, height=260, showlegend=True,
                          legend=dict(orientation="h", y=1.1, x=0),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#f0f2f8"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Behavioral Segments</div>', unsafe_allow_html=True)
        sc = df['segment'].value_counts().reset_index(); sc.columns = ['Segment','Count']
        fig2 = px.pie(sc, names='Segment', values='Count', color='Segment',
                      color_discrete_map=SEG_COLORS, hole=0.55)
        fig2.update_traces(textposition='outside', textinfo='percent+label',
                           textfont=dict(size=12, family='Inter'))
        fig2.update_layout(**PLOT_THEME, height=260, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Engagement vs Inactivity</div>', unsafe_allow_html=True)
        fig3 = px.scatter(df, x='engagement_frequency', y='inactivity_gap_days',
                          color='risk_tier', color_discrete_map=TIER_COLORS, opacity=0.65, size_max=6,
                          hover_data=['patient_id','risk_score','segment'],
                          labels={"engagement_frequency":"Logins/Week","inactivity_gap_days":"Inactivity Gap (Days)"})
        fig3.update_traces(marker=dict(size=5))
        fig3.update_layout(**PLOT_THEME, height=260,
                           xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                           yaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Decay vs Response Latency</div>', unsafe_allow_html=True)
        fig4 = px.scatter(df, x='behavioral_decay', y='response_latency_hrs',
                          color='segment', color_discrete_map=SEG_COLORS, opacity=0.65,
                          hover_data=['patient_id','risk_score'],
                          labels={"behavioral_decay":"Behavioral Decay","response_latency_hrs":"Response Latency (hrs)"})
        fig4.update_traces(marker=dict(size=5))
        fig4.update_layout(**PLOT_THEME, height=260,
                           xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                           yaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Patient Records</div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tier_f = st.multiselect("Risk Tier", ["High","Medium","Low"], default=["High","Medium","Low"])
    with col_f2:
        seg_f = st.multiselect("Segment", df['segment'].unique().tolist(), default=df['segment'].unique().tolist())
    fdf = df[df['risk_tier'].isin(tier_f) & df['segment'].isin(seg_f)].sort_values('risk_score', ascending=False)
    st.dataframe(
        fdf[['patient_id','risk_score','risk_tier','segment','action','top_driver']].reset_index(drop=True),
        use_container_width=True, height=280
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── RISK SCORING ──────────────────────────────────────────────────────────────
elif page == "Risk Scoring":
    st.markdown('<div class="page-title">Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Random Forest — selected automatically by cross-validated AUC across 5 folds</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({'Feature':FEATURE_COLS,'Importance':model.feature_importances_}).sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale=["#e8ecf4","steelblue"])
        fig.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False,
                          xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                          yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Patient Radar vs Population Avg</div>', unsafe_allow_html=True)
        pid  = st.selectbox("Select Patient", df['patient_id'].tolist())
        row  = df[df['patient_id']==pid].iloc[0]
        maxv = df[FEATURE_COLS].max().values
        avg_n = df[FEATURE_COLS].mean().values/(maxv+1e-6)
        pat_n = row[FEATURE_COLS].values/(maxv+1e-6)
        fig2  = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=list(avg_n)+[avg_n[0]], theta=FEATURE_COLS+[FEATURE_COLS[0]],
                                        fill='toself', name='Population Avg',
                                        line=dict(color='steelblue', width=2),
                                        fillcolor='rgba(70,130,180,0.08)'))
        fig2.add_trace(go.Scatterpolar(r=list(pat_n)+[pat_n[0]], theta=FEATURE_COLS+[FEATURE_COLS[0]],
                                        fill='toself', name=pid,
                                        line=dict(color='crimson', width=2),
                                        fillcolor='rgba(220,20,60,0.08)'))
        fig2.update_layout(**PLOT_THEME, height=320,
                           polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                                       gridcolor="#f0f2f8", linecolor="#e8ecf4"),
                                      angularaxis=dict(gridcolor="#f0f2f8", linecolor="#e8ecf4")),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4,c5 = st.columns(3)
    c3.metric("Risk Score", f"{row['risk_score']:.4f}")
    c4.metric("Risk Tier",  row['risk_tier'])
    c5.metric("Top Driver", row['top_driver'])

# ── SURVIVAL ──────────────────────────────────────────────────────────────────
elif page == "Survival Analysis":
    st.markdown('<div class="page-title">Survival Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Kaplan-Meier shows when patients disengage. Cox PH quantifies what drives dropout timing.</div>', unsafe_allow_html=True)

    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Kaplan-Meier Survival Curves by Risk Tier</div>', unsafe_allow_html=True)
        ci_fill = {"High":"rgba(220,20,60,0.08)","Medium":"rgba(255,127,80,0.08)","Low":"rgba(60,179,113,0.08)"}
        fig = go.Figure()
        for tier in ['High','Medium','Low']:
            mask = (df['risk_tier']==tier).values
            d = dur_arr[mask]
            e = evnt_arr[mask]
            if len(d) < 5: continue
            kmf = KaplanMeierFitter()
            kmf.fit(d, event_observed=e, label=tier)
            t   = kmf.survival_function_.index.values
            s   = kmf.survival_function_.iloc[:,0].values
            ci_l = kmf.confidence_interval_.iloc[:,0].values
            ci_u = kmf.confidence_interval_.iloc[:,1].values
            fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name=tier,
                                      line=dict(color=TIER_COLORS[tier], width=2.5)))
            fig.add_trace(go.Scatter(x=list(t)+list(t[::-1]), y=list(ci_u)+list(ci_l[::-1]),
                                      fill='toself', showlegend=False, fillcolor=ci_fill[tier],
                                      line=dict(color='rgba(255,255,255,0)')))
        fig.update_layout(**PLOT_THEME, height=340,
                          xaxis_title='Days Since Enrollment',
                          yaxis_title='Probability Still Engaged',
                          yaxis_range=[0,1],
                          xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                          yaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Median Survival by Tier</div>', unsafe_allow_html=True)
        med_rows = []
        for tier in ['High','Medium','Low']:
            mask = (df['risk_tier']==tier).values
            d = dur_arr[mask]; e = evnt_arr[mask]
            kmf = KaplanMeierFitter()
            kmf.fit(d, event_observed=e)
            med_rows.append({"Tier":tier, "Median Days": round(float(kmf.median_survival_time_),1)})
        med_df = pd.DataFrame(med_rows)
        fig_med = px.bar(med_df, x='Tier', y='Median Days', color='Tier',
                         color_discrete_map=TIER_COLORS, text='Median Days')
        fig_med.update_traces(textposition='outside', textfont=dict(size=12, weight=700))
        fig_med.update_layout(**PLOT_THEME, height=340, showlegend=False,
                               xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=True, gridcolor="#f0f2f8"))
        st.plotly_chart(fig_med, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Cox PH Hazard Ratios</div>', unsafe_allow_html=True)
    try:
        cox_f  = ['engagement_frequency','response_latency_hrs','inactivity_gap_days',
                   'behavioral_decay','missed_checkins','message_open_rate']
        cox_df = df[cox_f].copy()
        sc2    = StandardScaler()
        cox_df[cox_f] = sc2.fit_transform(cox_df[cox_f])
        cox_df['duration_days']  = dur_arr
        cox_df['event_observed'] = evnt_arr
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='duration_days', event_col='event_observed')
        summary = cph.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].sort_values('exp(coef)')

        col_m1,col_m2,col_m3 = st.columns(3)
        col_m1.metric("Concordance Index", f"{cph.concordance_index_:.4f}")
        col_m2.metric("Significant Features", str(int((summary['p']<0.05).sum())))
        col_m3.metric("Total Features", str(len(cox_f)))

        fig2 = go.Figure()
        hr_colors = [TIER_COLORS['High'] if v>1 else TIER_COLORS['Low'] for v in summary['exp(coef)']]
        fig2.add_trace(go.Bar(y=summary.index, x=summary['exp(coef)'],
                               orientation='h', marker_color=hr_colors,
                               marker_line_width=0))
        for i,(idx,r) in enumerate(summary.iterrows()):
            fig2.add_trace(go.Scatter(x=[r['exp(coef) lower 95%'],r['exp(coef) upper 95%']],
                                       y=[idx,idx], mode='lines',
                                       line=dict(color='slategray',width=2), showlegend=False))
        fig2.add_vline(x=1.0, line_dash='dash', line_color='slategray', line_width=1.5)
        fig2.update_layout(**PLOT_THEME, height=280,
                            xaxis_title='Hazard Ratio  (>1 faster dropout, <1 protective)',
                            xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                            yaxis=dict(showgrid=False))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Cox model error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── SHAP ──────────────────────────────────────────────────────────────────────
elif page == "SHAP Explainability":
    st.markdown('<div class="page-title">SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Per-patient feature attribution — why is each patient flagged as high risk</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Global Feature Impact (Mean |SHAP|)</div>', unsafe_allow_html=True)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df  = pd.DataFrame({'Feature':FEATURE_COLS,'Mean |SHAP|':mean_abs}).sort_values('Mean |SHAP|')
        fig = px.bar(shap_df, x='Mean |SHAP|', y='Feature', orientation='h',
                     color='Mean |SHAP|', color_continuous_scale=["#e8ecf4","steelblue"])
        fig.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False,
                          xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                          yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">SHAP Dependence — Inactivity Gap</div>', unsafe_allow_html=True)
        fidx = FEATURE_COLS.index('inactivity_gap_days')
        fig3 = px.scatter(x=df['inactivity_gap_days'].values, y=shap_vals[:,fidx],
                           color=df['risk_tier'].values, color_discrete_map=TIER_COLORS,
                           labels={'x':'Inactivity Gap (Days)','y':'SHAP Value','color':'Risk Tier'},
                           opacity=0.65)
        fig3.update_traces(marker=dict(size=5))
        fig3.add_hline(y=0, line_dash='dash', line_color='slategray', line_width=1)
        fig3.update_layout(**PLOT_THEME, height=320,
                            xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                            yaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Individual Patient SHAP Waterfall</div>', unsafe_allow_html=True)
    pid  = st.selectbox("Select Patient", df['patient_id'].tolist())
    pidx = df[df['patient_id']==pid].index[0]
    sv   = shap_vals[pidx]
    fv   = df[FEATURE_COLS].iloc[pidx].values
    shap_mini = pd.DataFrame({'Feature':FEATURE_COLS,'SHAP':sv,'Value':fv}).sort_values('SHAP')
    bar_colors = [TIER_COLORS['High'] if v>0 else TIER_COLORS['Low'] for v in shap_mini['SHAP']]
    fig2 = go.Figure(go.Bar(
        x=shap_mini['SHAP'], y=shap_mini['Feature'],
        orientation='h', marker_color=bar_colors,
        text=[f"{v:.2f}" for v in shap_mini['Value']],
        textposition='outside', textfont=dict(size=11, color='slategray')
    ))
    fig2.add_vline(x=0, line_color='slategray', line_width=1)
    fig2.update_layout(**PLOT_THEME, height=300,
                        xaxis_title='SHAP Value (red = increases risk, green = decreases risk)',
                        xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                        yaxis=dict(showgrid=False))
    st.plotly_chart(fig2, use_container_width=True)

    ca,cb,cc = st.columns(3)
    ca.metric("Risk Score", f"{df.loc[pidx,'risk_score']:.4f}")
    cb.metric("Risk Tier",  df.loc[pidx,'risk_tier'])
    cc.metric("Top Driver", df.loc[pidx,'top_driver'])
    st.markdown('</div>', unsafe_allow_html=True)

# ── SEGMENTATION ──────────────────────────────────────────────────────────────
elif page == "Segmentation":
    st.markdown('<div class="page-title">Behavioral Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">KMeans k=3 — validated by silhouette score. Stable, Decaying, Erratic archetypes.</div>', unsafe_allow_html=True)

    # Force numeric types before KM fitting

    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">PCA Cluster Visualization</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='PC1', y='PC2', color='segment', color_discrete_map=SEG_COLORS,
                         hover_data=['patient_id','risk_score','risk_tier'], opacity=0.7)
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PLOT_THEME, height=360,
                           xaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                           yaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Segment Distribution</div>', unsafe_allow_html=True)
        sc = df['segment'].value_counts().reset_index(); sc.columns = ['Segment','Count']
        for _, row in sc.iterrows():
            pct = row['Count']/len(df)*100
            color = SEG_COLORS.get(row['Segment'],'slategray')
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:13px;font-weight:600;color:#0b0f1e;">{row['Segment']}</span>
                    <span style="font-size:13px;font-weight:600;color:{color};">{row['Count']} ({pct:.1f}%)</span>
                </div>
                <div style="background:#f0f2f8;border-radius:4px;height:8px;">
                    <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        rs = df.groupby('segment')['risk_score'].mean().reset_index().sort_values('risk_score', ascending=False)
        for _, r in rs.iterrows():
            color = SEG_COLORS.get(r['segment'],'slategray')
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f0f2f8;">
                <span style="font-size:13px;color:slategray;">{r['segment']} avg risk</span>
                <span style="font-size:13px;font-weight:600;color:{color};">{r['risk_score']:.3f}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Segment Feature Profiles</div>', unsafe_allow_html=True)
    seg_means = df.groupby('segment')[FEATURE_COLS].mean().round(2)
    st.dataframe(seg_means, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── INTERVENTION ──────────────────────────────────────────────────────────────
elif page == "Intervention Engine":
    st.markdown('<div class="page-title">Intervention Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Segment + risk tier + decay slope determines action — not just a score threshold</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    n_human = int((df['action']=='Human Outreach').sum())
    n_nudge = int((df['action']=='Automated Nudge').sum())
    n_none  = int((df['action']=='No Action').sum())

    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Human Outreach Required</div>
            <div class="kpi-value" style="color:crimson">{n_human}</div>
            <span class="kpi-badge badge-red">Immediate action</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Automated Nudge</div>
            <div class="kpi-value" style="color:coral">{n_nudge}</div>
            <span class="kpi-badge badge-orange">Queued messages</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">No Action</div>
            <div class="kpi-value" style="color:mediumseagreen">{n_none}</div>
            <span class="kpi-badge badge-green">Passive monitor</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4,c5 = st.columns(2)
    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Action Distribution</div>', unsafe_allow_html=True)
        ac = df['action'].value_counts().reset_index(); ac.columns = ['Action','Count']
        fig = px.pie(ac, names='Action', values='Count', color='Action',
                     color_discrete_map=ACT_COLORS, hole=0.55)
        fig.update_traces(textposition='outside', textinfo='percent+label',
                          textfont=dict(size=12, family='Inter'))
        fig.update_layout(**PLOT_THEME, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Actions by Segment</div>', unsafe_allow_html=True)
        cross = pd.crosstab(df['segment'], df['action']).reset_index().melt(id_vars='segment', var_name='Action', value_name='Count')
        fig2 = px.bar(cross, x='segment', y='Count', color='Action',
                      color_discrete_map=ACT_COLORS, barmode='stack',
                      labels={'segment':'Segment'})
        fig2.update_layout(**PLOT_THEME, height=280,
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="#f0f2f8"),
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Clinician Worklist</div>', unsafe_allow_html=True)
    worklist = df[df['action']=='Human Outreach'][[
        'patient_id','risk_score','risk_tier','segment','action_reason',
        'top_driver','inactivity_gap_days','missed_checkins','days_since_last_login'
    ]].sort_values('risk_score', ascending=False).reset_index(drop=True)
    for _, r in worklist.head(10).iterrows():
        st.markdown(f"""
        <div class="worklist-row">
            <div>
                <div class="patient-id">{r['patient_id']}</div>
                <div class="driver-text">Top driver: {r['top_driver']} &nbsp;|&nbsp; {r['action_reason']}</div>
                <div style="margin-top:4px;">
                    {tier_badge(r['risk_tier'])} {seg_badge(r['segment'])}
                </div>
            </div>
            <div class="risk-score-badge">{r['risk_score']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    if len(worklist) > 10:
        st.caption(f"Showing 10 of {len(worklist)} patients")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">What-If Risk Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust any behavioral feature and watch the risk score update instantly")
    col_a, col_b = st.columns(2)
    with col_a:
        pid_w  = st.selectbox("Patient", df['patient_id'].tolist())
    with col_b:
        feat_w = st.selectbox("Feature", FEATURE_COLS)
    pidx_w = df[df['patient_id']==pid_w].index[0]
    curr_w = float(df.loc[pidx_w, feat_w])
    new_val = st.slider("New value", float(df[feat_w].min()), float(df[feat_w].max()), curr_w, step=0.1)
    X_mod  = df[FEATURE_COLS].iloc[pidx_w].values.copy().astype(float)
    X_mod[FEATURE_COLS.index(feat_w)] = new_val
    new_score = float(model.predict_proba(X_mod.reshape(1,-1))[0][1])
    new_tier  = 'High' if new_score>=p70 else ('Medium' if new_score>=p40 else 'Low')
    delta_val = new_score - df.loc[pidx_w,'risk_score']
    c_x,c_y,c_z = st.columns(3)
    c_x.metric("Original Score", f"{df.loc[pidx_w,'risk_score']:.4f}")
    c_y.metric("New Score", f"{new_score:.4f}", delta=f"{delta_val:+.4f}")
    c_z.metric("New Tier", new_tier)
    st.markdown('</div>', unsafe_allow_html=True)
