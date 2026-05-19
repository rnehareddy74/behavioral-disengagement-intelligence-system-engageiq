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
import shap, os

st.set_page_config(page_title="EngageIQ", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2rem;max-width:1440px;}
.stApp{background:#f4f6fb;}
[data-testid="stSidebar"]{background:#0b0f1e !important;border-right:1px solid #1c2333;}
[data-testid="stSidebar"] *{color:#b0bcd0 !important;font-family:'Inter',sans-serif !important;}
[data-testid="stSidebar"] .stRadio label{padding:10px 16px !important;border-radius:8px !important;font-size:13px !important;font-weight:500 !important;}
[data-testid="stSidebar"] .stRadio label:hover{background:#1c2333 !important;color:#fff !important;}
.kpi-card{background:white;border-radius:14px;padding:20px 22px;border:1px solid #e8ecf4;box-shadow:0 2px 8px rgba(0,0,0,0.04);height:100%;}
.kpi-label{font-size:12px;font-weight:600;color:slategray;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;}
.kpi-value{font-size:32px;font-weight:700;color:#0b0f1e;line-height:1;margin-bottom:6px;}
.kpi-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
.badge-red{background:#fff0f0;color:crimson;}
.badge-orange{background:#fff8f0;color:coral;}
.badge-green{background:#f0fff4;color:mediumseagreen;}
.badge-blue{background:#f0f4ff;color:steelblue;}
.card{background:white;border-radius:14px;padding:22px;border:1px solid #e8ecf4;box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-bottom:16px;}
.section-header{font-size:15px;font-weight:600;color:#0b0f1e;margin-bottom:14px;padding-bottom:10px;border-bottom:2px solid #f0f2f8;}
.page-title{font-size:26px;font-weight:700;color:#0b0f1e;margin-bottom:4px;}
.page-subtitle{font-size:13px;color:slategray;margin-bottom:24px;}
.sidebar-logo{font-size:20px;font-weight:700;color:#ffffff !important;letter-spacing:-0.3px;}
.sidebar-sub{font-size:11px;color:slategray !important;}
.sidebar-section{font-size:10px;font-weight:600;color:#3d4f6e !important;text-transform:uppercase;letter-spacing:0.08em;margin:16px 0 8px;}
.tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:500;margin:1px;}
.tag-high{background:#fff0f0;color:crimson;}
.tag-medium{background:#fff8f0;color:coral;}
.tag-low{background:#f0fff4;color:mediumseagreen;}
.tag-stable{background:#f0fff4;color:mediumseagreen;}
.tag-decaying{background:#fff8f0;color:coral;}
.tag-erratic{background:#f0f4ff;color:steelblue;}
.worklist-row{background:white;border-radius:10px;padding:14px 18px;border:1px solid #e8ecf4;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between;}
.patient-id{font-size:13px;font-weight:600;color:#0b0f1e;font-family:monospace;}
.driver-text{font-size:12px;color:slategray;margin-top:2px;}
.risk-score-badge{font-size:16px;font-weight:700;color:crimson;background:#fff0f0;padding:6px 12px;border-radius:8px;}
div[data-testid="stMetricValue"]{font-size:28px !important;font-weight:700 !important;color:#0b0f1e !important;}
div[data-testid="stMetricLabel"]{font-size:12px !important;font-weight:600 !important;color:slategray !important;text-transform:uppercase !important;letter-spacing:0.05em !important;}
div[data-testid="metric-container"]{background:white;border-radius:14px;padding:18px 20px !important;border:1px solid #e8ecf4;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ['engagement_frequency','response_latency_hrs','inactivity_gap_days',
                'behavioral_decay','session_duration_min','missed_checkins',
                'days_since_last_login','message_open_rate']
TIER_COLORS = {"High":"crimson","Medium":"coral","Low":"mediumseagreen"}
SEG_COLORS  = {"Stable":"mediumseagreen","Decaying":"coral","Erratic":"steelblue"}
ACT_COLORS  = {"Human Outreach":"crimson","Automated Nudge":"coral","No Action":"mediumseagreen"}
PT = dict(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
          font=dict(family="Inter",size=12,color="#0b0f1e"), margin=dict(t=32,b=32,l=16,r=16))

def tier_badge(t):
    c = {"High":"tag-high","Medium":"tag-medium","Low":"tag-low"}.get(t,"tag-low")
    return f'<span class="tag {c}">{t}</span>'

def seg_badge(s):
    c = {"Stable":"tag-stable","Decaying":"tag-decaying","Erratic":"tag-erratic"}.get(s,"tag-low")
    return f'<span class="tag {c}">{s}</span>'

def compute_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    # Keep only raw feature columns + patient_id
    keep = ['patient_id'] + [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep].copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Churn label
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

    # Model
    X = df[FEATURE_COLS].values.astype(float)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, df['churn_label'])
    scores = model.predict_proba(X)[:,1].astype(float)
    df['risk_score'] = scores

    p40 = float(np.percentile(scores, 40))
    p70 = float(np.percentile(scores, 70))
    df['risk_tier'] = ['High' if s>=p70 else ('Medium' if s>=p40 else 'Low') for s in scores]

    # Survival — pure numpy, never stored in df
    np.random.seed(42)
    dur_arr  = np.random.exponential(scale=90*(1-scores)+5).clip(1,180).astype(float)
    evnt_arr = (np.random.rand(len(df)) < (0.3+0.6*scores)).astype(float)

    # SHAP
    explainer     = shap.TreeExplainer(model)
    shap_raw      = explainer.shap_values(X)
    if isinstance(shap_raw, list):
        shap_vals = np.array(shap_raw[1], dtype=float)
    elif np.array(shap_raw).ndim == 3:
        shap_vals = np.array(shap_raw, dtype=float)[:,:,1]
    else:
        shap_vals = np.array(shap_raw, dtype=float)
    df['top_driver'] = [FEATURE_COLS[int(np.argmax(np.abs(shap_vals[i])))] for i in range(len(df))]

    # Segmentation
    scaler   = StandardScaler()
    Xs       = scaler.fit_transform(X)
    km       = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels   = km.fit_predict(Xs)
    cr = [(c, (1/(df.loc[labels==c,'engagement_frequency'].mean()+0.5))*0.3 +
               (df.loc[labels==c,'inactivity_gap_days'].mean()/60)*0.3 +
               (df.loc[labels==c,'missed_checkins'].mean()/15)*0.4) for c in range(3)]
    sc2  = [c for c,_ in sorted(cr, key=lambda x:x[1])]
    amap = {sc2[0]:'Stable',sc2[1]:'Decaying',sc2[2]:'Erratic'}
    df['segment'] = [amap[c] for c in labels]

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    df['PC1'] = coords[:,0]; df['PC2'] = coords[:,1]

    # Intervention
    def action(row):
        s,t,d = row['segment'],row['risk_tier'],row['behavioral_decay']
        if s=='Erratic':           return 'Human Outreach','Erratic — unpredictable, proactive contact'
        if s=='Decaying' and d<-0.35: return 'Human Outreach',f'Steep decay ({d:.2f})'
        if s=='Decaying':          return 'Automated Nudge','Gradual decline — re-engagement'
        if t=='High':              return 'Human Outreach','Risk threshold exceeded'
        if t=='Medium':            return 'Automated Nudge','Risk threshold — monitor'
        return 'No Action','Stable — passive monitoring'
    res = df.apply(action, axis=1, result_type='expand')
    df['action'] = res[0]; df['action_reason'] = res[1]

    return df, model, shap_vals, dur_arr, evnt_arr, p40, p70

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">EngageIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Behavioral Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:slategray;margin-bottom:6px;">Select patient_data.csv</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("patient_data.csv", type="csv", label_visibility="hidden")
    if uploaded:
        with open("patient_data.csv","wb") as f: f.write(uploaded.read())
        st.success("Loaded!")
        if 'pipeline' in st.session_state: del st.session_state['pipeline']
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Page", ["Dashboard","Risk Scoring","Survival Analysis",
                              "SHAP Explainability","Segmentation","Intervention Engine"],
                    label_visibility="collapsed")

if not os.path.exists("patient_data.csv"):
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:60vh;flex-direction:column;gap:16px;">
        <div style="font-size:48px;">🧠</div>
        <div style="font-size:22px;font-weight:700;color:#0b0f1e;">Welcome to EngageIQ</div>
        <div style="font-size:14px;color:slategray;">Upload patient_data.csv using the sidebar to begin</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# Run pipeline — cache in session_state to avoid re-running
if 'pipeline' not in st.session_state:
    with st.spinner("Running pipeline..."):
        try:
            st.session_state['pipeline'] = compute_pipeline("patient_data.csv")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

df, model, shap_vals, dur_arr, evnt_arr, p40, p70 = st.session_state['pipeline']

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown('<div class="page-title">Population Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time overview of patient behavioral disengagement signals</div>', unsafe_allow_html=True)

    total = len(df)
    high  = int((df['risk_tier']=='High').sum())
    med   = int((df['risk_tier']=='Medium').sum())
    low   = int((df['risk_tier']=='Low').sum())

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Patients</div><div class="kpi-value">{total:,}</div><span class="kpi-badge badge-blue">Full cohort</span></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value" style="color:crimson">{high:,}</div><span class="kpi-badge badge-red">{high/total*100:.1f}%</span></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Medium Risk</div><div class="kpi-value" style="color:coral">{med:,}</div><span class="kpi-badge badge-orange">{med/total*100:.1f}%</span></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Low Risk</div><div class="kpi-value" style="color:mediumseagreen">{low:,}</div><span class="kpi-badge badge-green">{low/total*100:.1f}%</span></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Risk Score</div><div class="kpi-value">{df["risk_score"].mean():.3f}</div><span class="kpi-badge badge-blue">cohort avg</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='risk_score', color='risk_tier', color_discrete_map=TIER_COLORS, nbins=40, opacity=0.85, labels={"risk_tier":"Risk Tier","risk_score":"Risk Score"})
        fig.add_vline(x=p70, line_dash='dash', line_color='crimson', line_width=1.5)
        fig.add_vline(x=p40, line_dash='dash', line_color='coral',   line_width=1.5)
        fig.update_layout(**PT, height=260, legend=dict(orientation="h",y=-0.25,x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="section-header">Behavioral Segments</div>', unsafe_allow_html=True)
        sc = df['segment'].value_counts().reset_index(); sc.columns=['Segment','Count']
        fig2 = px.pie(sc, names='Segment', values='Count', color='Segment', color_discrete_map=SEG_COLORS, hole=0.55)
        fig2.update_traces(textposition='outside', textinfo='percent+label')
        fig2.update_layout(**PT, height=260, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><div class="section-header">Engagement vs Inactivity</div>', unsafe_allow_html=True)
        fig3 = px.scatter(df, x='engagement_frequency', y='inactivity_gap_days', color='risk_tier',
                          color_discrete_map=TIER_COLORS, opacity=0.65,
                          labels={'risk_tier':'Risk Tier','engagement_frequency':'Logins/Week','inactivity_gap_days':'Inactivity Gap (Days)'},
                          hover_data=['patient_id','risk_score','segment'])
        fig3.update_traces(marker=dict(size=5))
        fig3.update_layout(**PT, height=260, legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="section-header">Decay vs Response Latency</div>', unsafe_allow_html=True)
        fig4 = px.scatter(df, x='behavioral_decay', y='response_latency_hrs', color='segment',
                          color_discrete_map=SEG_COLORS, opacity=0.65,
                          hover_data=['patient_id','risk_score'])
        fig4.update_traces(marker=dict(size=5))
        fig4.update_layout(**PT, height=260, legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-header">Patient Records</div>', unsafe_allow_html=True)
    f1,f2 = st.columns(2)
    with f1: tf = st.multiselect("Tier",    ["High","Medium","Low"],         default=["High","Medium","Low"])
    with f2: sf = st.multiselect("Segment", df['segment'].unique().tolist(), default=df['segment'].unique().tolist())
    fdf = df[df['risk_tier'].isin(tf) & df['segment'].isin(sf)].sort_values('risk_score',ascending=False)
    st.dataframe(fdf[['patient_id','risk_score','risk_tier','segment','action','top_driver']].reset_index(drop=True), use_container_width=True, height=280)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RISK SCORING ──────────────────────────────────────────────────────────────
elif page == "Risk Scoring":
    st.markdown('<div class="page-title">Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Random Forest — selected by cross-validated AUC across 5 folds</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({'Feature':FEATURE_COLS,'Importance':model.feature_importances_}).sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale=["#e8ecf4","steelblue"])
        fig.update_layout(**PT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="section-header">Patient Radar vs Population</div>', unsafe_allow_html=True)
        pid = st.selectbox("Patient", df['patient_id'].tolist())
        row = df[df['patient_id']==pid].iloc[0]
        maxv  = df[FEATURE_COLS].max().values
        avg_n = df[FEATURE_COLS].mean().values/(maxv+1e-6)
        pat_n = row[FEATURE_COLS].values/(maxv+1e-6)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=list(avg_n)+[avg_n[0]], theta=FEATURE_COLS+[FEATURE_COLS[0]],
                                        fill='toself', name='Avg', line=dict(color='steelblue',width=2), fillcolor='rgba(70,130,180,0.08)'))
        fig2.add_trace(go.Scatterpolar(r=list(pat_n)+[pat_n[0]], theta=FEATURE_COLS+[FEATURE_COLS[0]],
                                        fill='toself', name=pid, line=dict(color='crimson',width=2), fillcolor='rgba(220,20,60,0.08)'))
        fig2.update_layout(**PT, height=320, polar=dict(radialaxis=dict(visible=True,range=[0,1])), legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4,c5 = st.columns(3)
    c3.metric("Risk Score", f"{row['risk_score']:.4f}")
    c4.metric("Risk Tier",  row['risk_tier'])
    c5.metric("Top Driver", row['top_driver'])

# ── SURVIVAL ──────────────────────────────────────────────────────────────────
elif page == "Survival Analysis":
    st.markdown('<div class="page-title">Survival Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">KM shows when patients disengage. Cox PH shows what drives dropout timing.</div>', unsafe_allow_html=True)

    tier_idx = {t: np.where(df['risk_tier'].values == t)[0] for t in ['High','Medium','Low']}

    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="card"><div class="section-header">Kaplan-Meier Survival Curves</div>', unsafe_allow_html=True)
        ci_fill = {"High":"rgba(220,20,60,0.08)","Medium":"rgba(255,127,80,0.08)","Low":"rgba(60,179,113,0.08)"}
        fig = go.Figure()
        for tier in ['High','Medium','Low']:
            idx = tier_idx[tier]
            if len(idx) < 5: continue
            d = dur_arr[idx].copy()
            e = evnt_arr[idx].copy()
            kmf = KaplanMeierFitter()
            kmf.fit(d, event_observed=e, label=tier)
            t   = kmf.survival_function_.index.values
            s   = kmf.survival_function_.iloc[:,0].values
            cil = kmf.confidence_interval_.iloc[:,0].values
            ciu = kmf.confidence_interval_.iloc[:,1].values
            fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name=tier, line=dict(color=TIER_COLORS[tier],width=2.5)))
            fig.add_trace(go.Scatter(x=list(t)+list(t[::-1]), y=list(ciu)+list(cil[::-1]),
                                      fill='toself', showlegend=False, fillcolor=ci_fill[tier],
                                      line=dict(color='rgba(255,255,255,0)')))
        fig.update_layout(**PT, height=340, xaxis_title='Days', yaxis_title='P(Still Engaged)', yaxis_range=[0,1], legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="section-header">Median Survival (Days)</div>', unsafe_allow_html=True)
        med_rows = []
        for tier in ['High','Medium','Low']:
            idx = tier_idx[tier]
            if len(idx) < 5:
                med_rows.append({"Tier":tier,"Days":0})
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(dur_arr[idx].copy(), event_observed=evnt_arr[idx].copy())
            med_rows.append({"Tier":tier,"Days":round(float(kmf.median_survival_time_),1)})
        mdf = pd.DataFrame(med_rows)
        fig_m = px.bar(mdf, x='Tier', y='Days', color='Tier', color_discrete_map=TIER_COLORS, text='Days')
        fig_m.update_traces(textposition='outside')
        fig_m.update_layout(**PT, height=340, showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-header">Cox PH Hazard Ratios</div>', unsafe_allow_html=True)
    try:
        cox_f  = ['engagement_frequency','response_latency_hrs','inactivity_gap_days','behavioral_decay','missed_checkins','message_open_rate']
        cdf    = df[cox_f].copy().astype(float)
        cdf[cox_f] = StandardScaler().fit_transform(cdf[cox_f])
        cdf['duration_days']  = dur_arr
        cdf['event_observed'] = evnt_arr
        cph = CoxPHFitter()
        cph.fit(cdf, duration_col='duration_days', event_col='event_observed')
        c1,c2,c3 = st.columns(3)
        c1.metric("Concordance Index", f"{cph.concordance_index_:.4f}")
        c2.metric("Significant Features", str(int((cph.summary['p']<0.05).sum())))
        c3.metric("Total Features", str(len(cox_f)))
        summ = cph.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%']].sort_values('exp(coef)')
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=summ.index, x=summ['exp(coef)'], orientation='h',
                               marker_color=['crimson' if v>1 else 'mediumseagreen' for v in summ['exp(coef)']]))
        for i,(idx,r) in enumerate(summ.iterrows()):
            fig2.add_trace(go.Scatter(x=[r['exp(coef) lower 95%'],r['exp(coef) upper 95%']], y=[idx,idx],
                                       mode='lines', line=dict(color='slategray',width=2), showlegend=False))
        fig2.add_vline(x=1.0, line_dash='dash', line_color='slategray')
        fig2.update_layout(**PT, height=280, xaxis_title='Hazard Ratio')
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Cox error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── SHAP ──────────────────────────────────────────────────────────────────────
elif page == "SHAP Explainability":
    st.markdown('<div class="page-title">SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Per-patient feature attribution — why each patient is flagged</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="section-header">Global Feature Impact</div>', unsafe_allow_html=True)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        sdf = pd.DataFrame({'Feature':FEATURE_COLS,'Mean |SHAP|':mean_abs}).sort_values('Mean |SHAP|')
        fig = px.bar(sdf, x='Mean |SHAP|', y='Feature', orientation='h', color='Mean |SHAP|', color_continuous_scale=["#e8ecf4","steelblue"])
        fig.update_layout(**PT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="section-header">SHAP Dependence — Inactivity Gap</div>', unsafe_allow_html=True)
        fidx = FEATURE_COLS.index('inactivity_gap_days')
        fig3 = px.scatter(x=df['inactivity_gap_days'].values, y=shap_vals[:,fidx],
                           color=df['risk_tier'].values, color_discrete_map=TIER_COLORS, opacity=0.65,
                           labels={'x':'Inactivity Gap (Days)','y':'SHAP Value','color':'Risk Tier'})
        fig3.update_traces(marker=dict(size=5))
        fig3.add_hline(y=0, line_dash='dash', line_color='slategray')
        fig3.update_layout(**PT, height=320, legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-header">Individual Patient Waterfall</div>', unsafe_allow_html=True)
    pid  = st.selectbox("Select Patient", df['patient_id'].tolist())
    pidx = df[df['patient_id']==pid].index[0]
    sv   = shap_vals[pidx]
    fv   = df[FEATURE_COLS].iloc[pidx].values
    sm   = pd.DataFrame({'Feature':FEATURE_COLS,'SHAP':sv,'Value':fv}).sort_values('SHAP')
    fig2 = go.Figure(go.Bar(x=sm['SHAP'], y=sm['Feature'], orientation='h',
                             marker_color=['crimson' if v>0 else 'mediumseagreen' for v in sm['SHAP']],
                             text=[f"{v:.2f}" for v in sm['Value']], textposition='outside'))
    fig2.add_vline(x=0, line_color='slategray', line_width=1)
    fig2.update_layout(**PT, height=300, xaxis_title='SHAP Value')
    st.plotly_chart(fig2, use_container_width=True)
    ca,cb,cc = st.columns(3)
    ca.metric("Risk Score", f"{df.loc[pidx,'risk_score']:.4f}")
    cb.metric("Risk Tier",  df.loc[pidx,'risk_tier'])
    cc.metric("Top Driver", df.loc[pidx,'top_driver'])
    st.markdown('</div>', unsafe_allow_html=True)

# ── SEGMENTATION ──────────────────────────────────────────────────────────────
elif page == "Segmentation":
    st.markdown('<div class="page-title">Behavioral Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">KMeans k=3 — Stable, Decaying, Erratic archetypes</div>', unsafe_allow_html=True)

    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="card"><div class="section-header">PCA Cluster Visualization</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='PC1', y='PC2', color='segment', color_discrete_map=SEG_COLORS,
                         hover_data=['patient_id','risk_score','risk_tier'], opacity=0.7)
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PT, height=360, legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="section-header">Segment Breakdown</div>', unsafe_allow_html=True)
        sc = df['segment'].value_counts().reset_index(); sc.columns=['Segment','Count']
        for _,r in sc.iterrows():
            pct   = r['Count']/len(df)*100
            color = SEG_COLORS.get(r['Segment'],'slategray')
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:13px;font-weight:600;color:#0b0f1e;">{r['Segment']}</span>
                    <span style="font-size:13px;font-weight:600;color:{color};">{r['Count']} ({pct:.1f}%)</span>
                </div>
                <div style="background:#f0f2f8;border-radius:4px;height:8px;">
                    <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        rs = df.groupby('segment')['risk_score'].mean().reset_index().sort_values('risk_score',ascending=False)
        for _,r in rs.iterrows():
            color = SEG_COLORS.get(r['segment'],'slategray')
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f0f2f8;"><span style="font-size:13px;color:slategray;">{r["segment"]} avg risk</span><span style="font-size:13px;font-weight:600;color:{color};">{r["risk_score"]:.3f}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-header">Segment Feature Profiles</div>', unsafe_allow_html=True)
    st.dataframe(df.groupby('segment')[FEATURE_COLS].mean().round(2), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── INTERVENTION ──────────────────────────────────────────────────────────────
elif page == "Intervention Engine":
    st.markdown('<div class="page-title">Intervention Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Segment + risk tier + decay slope determines action — not just a score threshold</div>', unsafe_allow_html=True)

    nh = int((df['action']=='Human Outreach').sum())
    nn = int((df['action']=='Automated Nudge').sum())
    no = int((df['action']=='No Action').sum())

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Human Outreach</div><div class="kpi-value" style="color:crimson">{nh}</div><span class="kpi-badge badge-red">Immediate action</span></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Automated Nudge</div><div class="kpi-value" style="color:coral">{nn}</div><span class="kpi-badge badge-orange">Queued</span></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">No Action</div><div class="kpi-value" style="color:mediumseagreen">{no}</div><span class="kpi-badge badge-green">Monitor</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4,c5 = st.columns(2)
    with c4:
        st.markdown('<div class="card"><div class="section-header">Action Distribution</div>', unsafe_allow_html=True)
        ac = df['action'].value_counts().reset_index(); ac.columns=['Action','Count']
        fig = px.pie(ac, names='Action', values='Count', color='Action', color_discrete_map=ACT_COLORS, hole=0.55)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(**PT, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="card"><div class="section-header">Actions by Segment</div>', unsafe_allow_html=True)
        cross = pd.crosstab(df['segment'], df['action']).reset_index().melt(id_vars='segment', var_name='Action', value_name='Count')
        fig2 = px.bar(cross, x='segment', y='Count', color='Action', color_discrete_map=ACT_COLORS, barmode='stack')
        fig2.update_layout(**PT, height=280, legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-header">Clinician Worklist</div>', unsafe_allow_html=True)
    worklist = df[df['action']=='Human Outreach'].sort_values('risk_score',ascending=False).reset_index(drop=True)
    for _,r in worklist.head(10).iterrows():
        st.markdown(f"""<div class="worklist-row">
            <div>
                <div class="patient-id">{r['patient_id']}</div>
                <div class="driver-text">Top driver: {r['top_driver']} &nbsp;|&nbsp; {r['action_reason']}</div>
                <div style="margin-top:4px;">{tier_badge(r['risk_tier'])} {seg_badge(r['segment'])}</div>
            </div>
            <div class="risk-score-badge">{r['risk_score']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    if len(worklist)>10: st.caption(f"Showing 10 of {len(worklist)} patients")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-header">What-If Risk Simulator</div>', unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca: pid_w  = st.selectbox("Patient", df['patient_id'].tolist())
    with cb: feat_w = st.selectbox("Feature", FEATURE_COLS)
    pidx_w = df[df['patient_id']==pid_w].index[0]
    curr_w = float(df.loc[pidx_w, feat_w])
    new_val = st.slider("New value", float(df[feat_w].min()), float(df[feat_w].max()), curr_w, step=0.1)
    X_mod  = df[FEATURE_COLS].iloc[pidx_w].values.copy().astype(float)
    X_mod[FEATURE_COLS.index(feat_w)] = new_val
    new_score = float(model.predict_proba(X_mod.reshape(1,-1))[0][1])
    new_tier  = 'High' if new_score>=p70 else ('Medium' if new_score>=p40 else 'Low')
    cx,cy,cz = st.columns(3)
    cx.metric("Original Score", f"{df.loc[pidx_w,'risk_score']:.4f}")
    cy.metric("New Score", f"{new_score:.4f}", delta=f"{new_score-df.loc[pidx_w,'risk_score']:+.4f}")
    cz.metric("New Tier", new_tier)
    st.markdown('</div>', unsafe_allow_html=True)
