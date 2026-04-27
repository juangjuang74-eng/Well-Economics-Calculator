import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Well Economics | Portfolio",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="stSidebar"] { background: #fafaf8; border-right: 1px solid #e5e7e0; }
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 1.5rem; }
.kpi { background: white; border: 1px solid #e5e7e0; border-radius: 10px; padding: 14px 18px; }
.kpi-label { font-size: 11px; font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 6px; }
.kpi-val { font-size: 24px; font-weight: 600; font-family: 'DM Mono', monospace; }
.kpi-sub { font-size: 11px; color: #9ca3af; margin-top: 3px; }
.kpi-pos { color: #16a34a; }
.kpi-neg { color: #dc2626; }
.kpi-neu { color: #1d4ed8; }
.section-title { font-family: 'DM Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: #6b7280; border-bottom: 1px solid #e5e7e0; padding-bottom: 6px; margin-bottom: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Economics functions ─────────────────────────────────────────────
def arps_hyperbolic(t, qi, di, b):
    if b == 0: return qi * np.exp(-di * t)
    return qi / (1 + b * di * t) ** (1 / b)

def npv_calc(cash_flows, discount_rate, freq=12):
    """NPV with monthly cash flows, annual discount rate."""
    r_monthly = (1 + discount_rate) ** (1/freq) - 1
    t = np.arange(len(cash_flows))
    pv = np.array(cash_flows) / (1 + r_monthly) ** t
    return np.sum(pv)

def irr_calc(cash_flows, guess=0.1, tol=1e-6, max_iter=500):
    """Monthly IRR → annualized."""
    r = guess / 12
    for _ in range(max_iter):
        t = np.arange(len(cash_flows))
        f = np.sum(np.array(cash_flows) / (1 + r) ** t)
        df = np.sum(-t * np.array(cash_flows) / (1 + r) ** (t + 1))
        if abs(df) < 1e-12: break
        r_new = r - f / df
        if abs(r_new - r) < tol: break
        r = max(r_new, -0.9999)
    return (1 + r) ** 12 - 1

def payback_period(cum_cf):
    """Month index where cumulative CF turns positive."""
    for i, v in enumerate(cum_cf):
        if v >= 0: return i
    return None

def build_cashflow(qi, di, b, oil_price, gas_price, water_disposal,
                   wi, nri, severance_tax, opex_fixed, opex_var,
                   capex, months, q_aban=20):
    t = np.arange(months)
    q_oil = arps_hyperbolic(t, qi, di, b)
    q_oil = np.where(q_oil < q_aban, 0, q_oil)
    
    gross_revenue = q_oil * 30.44 * oil_price  # bopd × days/mo × $/bbl
    nri_revenue   = gross_revenue * nri
    sev_tax       = nri_revenue * severance_tax
    net_revenue   = nri_revenue - sev_tax
    
    opex_total    = opex_fixed + q_oil * 30.44 * opex_var
    water_cost    = q_oil * 30.44 * water_disposal  # simplified WOR=0 for demo
    
    net_cf = net_revenue - opex_total * wi - water_cost * wi
    
    # CAPEX in month 0
    net_cf[0] -= capex
    
    df = pd.DataFrame({
        'Month': t + 1,
        'q_oil_bopd': q_oil.round(1),
        'Gross_Revenue': gross_revenue.round(0),
        'NRI_Revenue': nri_revenue.round(0),
        'Severance_Tax': sev_tax.round(0),
        'OPEX': (opex_total * wi).round(0),
        'Water_Cost': (water_cost * wi).round(0),
        'Net_CF': net_cf.round(0),
        'Cum_CF': net_cf.cumsum().round(0),
    })
    return df

def sensitivity_analysis(base_params, param_names, variations=(-30,-20,-10,0,10,20,30)):
    """±% variation on each param, return NPV change."""
    results = {}
    base_df = build_cashflow(**base_params)
    base_npv = npv_calc(base_df['Net_CF'].values, 0.10)
    
    for pname in param_names:
        npvs = []
        for pct in variations:
            p2 = base_params.copy()
            p2[pname] = base_params[pname] * (1 + pct/100)
            df2 = build_cashflow(**p2)
            npvs.append(npv_calc(df2['Net_CF'].values, 0.10) - base_npv)
        results[pname] = npvs
    return results, base_npv

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Well Economics")
    st.markdown("*E&P Financial Modeling*")
    st.markdown("---")

    st.markdown("### Production")
    qi   = st.number_input("Initial Rate qi (bopd)", value=800, step=50)
    di   = st.number_input("Decline Rate di (%/mo)", value=6.0, step=0.5) / 100
    b    = st.number_input("b factor (Arps)", value=0.8, min_value=0.0, max_value=2.0, step=0.1)
    proj_months = st.slider("Project life (months)", 24, 240, 120, 12)
    q_aban = st.number_input("Abandonment rate (bopd)", value=20)

    st.markdown("---")
    st.markdown("### Pricing")
    oil_price     = st.number_input("Oil price ($/bbl)", value=75.0, step=1.0)
    gas_price     = st.number_input("Gas price ($/MSCF)", value=3.5, step=0.1)
    water_disp    = st.number_input("Water disposal ($/bbl)", value=1.5, step=0.1)

    st.markdown("---")
    st.markdown("### Ownership & Tax")
    wi  = st.number_input("Working Interest (WI %)", value=100.0, step=5.0) / 100
    nri = st.number_input("Net Revenue Interest (NRI %)", value=80.0, step=1.0) / 100
    sev = st.number_input("Severance Tax (%)", value=4.6, step=0.1) / 100

    st.markdown("---")
    st.markdown("### CAPEX & OPEX")
    capex     = st.number_input("CAPEX (USD)", value=3_000_000, step=100_000)
    opex_fix  = st.number_input("Fixed OPEX ($/month)", value=8000, step=500)
    opex_var  = st.number_input("Variable OPEX ($/bbl)", value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("### Hurdle Rate")
    discount  = st.number_input("Discount rate (%/yr)", value=10.0, step=0.5) / 100

# ── Compute ─────────────────────────────────────────────────────────
base_params = dict(qi=qi, di=di, b=b, oil_price=oil_price, gas_price=gas_price,
                   water_disposal=water_disp, wi=wi, nri=nri, severance_tax=sev,
                   opex_fixed=opex_fix, opex_var=opex_var, capex=capex,
                   months=proj_months, q_aban=q_aban)

df = build_cashflow(**base_params)
npv = npv_calc(df['Net_CF'].values, discount)
try:
    irr = irr_calc(df['Net_CF'].values)
except: irr = None
pb = payback_period(df['Cum_CF'].values)
eur_mstb = df['q_oil_bopd'].sum() * 30.44 / 1e6
payout_mo = pb if pb else proj_months

# ── KPIs ────────────────────────────────────────────────────────────
st.markdown("# Well Economics Calculator")
st.markdown("*NPV · IRR · Payback Period · Cashflow · Tornado Sensitivity*")

col1, col2, col3, col4 = st.columns(4)
npv_color = "kpi-pos" if npv > 0 else "kpi-neg"
irr_color = "kpi-pos" if irr and irr > discount else "kpi-neg"

with col1:
    st.metric("NPV @ "+f"{discount*100:.1f}%", f"${npv/1e6:.2f}MM",
              "Economic ✓" if npv > 0 else "Uneconomic ✗")
with col2:
    st.metric("IRR", f"{irr*100:.1f}%" if irr else "N/A",
              f"Hurdle: {discount*100:.1f}%")
with col3:
    st.metric("Payback", f"{payout_mo} mo" if pb else "> project life",
              f"{payout_mo/12:.1f} years" if pb else "No payout")
with col4:
    st.metric("EUR", f"{eur_mstb:.3f} MMSTB",
              f"≈ ${eur_mstb*1000*oil_price*nri*(1-sev)/1e6:.1f}MM gross NRI")

st.markdown("---")

# ── Cashflow chart ───────────────────────────────────────────────────
col_cf, col_prod = st.columns(2)

with col_cf:
    st.markdown("**Monthly & Cumulative Cashflow**")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = df['Net_CF'].apply(lambda x: '#16a34a' if x >= 0 else '#dc2626')
    fig.add_trace(go.Bar(x=df['Month'], y=df['Net_CF']/1e3,
                         name='Net CF (k$)', marker_color=colors,
                         opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Month'], y=df['Cum_CF']/1e6,
                              name='Cumulative ($MM)', line=dict(color='#1d4ed8', width=2),
                              mode='lines'), secondary_y=True)
    fig.add_hline(y=0, line_color='#6b7280', line_dash='dash', line_width=1)
    fig.update_layout(template='plotly_white', height=320,
                      legend=dict(orientation='h', y=1.08, x=0),
                      margin=dict(t=40,b=40,l=60,r=60),
                      paper_bgcolor='white', plot_bgcolor='#f9fafb')
    fig.update_yaxes(title_text="Net CF (k$/mo)", secondary_y=False,
                     gridcolor='#e5e7e0', tickfont_size=10)
    fig.update_yaxes(title_text="Cumulative ($MM)", secondary_y=True,
                     gridcolor='#e5e7e0', tickfont_size=10)
    st.plotly_chart(fig, use_container_width=True)

with col_prod:
    st.markdown("**Production & Revenue Breakdown**")
    active = df[df['q_oil_bopd'] > 0]
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=active['Month'], y=active['q_oil_bopd'],
                               name='Oil Rate (bopd)', fill='tozeroy',
                               line=dict(color='#92400e', width=2),
                               fillcolor='rgba(146,64,14,0.1)'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=active['Month'], y=active['NRI_Revenue']/1e3,
                               name='NRI Revenue (k$/mo)', line=dict(color='#059669', width=1.5, dash='dot'),
                               mode='lines'), secondary_y=True)
    fig2.update_layout(template='plotly_white', height=320,
                       legend=dict(orientation='h', y=1.08, x=0),
                       margin=dict(t=40,b=40,l=60,r=60),
                       paper_bgcolor='white', plot_bgcolor='#f9fafb')
    fig2.update_yaxes(title_text="Rate (bopd)", secondary_y=False, gridcolor='#e5e7e0', tickfont_size=10)
    fig2.update_yaxes(title_text="Revenue (k$/mo)", secondary_y=True, gridcolor='#e5e7e0', tickfont_size=10)
    st.plotly_chart(fig2, use_container_width=True)

# ── Revenue waterfall ────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Revenue Waterfall — Lifetime Totals**")

total_gross  = df['Gross_Revenue'].sum()
total_nri    = df['NRI_Revenue'].sum()
total_sev    = df['Severance_Tax'].sum()
total_opex   = df['OPEX'].sum()
total_water  = df['Water_Cost'].sum()
total_net_cf = df['Net_CF'].sum() + capex  # ex-capex

labels = ['Gross Revenue','→ NRI Revenue','- Severance Tax','- OPEX','- Water Disposal','= Net Revenue (ex-CAPEX)','- CAPEX','= Total Net CF']
values = [total_gross, total_nri - total_gross, -total_sev, -total_opex, -total_water,
          0, -capex, 0]
measures = ['absolute','relative','relative','relative','relative','total','relative','total']
text = [f"${v/1e6:.2f}MM" for v in [total_gross, total_nri-total_gross, -total_sev,
                                     -total_opex,-total_water,total_net_cf,-capex,df['Net_CF'].sum()]]

fig_wf = go.Figure(go.Waterfall(
    name='Economics', orientation='v',
    measure=measures, x=labels, y=[total_gross, total_nri-total_gross, -total_sev,
                                    -total_opex, -total_water, total_net_cf-total_gross+total_sev+total_opex+total_water,
                                    -capex, df['Net_CF'].sum()-total_net_cf+total_gross-total_sev-total_opex-total_water],
    text=text, textposition='outside',
    connector=dict(line=dict(color='#e5e7e0', width=1)),
    increasing=dict(marker_color='#16a34a'),
    decreasing=dict(marker_color='#dc2626'),
    totals=dict(marker_color='#1d4ed8')
))
fig_wf.update_layout(template='plotly_white', height=360,
                     margin=dict(t=30,b=60,l=60,r=20),
                     yaxis_title='USD', paper_bgcolor='white', plot_bgcolor='#f9fafb',
                     yaxis=dict(gridcolor='#e5e7e0', tickformat='$,.0f'))
st.plotly_chart(fig_wf, use_container_width=True)

# ── Tornado Sensitivity ──────────────────────────────────────────────
st.markdown("---")
st.markdown("**Tornado Chart — NPV Sensitivity (±20%)**")

sens_params = {
    'oil_price': 'Oil Price',
    'qi': 'Initial Rate (qi)',
    'capex': 'CAPEX',
    'di': 'Decline Rate (di)',
    'opex_var': 'Variable OPEX',
    'nri': 'NRI',
}

variations_pct = [-20, 20]
tornado_data = []
for pname, label in sens_params.items():
    row = {'param': label}
    for pct in variations_pct:
        p2 = base_params.copy()
        p2[pname] = base_params[pname] * (1 + pct/100)
        df2 = build_cashflow(**p2)
        npv2 = npv_calc(df2['Net_CF'].values, discount)
        row[f'pct_{pct}'] = (npv2 - npv) / 1e6
    row['range'] = abs(row['pct_20'] - row['pct_-20'])
    tornado_data.append(row)

tornado_df = pd.DataFrame(tornado_data).sort_values('range')

fig_t = go.Figure()
fig_t.add_trace(go.Bar(
    y=tornado_df['param'], x=tornado_df['pct_-20'],
    name='-20%', orientation='h', marker_color='#fca5a5', marker_line_width=0
))
fig_t.add_trace(go.Bar(
    y=tornado_df['param'], x=tornado_df['pct_20'],
    name='+20%', orientation='h', marker_color='#86efac', marker_line_width=0
))
fig_t.add_vline(x=0, line_color='#374151', line_width=1.5)
fig_t.update_layout(
    template='plotly_white', barmode='overlay', height=320,
    xaxis_title='ΔNPV ($MM)', yaxis_title='',
    legend=dict(orientation='h', y=1.1, x=0),
    margin=dict(t=40,b=50,l=150,r=60),
    paper_bgcolor='white', plot_bgcolor='#f9fafb',
    xaxis=dict(gridcolor='#e5e7e0', tickformat='$,.1f', ticksuffix='MM')
)
st.plotly_chart(fig_t, use_container_width=True)

# ── Monthly Table ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Monthly Cashflow Table**")

display_df = df[['Month','q_oil_bopd','Gross_Revenue','NRI_Revenue',
                  'Severance_Tax','OPEX','Net_CF','Cum_CF']].copy()
display_df.columns = ['Month','Rate (bopd)','Gross Rev ($)','NRI Rev ($)',
                       'Sev Tax ($)','OPEX ($)','Net CF ($)','Cum CF ($)']
for col in ['Gross Rev ($)','NRI Rev ($)','Sev Tax ($)','OPEX ($)','Net CF ($)','Cum CF ($)']:
    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

st.dataframe(display_df, use_container_width=True, hide_index=True, height=280)

# ── Export ───────────────────────────────────────────────────────────
st.markdown("---")
col_e1, col_e2 = st.columns([1,3])
with col_e1:
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Export Cashflow CSV", csv, "well_economics.csv", "text/csv")

st.caption("Built with petropt · Streamlit · Plotly | Portfolio by [your name] · Methodology: Arps (1945), standard WI/NRI/Severance framework")
