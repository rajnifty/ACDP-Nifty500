import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import io
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ACDP Nifty 500 Analysis", 
    layout="wide", 
    page_icon="🇮🇳",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DYNAMIC NIFTY 500 FETCHING (NSE OFFICIAL)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400) # Cache for 24 hours to avoid spamming NSE
def get_nifty_500_assets():
    """Fetches the latest Nifty 500 constituents from the official NSE India archives."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "text/csv"
    }
    
    assets = {}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Read the CSV content
        df_nse = pd.read_csv(io.StringIO(response.text))
        
        # Build dictionary: "Company Name" -> "SYMBOL.NS"
        for _, row in df_nse.iterrows():
            company_name = str(row['Company Name']).strip()
            symbol = str(row['Symbol']).strip()
            assets[company_name] = f"{symbol}.NS"
            
    except Exception as e:
        st.error(f"⚠️ Could not fetch live Nifty 500 list from NSE. Please check connection.")
        # Fallback to a small sample if NSE blocks the request
        assets = {
            "Reliance Industries Ltd.": "RELIANCE.NS",
            "Tata Consultancy Services Ltd.": "TCS.NS",
            "HDFC Bank Ltd.": "HDFCBANK.NS",
            "Infosys Ltd.": "INFY.NS",
            "ICICI Bank Ltd.": "ICICIBANK.NS"
        }
        
    # Always ensure the Index is present for benchmark comparison
    assets["Nifty 50 Index"] = "^NSEI"
    return assets

# -----------------------------------------------------------------------------
# 3. CSS & STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #fbfaff;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9d5ff;
        box-shadow: 4px 0 15px rgba(139, 92, 246, 0.03);
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h1 {
        color: #4B365F;
        font-weight: 700;
        letter-spacing: -1px;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #6b5b95;
    }
    
    /* Canvas Container */
    .canvas-container {
        background: linear-gradient(135deg, #ffffff 0%, #f3e8ff 100%);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #d8b4fe;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.08);
        margin-bottom: 25px;
    }
    .big-title {
        font-family: 'Arial Black', sans-serif;
        font-size: 3em;
        text-transform: uppercase;
        background: -webkit-linear-gradient(top, #4B365F, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #8b5cf6;
        font-family: 'Courier New', monospace;
        text-align: center;
        font-size: 1em;
        font-weight: 600;
    }

    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        overflow: hidden;
        background-color: white;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        color: #4B365F;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #f3e8ff;
        color: #7c3aed;
        border: 1px solid #d8b4fe;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ANALYTICS ENGINE (Optimized for 500 Stocks)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_and_analyze_data(assets_dict):
    """
    Fetches bulk data for all assets to prevent timeouts.
    Calculates Momentum Score & Volatility without UI elements to keep caching clean.
    """
    stats_data = []
    history_dict = {}
    tickers = list(assets_dict.values())
    
    # Bulk Download
    df_bulk = yf.download(tickers, period="2y", threads=True, progress=False)
    
    if df_bulk.empty:
        return pd.DataFrame(), {}
        
    if isinstance(df_bulk.columns, pd.MultiIndex):
        df_close = df_bulk['Close']
    else:
        df_close = df_bulk

    df_close.index = df_close.index.tz_localize(None)
    
    for name, ticker in assets_dict.items():
        try:
            if ticker not in df_close.columns:
                continue
                
            hist = df_close[ticker].dropna()
            if len(hist) < 260:  # Need roughly a year of trading days
                continue
                
            current_price = float(hist.iloc[-1])
            
            def get_price_lag(days):
                target_date = datetime.now() - timedelta(days=days)
                idx = hist.index.get_indexer([target_date], method='nearest')[0]
                return float(hist.iloc[idx])

            # ACDP Momentum Logic
            r12 = (current_price - get_price_lag(365)) / get_price_lag(365)
            r6  = (current_price - get_price_lag(180)) / get_price_lag(180)
            r3  = (current_price - get_price_lag(90))  / get_price_lag(90)
            r1  = (current_price - get_price_lag(30))  / get_price_lag(30)
            
            avg_score = (r12 + r6 + r3 + r1) / 4
            volatility = hist.pct_change().dropna().std() * np.sqrt(252)
            
            stats_data.append({
                "Asset": name, 
                "Price": current_price, 
                "Score": avg_score, 
                "Vol": volatility
            })
            history_dict[name] = hist
        except Exception as e:
            continue
            
    df_stats = pd.DataFrame(stats_data)
    if not df_stats.empty:
        # Sort by Score to determine Rank
        df_stats = df_stats.sort_values("Score", ascending=False).reset_index(drop=True)
        df_stats['Rank'] = df_stats.index + 1
    
    return df_stats, history_dict

@st.cache_data(ttl=3600)
def calculate_correlation(history_dict):
    """Calculates correlation matrix from price history dictionary."""
    if not history_dict:
        return pd.DataFrame()
    
    df_prices = pd.DataFrame(history_dict)
    # Forward fill missing dates for correlation accuracy across assets
    df_prices = df_prices.ffill().dropna()
    df_returns = df_prices.pct_change().dropna()
    return df_returns.corr()

# -----------------------------------------------------------------------------
# 5. SIDEBAR & HEADER
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([6, 1])
with header_col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)

with st.sidebar:
    st.title("ACDP")
    st.caption("Automated Concentrated\nDiversified Portfolio")
    st.write("---")
    st.info("Live parsing of 500 Nifty constituents. Calculating optimal quantitative rotations.")
    st.write("---")
    st.caption(f"Last Update:\n{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# -----------------------------------------------------------------------------
# 6. MAIN APP LOGIC
# -----------------------------------------------------------------------------

# Header
st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
st.markdown('<div class="big-title">NIFTY 500: TOP 10 LEADERS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">QUANTITATIVE RANKING & RISK ARCHITECTURE</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Smart Loading Status
with st.status("Initializing ACDP Engine...", expanded=True) as status_box:
    st.write("🔍 Syncing Nifty 500 symbols from NSE...")
    nifty500_assets = get_nifty_500_assets()
    
    st.write("📥 Downloading bulk market data (250,000+ data points)...")
    df_stats, history_dict = fetch_and_analyze_data(nifty500_assets)
    
    if not df_stats.empty:
        status_box.update(label="Analysis Complete! Displaying Leaders.", state="complete", expanded=False)
    else:
        status_box.update(label="Error: No data retrieved. Check internet connection.", state="error")

if not df_stats.empty:
    
    # --- FILTERING LOGIC: Top 10 Stocks + Index ---
    index_name = "Nifty 50 Index"
    
    df_index = df_stats[df_stats['Asset'] == index_name]
    df_stocks = df_stats[df_stats['Asset'] != index_name]
    
    # Isolate the exact Top 10 from the remaining 500 pool
    df_top10 = df_stocks.head(10) 
    
    # Combine Top 10 and Index
    display_df = pd.concat([df_top10, df_index]).reset_index(drop=True)
    
    # Build correlation engine using ONLY the filtered assets to prevent UI freezing
    assets_to_keep = display_df['Asset'].tolist()
    filtered_history = {k: v for k, v in history_dict.items() if k in assets_to_keep}
    df_corr = calculate_correlation(filtered_history)

    # --- TABS FOR ANALYSIS ---
    tab1, tab2 = st.tabs(["🏆 Performance Heatmap", "🧩 Risk Architecture"])
    
    # --- TAB 1: RANKING SYSTEM ---
    with tab1:
        st.caption("ACDP Algorithm - Nifty 500 Top Percentile View")
        
        heatmap_df = display_df[['Rank', 'Asset', 'Price', 'Vol', 'Score']].copy()
        heatmap_df = heatmap_df.set_index('Rank')

        st.dataframe(
            heatmap_df.style
            .format({
                'Price': '₹ {:,.2f}',
                'Score': '{:+.2%}',
                'Vol': '{:.2%}'     
            })
            .background_gradient(
                cmap='RdYlGn_r', 
                subset=['Price']
            )
            # Highlight the Index row in light blue
            .apply(lambda x: ['background-color: #e0e7ff; font-weight: bold' if x['Asset'] == index_name else '' for i in x], axis=1)
            .set_properties(**{'text-align': 'center', 'font-weight': '600', 'color': '#2e2e2e'})
            .set_table_styles([{
                'selector': 'th',
                'props': [
                    ('text-align', 'center'), 
                    ('background-color', '#f3e8ff'), 
                    ('color', '#4B365F'),
                    ('font-size', '1.1em')
                ]
            }]),
            use_container_width=True,
            height=500
        )

    # --- TAB 2: RISK ANALYSIS ---
    with tab2:
        st.subheader("Leaderboard Correlation & Risk Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Correlation Matrix (1 Year)**")
            if not df_corr.empty:
                fig_corr = px.imshow(
                    df_corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r", 
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#4B365F"),
                    height=650
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
        with col2:
            st.markdown("**Annualized Volatility Profile**")
            
            fig_vol = px.scatter(
                display_df,
                x="Vol",
                y="Score",
                text="Asset",
                size=[15]*len(display_df),
                color="Score",
                color_continuous_scale="Viridis",
                labels={"Vol": "Volatility (Risk)", "Score": "Momentum (Reward)"}
            )
            fig_vol.update_traces(textposition='top center')
            fig_vol.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(240,240,255,0.5)",
                font=dict(color="#4B365F"),
                height=650,
                showlegend=False
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            st.success("💡 **Note:** Assets clustering in the top-left (High Score, Low Vol) indicate superior risk-adjusted efficiency compared to the benchmark index.")

# Footer
st.write("---")
st.markdown(
    """
    <div style='text-align: center; color: #887bb0; font-size: 0.8em; font-family: sans-serif;'>
        ACDP Framework • Built for Rajan Yadav • Source: Investopedia Analytics Logic
    </div>
    """, 
    unsafe_allow_html=True
)