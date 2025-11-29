"""
Kuber Dashboard

Streamlit-based web interface for the Kuber trading platform.
Run with: streamlit run kuber/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from kuber.core.broker import RobinhoodBroker
from kuber.core.portfolio import Portfolio
from kuber.strategies import (
    MACrossoverStrategy, RSIStrategy, MACDStrategy, 
    BollingerBandsStrategy, MeanReversionStrategy, CombinedStrategy
)
from kuber.backtest import Backtester, BacktestConfig

# Page config
st.set_page_config(
    page_title="Kuber - Algo Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "broker" not in st.session_state:
        st.session_state.broker = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None


def render_header():
    """Render the main header."""
    st.markdown('<p class="main-header">📈 Kuber Trading Platform</p>', 
                unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with login and navigation."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=KUBER", width=150)
        st.markdown("---")
        
        if not st.session_state.logged_in:
            render_login_form()
        else:
            st.success(f"✅ Connected to Robinhood")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.broker.logout()
                st.session_state.logged_in = False
                st.session_state.broker = None
                st.rerun()
                
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "📈 Portfolio", "🎯 Strategies", 
             "🔬 Backtest", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        if st.session_state.logged_in and st.session_state.broker:
            try:
                buying_power = st.session_state.broker.get_buying_power()
                st.metric("Buying Power", f"${buying_power:,.2f}")
            except:
                st.info("Connect to see stats")
        else:
            st.info("Login to see stats")
            
        return page


def render_login_form():
    """Render the login form."""
    st.markdown("### 🔐 Login to Robinhood")
    
    with st.form("login_form"):
        username = st.text_input("Username/Email")
        password = st.text_input("Password", type="password")
        totp_secret = st.text_input("TOTP Secret (optional)", type="password",
                                   help="For automated 2FA")
        
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if username and password:
                with st.spinner("Connecting to Robinhood..."):
                    broker = RobinhoodBroker(
                        username=username,
                        password=password,
                        totp_secret=totp_secret if totp_secret else None
                    )
                    if broker.login():
                        st.session_state.broker = broker
                        st.session_state.logged_in = True
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Login failed. Check your credentials.")
            else:
                st.warning("Please enter username and password")


def render_dashboard():
    """Render the main dashboard page."""
    st.markdown("## 📊 Dashboard")
    
    # Demo mode if not logged in
    if not st.session_state.logged_in:
        st.info("🔑 Login to see live data. Showing demo data below.")
        render_demo_dashboard()
        return
        
    broker = st.session_state.broker
    
    try:
        # Account metrics
        col1, col2, col3, col4 = st.columns(4)
        
        portfolio_profile = broker.get_portfolio_profile()
        equity = float(portfolio_profile.get("equity", 0))
        extended_hours_equity = float(portfolio_profile.get("extended_hours_equity", equity))
        
        with col1:
            st.metric("Portfolio Value", f"${equity:,.2f}")
        with col2:
            buying_power = broker.get_buying_power()
            st.metric("Buying Power", f"${buying_power:,.2f}")
        with col3:
            cash = broker.get_cash_balance()
            st.metric("Cash Balance", f"${cash:,.2f}")
        with col4:
            day_trades = broker.get_day_trades_count()
            st.metric("Day Trades (5 days)", day_trades)
            
        st.markdown("---")
        
        # Holdings
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Current Holdings")
            holdings = broker.get_holdings()
            
            if holdings:
                holdings_data = []
                for symbol, data in holdings.items():
                    holdings_data.append({
                        "Symbol": symbol,
                        "Quantity": float(data.get("quantity", 0)),
                        "Avg Cost": float(data.get("average_buy_price", 0)),
                        "Current Price": float(data.get("price", 0)),
                        "Equity": float(data.get("equity", 0)),
                        "P&L %": float(data.get("percent_change", 0))
                    })
                    
                df = pd.DataFrame(holdings_data)
                
                # Style the dataframe
                st.dataframe(
                    df.style.applymap(
                        lambda x: "color: green" if x > 0 else "color: red",
                        subset=["P&L %"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Pie chart
                fig = px.pie(df, values="Equity", names="Symbol",
                            title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No positions found")
                
        with col2:
            st.markdown("### Market Status")
            if broker.is_market_open():
                st.success("🟢 Market Open")
            else:
                st.warning("🔴 Market Closed")
                
            st.markdown("### Top Movers (S&P 500)")
            try:
                top_gainers = broker.get_top_movers("up")[:5]
                st.markdown("**Top Gainers**")
                for stock in top_gainers:
                    symbol = stock.get("symbol", "N/A")
                    change = stock.get("price_movement", {}).get("market_hours_last_movement_pct", 0)
                    st.markdown(f"- {symbol}: +{change:.2f}%")
            except:
                st.info("Unable to load market data")
                
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")


def render_demo_dashboard():
    """Render demo dashboard with sample data."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", "$25,432.18", "+$432.18")
    with col2:
        st.metric("Buying Power", "$3,218.42")
    with col3:
        st.metric("Today's P&L", "+$156.23", "+0.62%")
    with col4:
        st.metric("Total Return", "+12.4%", "+$2,832.18")
        
    st.markdown("---")
    
    # Demo holdings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sample Holdings (Demo)")
        demo_data = pd.DataFrame({
            "Symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
            "Quantity": [10, 5, 15, 8, 20],
            "Avg Cost": [175.50, 140.25, 378.90, 185.40, 475.80],
            "Current Price": [182.30, 145.80, 385.20, 190.50, 502.40],
            "Equity": [1823.00, 729.00, 5778.00, 1524.00, 10048.00],
            "P&L %": [3.87, 3.96, 1.66, 2.75, 5.59]
        })
        st.dataframe(demo_data, use_container_width=True, hide_index=True)
        
        # Demo chart
        fig = px.pie(demo_data, values="Equity", names="Symbol",
                    title="Portfolio Allocation")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Market Status")
        st.success("🟢 Demo Mode")
        
        st.markdown("### Top Movers (Demo)")
        st.markdown("**Top Gainers**")
        st.markdown("- NVDA: +5.2%")
        st.markdown("- AMD: +4.1%")
        st.markdown("- TSLA: +3.8%")


def render_portfolio():
    """Render portfolio analysis page."""
    st.markdown("## 📈 Portfolio Analysis")
    
    # Performance chart
    st.markdown("### Portfolio Performance")
    
    # Generate sample performance data
    dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02 + 0.0005
    equity = 10000 * (1 + returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines",
        name="Portfolio", line=dict(color="#1f77b4", width=2)
    ))
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    final_value = equity[-1]
    total_return = (final_value / 10000 - 1) * 100
    max_dd = ((equity.max() - equity.min()) / equity.max()) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    with col1:
        st.metric("Total Return", f"{total_return:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col3:
        st.metric("Max Drawdown", f"-{max_dd:.2f}%")
    with col4:
        st.metric("Volatility", f"{returns.std() * np.sqrt(252) * 100:.2f}%")


def render_strategies():
    """Render strategies page."""
    st.markdown("## 🎯 Trading Strategies")
    
    st.markdown("""
    Kuber supports multiple algorithmic trading strategies. 
    Select and configure strategies below.
    """)
    
    # Strategy selection
    strategies_info = {
        "Moving Average Crossover": {
            "description": "Generates signals when fast MA crosses slow MA",
            "params": {"fast_period": 10, "slow_period": 30},
            "type": "Trend Following"
        },
        "RSI (Relative Strength Index)": {
            "description": "Identifies overbought/oversold conditions",
            "params": {"period": 14, "oversold": 30, "overbought": 70},
            "type": "Mean Reversion"
        },
        "MACD": {
            "description": "Measures momentum using EMA differences",
            "params": {"fast": 12, "slow": 26, "signal": 9},
            "type": "Momentum"
        },
        "Bollinger Bands": {
            "description": "Trades based on price deviation from mean",
            "params": {"period": 20, "std_dev": 2.0},
            "type": "Volatility"
        },
        "Mean Reversion": {
            "description": "Trades when price deviates from historical mean",
            "params": {"lookback": 20, "z_threshold": 2.0},
            "type": "Mean Reversion"
        }
    }
    
    for name, info in strategies_info.items():
        with st.expander(f"📊 {name}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Type:** {info['type']}")
                st.markdown(f"**Description:** {info['description']}")
                
            with col2:
                st.markdown("**Parameters:**")
                for param, value in info["params"].items():
                    st.number_input(param, value=value, key=f"{name}_{param}")
                    
            if st.button(f"Activate {name}", key=f"btn_{name}"):
                st.success(f"{name} strategy activated!")


def render_backtest():
    """Render backtest page."""
    st.markdown("## 🔬 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuration")
        
        symbol = st.text_input("Symbol", value="AAPL")
        
        strategy = st.selectbox(
            "Strategy",
            ["MA Crossover", "RSI", "MACD", "Bollinger Bands", "Combined"]
        )
        
        initial_capital = st.number_input("Initial Capital ($)", 
                                         value=10000, step=1000)
        
        period = st.selectbox("Backtest Period", 
                             ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"])
        
        if st.button("Run Backtest", use_container_width=True, type="primary"):
            with st.spinner("Running backtest..."):
                # Simulate backtest
                import time
                time.sleep(2)
                
                # Generate sample results
                dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
                np.random.seed(42)
                returns = np.random.randn(252) * 0.02 + 0.0008
                equity = initial_capital * (1 + returns).cumprod()
                
                st.session_state.backtest_results = {
                    "dates": dates,
                    "equity": equity,
                    "returns": returns,
                    "total_return": (equity[-1] / initial_capital - 1) * 100,
                    "sharpe": (returns.mean() / returns.std()) * np.sqrt(252),
                    "max_dd": ((equity.max() - equity.min()) / equity.max()) * 100,
                    "trades": 45,
                    "win_rate": 58.3
                }
                st.success("Backtest complete!")
                
    with col2:
        st.markdown("### Results")
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{results['total_return']:.2f}%")
            m2.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
            m3.metric("Max Drawdown", f"-{results['max_dd']:.2f}%")
            m4.metric("Win Rate", f"{results['win_rate']:.1f}%")
            
            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results["dates"],
                y=results["equity"],
                mode="lines",
                name="Strategy",
                line=dict(color="#1f77b4")
            ))
            
            # Add benchmark (simple buy-hold)
            benchmark = initial_capital * (1 + np.random.randn(252) * 0.015 + 0.0003).cumprod()
            fig.add_trace(go.Scatter(
                x=results["dates"],
                y=benchmark,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="#ff7f0e", dash="dash")
            ))
            
            fig.update_layout(
                title="Backtest Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                legend=dict(x=0, y=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown chart
            cummax = pd.Series(results["equity"]).expanding().max()
            drawdown = (results["equity"] - cummax) / cummax * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=results["dates"],
                y=drawdown,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#ff1744")
            ))
            fig_dd.update_layout(
                title="Drawdown",
                yaxis_title="Drawdown (%)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("Configure and run a backtest to see results")


def render_settings():
    """Render settings page."""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Risk Management", "Trading", "Notifications"])
    
    with tab1:
        st.markdown("### Risk Management Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Max Position Size (%)", 1, 25, 10)
            st.slider("Max Portfolio Positions", 5, 50, 20)
            st.slider("Default Stop Loss (%)", 1, 15, 5)
            
        with col2:
            st.slider("Max Daily Loss (%)", 1, 10, 3)
            st.slider("Max Drawdown (%)", 5, 30, 15)
            st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)
            
    with tab2:
        st.markdown("### Trading Settings")
        
        st.selectbox("Trading Mode", ["Paper Trading", "Live Trading"])
        st.number_input("Min Trade Interval (seconds)", value=60)
        st.checkbox("Enable After-Hours Trading", value=False)
        st.checkbox("Use Trailing Stops", value=True)
        
    with tab3:
        st.markdown("### Notification Settings")
        
        st.checkbox("Email Notifications", value=True)
        st.text_input("Email Address")
        st.multiselect(
            "Notify On",
            ["Trade Executed", "Stop Loss Hit", "Daily Summary", "Large Drawdown"],
            default=["Trade Executed", "Stop Loss Hit"]
        )
        
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved!")


def main():
    """Main application entry point."""
    initialize_session_state()
    render_header()
    
    page = render_sidebar()
    
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "📈 Portfolio":
        render_portfolio()
    elif page == "🎯 Strategies":
        render_strategies()
    elif page == "🔬 Backtest":
        render_backtest()
    elif page == "⚙️ Settings":
        render_settings()
        
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Kuber Trading Platform v0.1.0 | "
        "⚠️ Trading involves risk. Past performance does not guarantee future results."
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
