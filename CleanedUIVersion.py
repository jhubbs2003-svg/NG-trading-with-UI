#!/usr/bin/env python3
"""
Natural Gas Trading Model - Interactive Web Interface
Run with: streamlit run ng_trading_ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Import the backend model from the separate .py file
from enhanced_ng_model import EnhancedNaturalGasModel, StrategyOptimizer

# Helper functions for the Streamlit interface
def generate_simple_signals(data, rsi_buy=30, rsi_sell=70):
    """Generate simple trading signals based on user-defined thresholds."""
    df = data.copy()
    df['signal'] = 0
    
    if 'rsi' not in df.columns or 'bb_low' not in df.columns or 'bb_high' not in df.columns:
        st.error("❌ Technical indicators (RSI, Bollinger Bands) not found. Please ensure they are calculated.")
        return pd.DataFrame({'signal': [0] * len(df)}, index=df.index)
    
    buy_condition = (df['rsi'] < rsi_buy) & (df['price'] < df['bb_low'])
    df.loc[buy_condition, 'signal'] = 1
    
    sell_condition = (df['rsi'] > rsi_sell) & (df['price'] > df['bb_high'])
    df.loc[sell_condition, 'signal'] = -1
    
    return df[['signal']]

def simple_backtest(data, signals_df, initial_capital=100000):
    """Run a simple backtest for the Streamlit interface."""
    try:
        df = data.join(signals_df, how='inner').copy()
        df = df.dropna(subset=['price', 'signal', 'returns'])
        
        if df.empty:
            st.error("❌ No valid data available for backtesting after merging signals.")
            return None
        
        df['position'] = df['signal'].shift(1).fillna(0)
        df['strategy_returns'] = df['position'] * df['returns']
        df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod() * initial_capital
        df['cumulative_market'] = (1 + df['returns']).cumprod() * initial_capital
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error during backtesting: {e}")
        return None

def calculate_performance_metrics(df, initial_capital):
    """Calculate and return key performance metrics."""
    try:
        strategy_returns = df['strategy_returns'].dropna()
        if strategy_returns.empty:
            return {
                'Total Return': 0, 'Market Return': 0, 'Sharpe Ratio': 0,
                'Max Drawdown': 0, 'Volatility': 0
            }

        total_return = (df['cumulative_strategy'].iloc[-1] / initial_capital) - 1
        market_return = (df['cumulative_market'].iloc[-1] / initial_capital) - 1
        
        volatility = strategy_returns.std() * np.sqrt(52) # Assuming weekly data from the model
        sharpe_ratio = (strategy_returns.mean() * 52) / volatility if volatility > 0 else 0
        
        max_dd = (df['cumulative_strategy'] / df['cumulative_strategy'].cummax() - 1).min()
        
        return {
            'Total Return': total_return, 'Market Return': market_return,
            'Sharpe Ratio': sharpe_ratio, 'Max Drawdown': max_dd, 'Volatility': volatility
        }
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {e}")
        return {}

# Configure Streamlit page
st.set_page_config(
    page_title="Natural Gas Trading Model",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header { font-size: 3rem; font-weight: bold; color: #1f4e79; text-align: center; margin-bottom: 2rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f4e79; }
.success-metric { border-left-color: #28a745; }
.warning-metric { border-left-color: #ffc107; }
.danger-metric { border-left-color: #dc3545; }
.sidebar-section { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = EnhancedNaturalGasModel()
if 'results_to_display' not in st.session_state:
    st.session_state.results_to_display = False

# Main title
st.markdown('<h1 class="main-header">⛽ Natural Gas Trading Model Dashboard</h1>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("🔧 Configuration")
    api_key = st.text_input("EIA API Key (Optional)", type="password", help="Enter your free EIA API key for storage data.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("🧠 Strategy & Model")
    strategy_type = st.selectbox("Strategy Type", ["Simple Technicals", "Machine Learning Prediction"])
    
    if strategy_type == "Machine Learning Prediction":
        model_type = st.selectbox("Model Type", ['Random Forest', 'Gradient Boosting'])
        prediction_target = st.selectbox("Prediction Target", ['direction_1d', 'direction_5d', 'direction_20d'])
    else:
        rsi_buy = st.slider("RSI Buy Threshold", 10, 40, 30, 1)
        rsi_sell = st.slider("RSI Sell Threshold", 60, 90, 70, 1)

    initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.model.api_key = api_key if api_key else None
        # ng_trading_ui.py (in the sidebar section)

    st.sidebar.header("🧠 Model Configuration")

    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        ["Simple Technicals", "Machine Learning Prediction"]
    )

    if strategy_type == "Machine Learning Prediction":
        model_type = st.sidebar.selectbox(
            "Model Type",
            ['Random Forest', 'Gradient Boosting'],
            help="Choose the algorithm for the predictive model."
        )
        prediction_target = st.sidebar.selectbox(
            "Prediction Target",
            ['direction_1d', 'direction_5d', 'direction_20d'],
            help="What should the model predict? (Price direction in X days)"
        )
        ml_confidence = st.sidebar.slider(
            "ML Signal Confidence", 0.55, 0.80, 0.60, 0.05,
            help="The probability threshold required to generate a buy/sell signal."
        )

    else:
            with st.spinner("Running simple technical analysis..."):
                try:
                    # For simple mode, we still need to prepare basic data
                    st.session_state.model.prepare_features() # This will get data and add all indicators
                    signals_df = generate_simple_signals(st.session_state.model.data, rsi_buy, rsi_sell)
                    backtest_results = simple_backtest(st.session_state.model.data, signals_df, initial_capital)
                    performance_metrics = calculate_performance_metrics(backtest_results, initial_capital)

                    st.session_state.backtest_results = backtest_results
                    st.session_state.performance_metrics = performance_metrics
                    st.session_state.signals = signals_df
                    st.session_state.results_to_display = True
                except Exception as e:
                    st.error(f"Simple Analysis Failed: {e}")
                    st.session_state.results_to_display = False

    st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN CONTENT AREA ---
if st.session_state.results_to_display:
    st.header("🎯 Performance Metrics")
    metrics = st.session_state.performance_metrics
    m_cols = st.columns(len(metrics))
    for i, (metric, value) in enumerate(metrics.items()):
        with m_cols[i]:
            if isinstance(value, float):
                if 'Return' in metric or 'Drawdown' in metric or 'Rate' in metric:
                     m_cols[i].metric(metric, f"{value:.2%}")
                else:
                     m_cols[i].metric(metric, f"{value:.2f}")
            else:
                 m_cols[i].metric(metric, f"{value:,.0f}")
    
    st.header("📊 Strategy Performance")
    backtest_results = st.session_state.backtest_results
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Plot 1: Cumulative Returns
    fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results['cumulative_strategy'], name='Strategy', line=dict(color='#28a745')), row=1, col=1)
    fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results['cumulative_market'], name='Buy & Hold', line=dict(color='#1f4e79')), row=1, col=1)

    # Plot 2: Price and Signals
    signal_col = 'combined_signal' if 'combined_signal' in backtest_results.columns else 'signal'
    buy_signals = backtest_results[backtest_results[signal_col] > 0]
    sell_signals = backtest_results[backtest_results[signal_col] < 0]

    fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results['price'], name='Price', line=dict(color='grey')), row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)), row=2, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)), row=2, col=1)

    fig.update_layout(height=700, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # Display feature importance if ML model was run
    if strategy_type == "Machine Learning Prediction" and hasattr(st.session_state.model, 'selected_features'):
        st.header("🧠 Model Insights: Top Predictive Features")
        importances = st.session_state.model.model.feature_importances_
        feature_imp = pd.DataFrame(sorted(zip(importances, st.session_state.model.selected_features)), columns=['Value','Feature'])
        fig_imp = px.bar(feature_imp.tail(15), x="Value", y="Feature", orientation='h', title="Top 15 Most Important Features")
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.header("🚀 Getting Started")
    st.markdown("""
    Welcome to the Natural Gas Trading Model Dashboard! This tool analyzes price movements and backtests trading strategies.
    1.  **Configure**: Use the sidebar to enter an optional EIA API key and select a strategy.
    2.  **Analyze**: Click "Run Analysis" to perform a backtest.
    3.  **Review**: Examine the performance metrics and charts that appear.
    """)
    # Sample data visualization
    st.header("📊 Sample Price Chart")
    dates = pd.date_range(start='2024-01-01', end=datetime.today(), freq='D')
    sample_price = 4 + np.cumsum(np.random.randn(len(dates)) * 0.05)
    fig_sample = go.Figure(go.Scatter(x=dates, y=sample_price, mode='lines', name='Sample NG Price'))
    fig_sample.update_layout(title="Sample Natural Gas Price Movement", template="plotly_white")
    st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with Streamlit & Plotly | For educational purposes only.</p>", unsafe_allow_html=True)
