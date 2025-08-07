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
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64

# Import your enhanced model (assuming it's in the same directory)
# from enhanced_ng_model import EnhancedNaturalGasModel, StrategyOptimizer

# For demo purposes, I'll include a simplified version of the model
# In practice, you'd import from your enhanced_ng_model.py file

class SimplifiedNGModel:
    """Simplified version for demo - replace with your full model"""
    
    def __init__(self):
        self.data = None
        self.signals = None
        self.performance_metrics = None
        
    def fetch_data(self, ticker='NG=F', start_date='2020-01-01'):
        """Fetch natural gas data"""
        try:
            data = yf.download(ticker, start=start_date, interval='1d')
            if not data.empty:
                self.data = data[['Close']].rename(columns={'Close': 'price'})
                self.data['returns'] = self.data['price'].pct_change()
                return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        return False
    
    def add_technical_indicators(self):
        """Add basic technical indicators"""
        if self.data is None:
            return
            
        # Simple moving averages
        self.data['sma_20'] = self.data['price'].rolling(window=20).mean()
        self.data['sma_50'] = self.data['price'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        self.data['bb_mid'] = self.data['price'].rolling(window=bb_period).mean()
        bb_std_val = self.data['price'].rolling(window=bb_period).std()
        self.data['bb_upper'] = self.data['bb_mid'] + (bb_std_val * bb_std)
        self.data['bb_lower'] = self.data['bb_mid'] - (bb_std_val * bb_std)
    
    def generate_signals(self, rsi_buy=30, rsi_sell=70):
        """Generate simple trading signals"""
        if self.data is None:
            return
            
        self.data['signal'] = 0
        
        # Buy signals
        buy_condition = (
            (self.data['rsi'] < rsi_buy) & 
            (self.data['price'] < self.data['bb_lower'])
        )
        self.data.loc[buy_condition, 'signal'] = 1
        
        # Sell signals
        sell_condition = (
            (self.data['rsi'] > rsi_sell) & 
            (self.data['price'] > self.data['bb_upper'])
        )
        self.data.loc[sell_condition, 'signal'] = -1
        
        return self.data['signal']
    
    def backtest_strategy(self, initial_capital=100000):
        """Simple backtesting"""
        if self.data is None or 'signal' not in self.data.columns:
            return None
            
        df = self.data.copy()
        df['position'] = df['signal'].shift(1).fillna(0)
        df['strategy_returns'] = df['position'] * df['returns']
        df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod() * initial_capital
        df['cumulative_market'] = (1 + df['returns']).cumprod() * initial_capital
        
        # Calculate metrics
        total_return = (df['cumulative_strategy'].iloc[-1] / initial_capital) - 1
        market_return = (df['cumulative_market'].iloc[-1] / initial_capital) - 1
        
        volatility = df['strategy_returns'].std() * np.sqrt(252)
        sharpe_ratio = (df['strategy_returns'].mean() * 252) / volatility if volatility > 0 else 0
        
        max_dd = (df['cumulative_strategy'] / df['cumulative_strategy'].cummax() - 1).min()
        
        self.performance_metrics = {
            'Total Return': total_return,
            'Market Return': market_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_dd,
            'Volatility': volatility
        }
        
        return df

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
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f4e79;
}

.success-metric {
    border-left-color: #28a745;
}

.warning-metric {
    border-left-color: #ffc107;
}

.danger-metric {
    border-left-color: #dc3545;
}

.sidebar-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = SimplifiedNGModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Main title
st.markdown('<h1 class="main-header">⛽ Natural Gas Trading Model Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.header("🔧 Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Yahoo Finance (NG=F)", "Custom Upload", "EIA API"],
    help="Select your preferred data source"
)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=1095),  # 3 years ago
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# EIA API Key input (if selected)
if data_source == "EIA API":
    eia_api_key = st.sidebar.text_input(
        "EIA API Key",
        type="password",
        help="Enter your EIA API key for official data"
    )

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Strategy parameters
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.header("📊 Strategy Parameters")

rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 40, 30, 1)
rsi_sell = st.sidebar.slider("RSI Sell Threshold", 60, 90, 70, 1)
initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Action buttons
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
load_data_btn = st.sidebar.button("📥 Load Data", type="primary")
run_analysis_btn = st.sidebar.button("🚀 Run Analysis", type="secondary")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content area
if load_data_btn:
    with st.spinner("Loading natural gas data..."):
        if data_source == "Yahoo Finance (NG=F)":
            success = st.session_state.model.fetch_data('NG=F', start_date.strftime('%Y-%m-%d'))
            if success:
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            else:
                st.error("❌ Failed to load data")
        else:
            st.info("🔄 Custom upload and EIA API integration coming soon!")

# Display data overview
if st.session_state.data_loaded and st.session_state.model.data is not None:
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    data = st.session_state.model.data
    
    with col1:
        st.metric(
            "Current Price",
            f"${data['price'].iloc[-1]:.2f}",
            f"{data['returns'].iloc[-1]:.2%}"
        )
    
    with col2:
        st.metric(
            "30-Day Average",
            f"${data['price'].tail(30).mean():.2f}",
            f"{((data['price'].iloc[-1] / data['price'].tail(30).mean()) - 1):.2%}"
        )
    
    with col3:
        st.metric(
            "30-Day Volatility",
            f"{data['returns'].tail(30).std() * np.sqrt(252):.1%}",
            ""
        )
    
    with col4:
        st.metric(
            "Total Observations",
            f"{len(data):,}",
            ""
        )
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['price'],
        mode='lines',
        name='Natural Gas Price',
        line=dict(color='#1f4e79', width=2)
    ))
    
    fig.update_layout(
        title="Natural Gas Price History",
        xaxis_title="Date",
        yaxis_title="Price ($/MMBtu)",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Run analysis
if run_analysis_btn and st.session_state.data_loaded:
    with st.spinner("Running trading strategy analysis..."):
        # Add technical indicators
        st.session_state.model.add_technical_indicators()
        
        # Generate signals
        st.session_state.model.generate_signals(rsi_buy, rsi_sell)
        
        # Backtest
        backtest_results = st.session_state.model.backtest_strategy(initial_capital)
        
        if backtest_results is not None:
            st.success("✅ Analysis completed!")
            
            # Performance metrics
            st.header("🎯 Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = st.session_state.model.performance_metrics
            
            with col1:
                color = "success" if metrics['Total Return'] > 0 else "danger"
                st.markdown(f'<div class="metric-card {color}-metric">', unsafe_allow_html=True)
                st.metric("Strategy Return", f"{metrics['Total Return']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Market Return", f"{metrics['Market Return']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                color = "success" if metrics['Sharpe Ratio'] > 1 else "warning" if metrics['Sharpe Ratio'] > 0 else "danger"
                st.markdown(f'<div class="metric-card {color}-metric">', unsafe_allow_html=True)
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                color = "success" if metrics['Max Drawdown'] > -0.1 else "warning" if metrics['Max Drawdown'] > -0.2 else "danger"
                st.markdown(f'<div class="metric-card {color}-metric">', unsafe_allow_html=True)
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col5:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Volatility", f"{metrics['Volatility']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Strategy vs Market comparison chart
            st.header("📊 Strategy Performance")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Cumulative Returns', 'Trading Signals'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['cumulative_strategy'],
                    mode='lines',
                    name='Strategy',
                    line=dict(color='#28a745', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['cumulative_market'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='#1f4e79', width=2)
                ),
                row=1, col=1
            )
            
            # Trading signals
            buy_signals = backtest_results[backtest_results['signal'] == 1]
            sell_signals = backtest_results[backtest_results['signal'] == -1]
            
            fig.add_trace(
                go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', symbol='triangle-up', size=8)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', symbol='triangle-down', size=8)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=700,
                template="plotly_white",
                title="Strategy Performance Analysis"
            )
            
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Price ($/MMBtu)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators chart
            st.header("🔧 Technical Indicators")
            
            fig_tech = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price with Bollinger Bands', 'RSI'),
                vertical_spacing=0.15
            )
            
            # Price and Bollinger Bands
            fig_tech.add_trace(
                go.Scatter(x=backtest_results.index, y=backtest_results['price'], 
                          mode='lines', name='Price', line=dict(color='#1f4e79')),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(x=backtest_results.index, y=backtest_results['bb_upper'], 
                          mode='lines', name='BB Upper', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(x=backtest_results.index, y=backtest_results['bb_lower'], 
                          mode='lines', name='BB Lower', line=dict(color='green', dash='dash')),
                row=1, col=1
            )
            
            # RSI
            fig_tech.add_trace(
                go.Scatter(x=backtest_results.index, y=backtest_results['rsi'], 
                          mode='lines', name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            
            # RSI threshold lines
            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig_tech.update_layout(height=600, template="plotly_white")
            fig_tech.update_yaxes(title_text="Price ($/MMBtu)", row=1, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=2, col=1)
            
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # Download results
            st.header("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert to CSV
                csv_buffer = io.StringIO()
                backtest_results.to_csv(csv_buffer)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📄 Download CSV Data",
                    data=csv_data,
                    file_name=f"ng_trading_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary report
                summary_report = f"""
Natural Gas Trading Strategy Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Strategy Parameters:
- RSI Buy Threshold: {rsi_buy}
- RSI Sell Threshold: {rsi_sell}
- Initial Capital: ${initial_capital:,}

Performance Metrics:
- Total Return: {metrics['Total Return']:.2%}
- Market Return: {metrics['Market Return']:.2%}
- Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}
- Max Drawdown: {metrics['Max Drawdown']:.2%}
- Volatility: {metrics['Volatility']:.2%}

Data Period: {start_date} to {end_date}
Total Observations: {len(backtest_results):,}
                """
                
                st.download_button(
                    label="📋 Download Report",
                    data=summary_report,
                    file_name=f"ng_strategy_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

else:
    if not st.session_state.data_loaded:
        st.info("👆 Please load data first using the sidebar controls.")
    
    # Getting started guide
    st.header("🚀 Getting Started")
    
    st.markdown("""
    ### Welcome to the Natural Gas Trading Model Dashboard!
    
    This interactive tool helps you analyze natural gas price movements and test trading strategies.
    
    **To get started:**
    
    1. **📥 Load Data**: Choose your data source and click "Load Data"
    2. **🔧 Configure**: Adjust strategy parameters in the sidebar
    3. **🚀 Analyze**: Click "Run Analysis" to backtest your strategy
    4. **📊 Review**: Examine the performance metrics and charts
    5. **💾 Export**: Download results for further analysis
    
    ### Features:
    
    - 📈 Real-time natural gas price data
    - 🔧 Customizable technical indicators
    - 📊 Interactive performance charts
    - 🎯 Comprehensive backtesting metrics
    - 💾 Export capabilities for results
    
    ### Strategy Overview:
    
    The current strategy uses:
    - **RSI (Relative Strength Index)** for momentum signals
    - **Bollinger Bands** for volatility-based entries
    - **Risk management** through position sizing
    
    Adjust the parameters in the sidebar to customize the strategy to your preferences.
    """)
    
    # Sample data visualization
    st.header("📊 Sample Analysis")
    
    # Create sample data for demo
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    sample_price = 3 + np.cumsum(np.random.randn(len(dates)) * 0.05)
    
    fig_sample = go.Figure()
    fig_sample.add_trace(go.Scatter(
        x=dates,
        y=sample_price,
        mode='lines',
        name='Sample Natural Gas Price',
        line=dict(color='#1f4e79', width=2)
    ))
    
    fig_sample.update_layout(
        title="Sample Natural Gas Price Movement",
        xaxis_title="Date",
        yaxis_title="Price ($/MMBtu)",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem;'>
    <p>Natural Gas Trading Model Dashboard | Built with Streamlit & Plotly</p>
    <p>⚠️ This tool is for educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)