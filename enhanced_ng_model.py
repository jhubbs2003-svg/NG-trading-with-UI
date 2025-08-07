#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests as rqst
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Technical Analysis
try:
    import ta
except ImportError:
    print("Installing ta library...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'ta'])
    import ta

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class EnhancedNaturalGasModel:
    def __init__(self, api_key=None):
        """
        Enhanced Natural Gas Trading Model with multiple data sources and advanced features
        """
        self.api_key = api_key
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def fetch_eia_data(self, series_id, start_date='2015-01-01'):
        """Fetch data from EIA API"""
        if not self.api_key:
            print("No EIA API key provided, using Yahoo Finance only")
            return None
            
        url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={self.api_key}"
        try:
            response = rqst.get(url)
            data_json = response.json()
            
            if "response" in data_json:
                raw_data = data_json["response"]["data"]
                df = pd.DataFrame(raw_data)
                df['period'] = pd.to_datetime(df['period'])
                df = df.rename(columns={'value': 'price'})
                df = df.set_index('period')
                df = df.sort_index()
                return df
            else:
                print(f"EIA API Error: {data_json.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error fetching EIA data: {e}")
            return None
    
    def fetch_yahoo_data(self, ticker='NG=F', start_date='2015-01-01'):
        """Fetch natural gas futures data from Yahoo Finance"""
        try:
            ng_data = yf.download(ticker, start=start_date, interval='1d')
            if ng_data.empty:
                raise ValueError("No data downloaded from Yahoo Finance")
            
            # Use Close price
            ng_data = ng_data[['Close']].rename(columns={'Close': 'price'})
            ng_data.index = pd.to_datetime(ng_data.index)
            return ng_data
        except Exception as e:
            print(f"Error fetching Yahoo data: {e}")
            return None
    
    def fetch_weather_proxy(self):
        """
        Fetch weather-related data (using HDD/CDD proxy via additional tickers)
        This is a simplified approach - in practice, you'd use weather APIs
        """
        try:
            # Heating Oil as weather proxy (correlated with heating demand)
            ho_data = yf.download('HO=F', start='2015-01-01', interval='1d')
            if not ho_data.empty:
                weather_proxy = ho_data[['Close']].rename(columns={'Close': 'weather_proxy'})
                return weather_proxy
        except:
            pass
        return None
    
    def fetch_related_commodities(self):
        """Fetch related commodity prices for market context"""
        commodities = {
            'crude_oil': 'CL=F',
            'heating_oil': 'HO=F', 
            'gasoline': 'RB=F',
            'coal': 'MTF=F'  # Coal futures
        }
        
        commodity_data = {}
        for name, ticker in commodities.items():
            try:
                data = yf.download(ticker, start='2015-01-01', interval='1d')
                if not data.empty:
                    commodity_data[name] = data[['Close']].rename(columns={'Close': name})
            except:
                continue
        
        return commodity_data
    
    def load_storage_data(self, file_path='storage_data.xls'):
        """Load EIA storage data from Excel file"""
        try:
            storage_df = pd.read_excel(file_path, sheet_name='Data 1', skiprows=2)
            
            # Rename columns
            column_mapping = {
                'Date': 'date',
                'Weekly Lower 48 States Natural Gas Working Underground Storage (Billion Cubic Feet)': 'storage_total_bcf'
            }
            
            # Handle different possible column names
            for col in storage_df.columns:
                if 'Lower 48' in col and 'Storage' in col:
                    column_mapping[col] = 'storage_total_bcf'
                    break
            
            storage_df.rename(columns=column_mapping, inplace=True)
            
            if 'date' in storage_df.columns:
                storage_df['date'] = pd.to_datetime(storage_df['date'])
                storage_df.set_index('date', inplace=True)
            
            return storage_df
        except Exception as e:
            print(f"Could not load storage data: {e}")
            return None
    
    def create_synthetic_storage(self, df):
        """Create synthetic storage data based on seasonality if real data unavailable"""
        print("Creating synthetic storage data based on seasonal patterns...")
        
        # Create seasonal storage pattern (higher in fall/winter, lower in summer)
        df['day_of_year'] = df.index.dayofyear
        
        # Synthetic storage based on seasonal pattern
        storage_base = 3500  # Base storage level in Bcf
        seasonal_amplitude = 1000  # Seasonal variation
        
        # Peak storage in November (day 315), minimum in March (day 80)
        df['storage_total_bcf'] = storage_base + seasonal_amplitude * np.cos(
            2 * np.pi * (df['day_of_year'] - 315) / 365
        )
        
        # Add some noise and trend
        np.random.seed(42)
        df['storage_total_bcf'] += np.random.normal(0, 100, len(df))
        
        return df
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical analysis indicators"""
        
        # Price-based indicators
        df['sma_5'] = ta.trend.sma_indicator(df['price'], window=5)
        df['sma_20'] = ta.trend.sma_indicator(df['price'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['price'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['price'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['price'], window=26)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['price'], window=14)
        df['rsi_30'] = ta.momentum.rsi(df['price'], window=30)
        df['stoch_k'] = ta.momentum.stoch(df['price'], df['price'], df['price'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['price'], df['price'], df['price'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['price'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['price'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['price'], df['price'], df['price'], window=14)
        df['volatility_20'] = df['price'].rolling(window=20).std()
        
        # Volume indicators (using price as proxy if volume not available)
        df['price_volume'] = df['price'] * abs(df['price'].pct_change())
        df['obv'] = ta.volume.on_balance_volume(df['price'], df['price_volume'].fillna(1))
        
        return df
    
    def add_market_structure_features(self, df):
        """Add market structure and regime features"""
        
        # Price returns and volatility
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Volatility measures
        df['realized_vol_5'] = df['returns'].rolling(window=5).std() * np.sqrt(252)
        df['realized_vol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Trend strength
        df['trend_strength'] = abs(df['sma_20'].pct_change(20))
        
        # Support/Resistance levels
        df['high_20'] = df['price'].rolling(window=20).max()
        df['low_20'] = df['price'].rolling(window=20).min()
        df['price_position'] = (df['price'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Momentum features
        df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_20'] = df['price'] / df['price'].shift(20) - 1
        df['momentum_60'] = df['price'] / df['price'].shift(60) - 1
        
        return df
    
    def add_seasonal_features(self, df):
        """Add seasonal and calendar features"""
        
        # Basic calendar features
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        
        # Seasonal indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_injection_season'] = df['month'].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
        df['is_withdrawal_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
        
        # Cyclical encoding of seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def add_storage_features(self, df):
        """Add storage-related features"""
        
        if 'storage_total_bcf' not in df.columns:
            df = self.create_synthetic_storage(df)
        
        # Storage change features
        df['storage_change'] = df['storage_total_bcf'].diff()
        df['storage_change_pct'] = df['storage_total_bcf'].pct_change()
        
        # Storage relative to seasonal norms
        df['storage_seasonal_avg'] = df.groupby(df.index.dayofyear)['storage_total_bcf'].transform('mean')
        df['storage_vs_seasonal'] = df['storage_total_bcf'] - df['storage_seasonal_avg']
        df['storage_vs_seasonal_pct'] = df['storage_vs_seasonal'] / df['storage_seasonal_avg']
        
        # Storage percentiles
        df['storage_percentile'] = df['storage_total_bcf'].rolling(window=252).rank(pct=True)
        
        # Storage momentum
        df['storage_momentum_4w'] = df['storage_total_bcf'] - df['storage_total_bcf'].shift(28)
        df['storage_momentum_12w'] = df['storage_total_bcf'] - df['storage_total_bcf'].shift(84)
        
        return df
    
    def add_spread_features(self, df, commodity_data):
        """Add spread and correlation features with other commodities"""
        
        for name, data in commodity_data.items():
            # Align dates
            aligned_data = data.reindex(df.index, method='ffill')
            df[f'{name}_price'] = aligned_data.iloc[:, 0]
            
            # Calculate spreads
            df[f'ng_{name}_spread'] = df['price'] - df[f'{name}_price']
            df[f'ng_{name}_ratio'] = df['price'] / df[f'{name}_price']
            
            # Rolling correlations
            df[f'ng_{name}_corr_20'] = df['price'].rolling(window=20).corr(df[f'{name}_price'])
        
        return df
    
    def create_targets(self, df, horizons=[1, 5, 20]):
        """Create multiple prediction targets"""
        
        targets = {}
        
        for horizon in horizons:
            # Price direction
            future_price = df['price'].shift(-horizon)
            targets[f'direction_{horizon}d'] = (future_price > df['price']).astype(int)
            
            # Return magnitude
            future_return = (future_price / df['price'] - 1)
            targets[f'return_{horizon}d'] = future_return
            
            # Volatility target
            future_vol = df['returns'].shift(-horizon).rolling(window=horizon).std()
            targets[f'volatility_{horizon}d'] = future_vol
        
        # Add targets to dataframe
        for name, target in targets.items():
            df[name] = target
        
        return df, list(targets.keys())
    
    def prepare_features(self):
        """Main method to prepare all features"""
        
        print("Fetching price data...")
        # Try EIA first, fallback to Yahoo
        price_data = None
        if self.api_key:
            price_data = self.fetch_eia_data("NG.RNGWHHD.D")
        
        if price_data is None:
            price_data = self.fetch_yahoo_data()
        
        if price_data is None:
            raise ValueError("Could not fetch price data from any source")
        
        # Convert to weekly data
        weekly_data = price_data.resample('W-FRI').last()
        
        print("Adding technical indicators...")
        weekly_data = self.add_technical_indicators(weekly_data)
        
        print("Adding market structure features...")
        weekly_data = self.add_market_structure_features(weekly_data)
        
        print("Adding seasonal features...")
        weekly_data = self.add_seasonal_features(weekly_data)
        
        print("Adding storage features...")
        storage_data = self.load_storage_data()
        if storage_data is not None:
            weekly_data = weekly_data.join(storage_data, how='left')
        weekly_data = self.add_storage_features(weekly_data)
        
        print("Fetching related commodities...")
        commodity_data = self.fetch_related_commodities()
        if commodity_data:
            weekly_data = self.add_spread_features(weekly_data, commodity_data)
        
        print("Creating targets...")
        weekly_data, target_names = self.create_targets(weekly_data)
        
        # Clean data
        weekly_data = weekly_data.replace([np.inf, -np.inf], np.nan)
        
        self.data = weekly_data
        self.target_names = target_names
        
        print(f"Feature preparation complete. Shape: {weekly_data.shape}")
        return weekly_data
    
    def select_features(self, target='direction_1d', k=20):
        """Feature selection using statistical tests"""
        
        if self.data is None:
            raise ValueError("Must prepare features first")
        
        # Get feature columns (exclude targets and non-feature columns)
        exclude_cols = self.target_names + ['price', 'returns', 'log_returns', 'day_of_year']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove features with too many NaNs
        valid_features = []
        for col in feature_cols:
            if self.data[col].notna().sum() / len(self.data) > 0.7:  # At least 70% non-NaN
                valid_features.append(col)
        
        # Prepare data
        X = self.data[valid_features].fillna(method='ffill').fillna(method='bfill')
        y = self.data[target].fillna(0)
        
        # Remove rows with NaN targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, len(valid_features)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [valid_features[i] for i in range(len(valid_features)) if selected_mask[i]]
        
        print(f"Selected {len(self.selected_features)} features:")
        for i, feature in enumerate(self.selected_features[:10]):  # Show top 10
            score = self.feature_selector.scores_[selected_mask][i]
            print(f"  {feature}: {score:.2f}")
        
        return self.selected_features
    
    def train_model(self, target='direction_1d', model_type='rf', test_size=0.2):
        """Train predictive model"""
        
        if not hasattr(self, 'selected_features'):
            self.select_features(target)
        
        # Prepare data
        X = self.data[self.selected_features].fillna(method='ffill').fillna(method='bfill')
        y = self.data[target].fillna(0)
        
        # Remove rows with NaN targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained. Test accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store for backtesting
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.test_dates = X_test.index
        
        return self.model
    
    def generate_signals(self, lookback_days=252):
        """Generate trading signals using multiple approaches"""
        
        if self.data is None:
            raise ValueError("Must prepare features first")
        
        signals_df = self.data.copy()
        
        # 1. Technical Analysis Signal
        signals_df['ta_signal'] = 0
        
        # Buy conditions
        buy_conditions = (
            (signals_df['rsi'] < 30) &
            (signals_df['macd'] > signals_df['macd_signal']) &
            (signals_df['price'] < signals_df['bb_low']) &
            (signals_df['storage_vs_seasonal_pct'] < -0.1)  # Storage below seasonal average
        )
        
        # Sell conditions  
        sell_conditions = (
            (signals_df['rsi'] > 70) &
            (signals_df['macd'] < signals_df['macd_signal']) &
            (signals_df['price'] > signals_df['bb_high']) &
            (signals_df['storage_vs_seasonal_pct'] > 0.1)   # Storage above seasonal average
        )
        
        signals_df.loc[buy_conditions, 'ta_signal'] = 1
        signals_df.loc[sell_conditions, 'ta_signal'] = -1
        
        # 2. ML Signal (if model trained)
        if self.model is not None and hasattr(self, 'selected_features'):
            X = signals_df[self.selected_features].fillna(method='ffill').fillna(method='bfill')
            X_scaled = self.scaler.transform(X)
            ml_predictions = self.model.predict_proba(X_scaled)
            
            # Convert probabilities to signals
            signals_df['ml_signal'] = 0
            confidence_threshold = 0.6
            
            # Strong buy signal
            strong_buy = ml_predictions[:, 1] > confidence_threshold
            signals_df.loc[strong_buy, 'ml_signal'] = 1
            
            # Strong sell signal  
            strong_sell = ml_predictions[:, 0] > confidence_threshold
            signals_df.loc[strong_sell, 'ml_signal'] = -1
        
        # 3. Storage-based Signal
        signals_df['storage_signal'] = 0
        
        if 'storage_change' in signals_df.columns:
            storage_threshold = signals_df['storage_change'].std()
            
            # Large injection (bearish for prices)
            large_injection = signals_df['storage_change'] > 2 * storage_threshold
            signals_df.loc[large_injection, 'storage_signal'] = -1
            
            # Large withdrawal (bullish for prices)
            large_withdrawal = signals_df['storage_change'] < -2 * storage_threshold
            signals_df.loc[large_withdrawal, 'storage_signal'] = 1
        
        # 4. Combined Signal
        signals_df['combined_signal'] = (
            signals_df.get('ta_signal', 0) + 
            signals_df.get('ml_signal', 0) + 
            signals_df.get('storage_signal', 0)
        )
        
        # Normalize combined signal
        signals_df['combined_signal'] = np.clip(signals_df['combined_signal'], -1, 1)
        
        return signals_df[['ta_signal', 'ml_signal', 'storage_signal', 'combined_signal']].fillna(0)
    
    def backtest_strategy(self, signals_df, signal_col='combined_signal', 
                         transaction_cost=0.001, initial_capital=100000):
        """Comprehensive backtesting with transaction costs"""
        
        df = self.data.join(signals_df, how='inner').copy()
        df = df.dropna(subset=['price', signal_col])
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        df['position'] = df[signal_col].shift(1).fillna(0)
        
        # Calculate strategy returns with transaction costs
        position_changes = df['position'].diff().abs()
        df['transaction_costs'] = position_changes * transaction_cost
        df['strategy_returns'] = (df['position'] * df['returns']) - df['transaction_costs']
        
        # Cumulative performance
        df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod() * initial_capital
        df['cumulative_market'] = (1 + df['returns']).cumprod() * initial_capital
        
        # Performance metrics
        strategy_returns = df['strategy_returns'].dropna()
        market_returns = df['returns'].dropna()
        
        metrics = {
            'Total Return': (df['cumulative_strategy'].iloc[-1] / initial_capital - 1),
            'Market Return': (df['cumulative_market'].iloc[-1] / initial_capital - 1),
            'Annualized Return': strategy_returns.mean() * 52,
            'Annualized Volatility': strategy_returns.std() * np.sqrt(52),
            'Sharpe Ratio': (strategy_returns.mean() * 52) / (strategy_returns.std() * np.sqrt(52)) if strategy_returns.std() > 0 else 0,
            'Max Drawdown': (df['cumulative_strategy'] / df['cumulative_strategy'].cummax() - 1).min(),
            'Win Rate': (strategy_returns > 0).mean(),
            'Avg Win': strategy_returns[strategy_returns > 0].mean(),
            'Avg Loss': strategy_returns[strategy_returns < 0].mean(),
            'Total Trades': position_changes.sum(),
            'Transaction Costs': df['transaction_costs'].sum() * initial_capital
        }
        
        return df, metrics
    
    def plot_results(self, backtest_df, signals_df, save_plots=True):
        """Create comprehensive visualization of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Price and Signals
        ax1 = axes[0, 0]
        ax1.plot(backtest_df.index, backtest_df['price'], label='NG Price', color='blue', alpha=0.7)
        
        # Plot buy/sell signals
        buy_signals = backtest_df[backtest_df['combined_signal'] > 0]
        sell_signals = backtest_df[backtest_df['combined_signal'] < 0]
        
        ax1.scatter(buy_signals.index, buy_signals['price'], 
                   marker='^', color='green', s=50, label='Buy Signal', alpha=0.7)
        ax1.scatter(sell_signals.index, sell_signals['price'], 
                   marker='v', color='red', s=50, label='Sell Signal', alpha=0.7)
        
        ax1.set_title('Natural Gas Price with Trading Signals')
        ax1.set_ylabel('Price ($/MMBtu)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Returns
        ax2 = axes[0, 1]
        ax2.plot(backtest_df.index, backtest_df['cumulative_strategy'], 
                label='Strategy', color='green', linewidth=2)
        ax2.plot(backtest_df.index, backtest_df['cumulative_market'], 
                label='Buy & Hold', color='blue', linewidth=2)
        
        ax2.set_title('Cumulative Returns Comparison')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        rolling_sharpe = (backtest_df['strategy_returns'].rolling(window=52).mean() * 52) / \
                        (backtest_df['strategy_returns'].rolling(window=52).std() * np.sqrt(52))
        
        ax3.plot(backtest_df.index, rolling_sharpe, color='purple', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Rolling 1-Year Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown
        ax4 = axes[1, 1]
        drawdown = backtest_df['cumulative_strategy'] / backtest_df['cumulative_strategy'].cummax() - 1
        ax4.fill_between(backtest_df.index, drawdown, 0, color='red', alpha=0.3)
        ax4.plot(backtest_df.index, drawdown, color='red', linewidth=1)
        ax4.set_title('Strategy Drawdown')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('natural_gas_strategy_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Feature importance plot if model exists
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title("Top Feature Importances")
            plt.bar(range(min(15, len(importances))), 
                    importances[indices[:15]], 
                    color='skyblue', alpha=0.7)
            plt.xticks(range(min(15, len(importances))), 
                      [self.selected_features[i] for i in indices[:15]],
                      rotation=45, ha='right')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

    def run_full_analysis(self, api_key=None, target='direction_1d'):
        """Run complete analysis pipeline"""
        
        self.api_key = api_key
        
        print("="*60)
        print("ENHANCED NATURAL GAS TRADING MODEL")
        print("="*60)
        
        try:
            # 1. Prepare features
            data = self.prepare_features()
            print(f"✓ Features prepared: {data.shape[1]} features, {data.shape[0]} observations")
            
            # 2. Feature selection
            selected_features = self.select_features(target=target)
            print(f"✓ Selected {len(selected_features)} most predictive features")
            
            # 3. Train model
            model = self.train_model(target=target, model_type='rf')
            print("✓ Model trained successfully")
            
            # 4. Generate signals
            signals = self.generate_signals()
            print("✓ Trading signals generated")
            
            # 5. Backtest strategy
            backtest_results, performance_metrics = self.backtest_strategy(signals)
            print("✓ Backtesting completed")
            
            # 6. Display results
            print("\n" + "="*50)
            print("PERFORMANCE SUMMARY")
            print("="*50)
            
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    if 'Return' in metric or 'Drawdown' in metric:
                        print(f"{metric:<25}: {value:>10.2%}")
                    elif 'Rate' in metric:
                        print(f"{metric:<25}: {value:>10.2%}")
                    elif 'Costs' in metric:
                        print(f"{metric:<25}: ${value:>10,.0f}")
                    else:
                        print(f"{metric:<25}: {value:>10.3f}")
                else:
                    print(f"{metric:<25}: {value:>10.0f}")
            
            # 7. Plot results
            print("\n✓ Generating visualizations...")
            self.plot_results(backtest_results, signals)
            
            # 8. Additional analysis
            self.print_signal_analysis(signals)
            self.print_monthly_analysis(backtest_results)
            
            return backtest_results, performance_metrics, signals
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def print_signal_analysis(self, signals_df):
        """Print analysis of trading signals"""
        
        print("\n" + "="*50)
        print("SIGNAL ANALYSIS")
        print("="*50)
        
        for signal_type in ['ta_signal', 'ml_signal', 'storage_signal', 'combined_signal']:
            if signal_type in signals_df.columns:
                signal_counts = signals_df[signal_type].value_counts().sort_index()
                total_signals = len(signals_df)
                
                print(f"\n{signal_type.upper()}:")
                for signal_val, count in signal_counts.items():
                    signal_name = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}.get(signal_val, str(signal_val))
                    percentage = count / total_signals * 100
                    print(f"  {signal_name:<6}: {count:>4} ({percentage:>5.1f}%)")

    def print_monthly_analysis(self, backtest_df):
        """Print monthly performance analysis"""
        
        print("\n" + "="*50)
        print("MONTHLY PERFORMANCE ANALYSIS")
        print("="*50)
        
        monthly_returns = backtest_df.groupby(backtest_df.index.to_period('M'))['strategy_returns'].sum()
        
        print("\nBest performing months:")
        top_months = monthly_returns.nlargest(5)
        for month, ret in top_months.items():
            print(f"  {month}: {ret:>8.2%}")
            
        print("\nWorst performing months:")
        worst_months = monthly_returns.nsmallest(5)
        for month, ret in worst_months.items():
            print(f"  {month}: {ret:>8.2%}")
        
        # Seasonal analysis
        seasonal_performance = backtest_df.groupby(backtest_df.index.month)['strategy_returns'].agg(['mean', 'std', 'count'])
        seasonal_performance.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        print(f"\nSeasonal Performance (Weekly Average):")
        print(f"{'Month':<5} {'Avg Return':<12} {'Std Dev':<10} {'Observations':<12}")
        print("-" * 45)
        for month, row in seasonal_performance.iterrows():
            print(f"{month:<5} {row['mean']:>10.3%} {row['std']:>9.3%} {row['count']:>10.0f}")

    def export_results(self, backtest_df, signals_df, performance_metrics, filename='ng_trading_results.csv'):
        """Export results to CSV for further analysis"""
        
        # Combine all data
        export_df = self.data.join(signals_df, how='inner')
        export_df = export_df.join(backtest_df[['cumulative_strategy', 'cumulative_market', 'strategy_returns']], how='inner')
        
        # Add performance metrics as a separate sheet would require Excel
        # For CSV, we'll add them as comments
        
        with open(filename, 'w') as f:
            f.write("# Natural Gas Trading Model Results\n")
            f.write("# Performance Metrics:\n")
            for metric, value in performance_metrics.items():
                f.write(f"# {metric}: {value}\n")
            f.write("#\n")
            
        # Append the data
        export_df.to_csv(filename, mode='a')
        print(f"✓ Results exported to {filename}")

# Example usage and testing
def main():
    """
    Main function to demonstrate the enhanced natural gas trading model
    """
    
    # Initialize model
    model = EnhancedNaturalGasModel()
    
    # Run full analysis
    # Note: Replace with your actual EIA API key if you have one
    api_key = "jpjOvueuEfbntotnS3i71UEWoISKmilDbof13d3Y"  # Replace with your key
    
    backtest_results, performance_metrics, signals = model.run_full_analysis(
        api_key=api_key,
        target='direction_1d'
    )
    
    if backtest_results is not None:
        # Export results
        model.export_results(backtest_results, signals, performance_metrics)
        
        # Additional custom analysis can be added here
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Check the generated plots and exported CSV file for detailed results.")
        
        # Return results for further use
        return model, backtest_results, performance_metrics, signals
    else:
        print("Analysis failed. Please check the error messages above.")
        return None, None, None, None

# Advanced optimization functions
class StrategyOptimizer:
    """
    Advanced strategy optimization and parameter tuning
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    def optimize_thresholds(self, param_grid=None):
        """Optimize signal generation thresholds"""
        
        if param_grid is None:
            param_grid = {
                'rsi_buy': [25, 30, 35],
                'rsi_sell': [65, 70, 75],
                'storage_threshold': [0.05, 0.1, 0.15],
                'ml_confidence': [0.55, 0.6, 0.65]
            }
        
        best_sharpe = -np.inf
        best_params = None
        results = []
        
        from itertools import product
        
        param_combinations = list(product(*param_grid.values()))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(param_combinations)}")
            
            param_dict = dict(zip(param_grid.keys(), params))
            
            try:
                # Generate signals with these parameters
                signals = self.generate_optimized_signals(param_dict)
                
                # Backtest
                backtest_df, metrics = self.base_model.backtest_strategy(signals, 'combined_signal')
                
                sharpe = metrics['Sharpe Ratio']
                
                results.append({
                    'params': param_dict,
                    'sharpe_ratio': sharpe,
                    'total_return': metrics['Total Return'],
                    'max_drawdown': metrics['Max Drawdown']
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = param_dict
                    
            except Exception as e:
                print(f"Error with params {param_dict}: {e}")
                continue
        
        print(f"\nOptimization complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best Sharpe ratio: {best_sharpe:.3f}")
        
        return best_params, results
    
    def generate_optimized_signals(self, params):
        """Generate signals with optimized parameters"""
        
        data = self.base_model.data
        signals_df = pd.DataFrame(index=data.index)
        
        # Optimized technical analysis signal
        signals_df['ta_signal'] = 0
        
        buy_conditions = (
            (data['rsi'] < params['rsi_buy']) &
            (data['macd'] > data['macd_signal']) &
            (data['price'] < data['bb_low']) &
            (data['storage_vs_seasonal_pct'] < -params['storage_threshold'])
        )
        
        sell_conditions = (
            (data['rsi'] > params['rsi_sell']) &
            (data['macd'] < data['macd_signal']) &
            (data['price'] > data['bb_high']) &
            (data['storage_vs_seasonal_pct'] > params['storage_threshold'])
        )
        
        signals_df.loc[buy_conditions, 'ta_signal'] = 1
        signals_df.loc[sell_conditions, 'ta_signal'] = -1
        
        # Add other signals...
        signals_df['combined_signal'] = signals_df['ta_signal']
        
        return signals_df

if __name__ == "__main__":
    main()