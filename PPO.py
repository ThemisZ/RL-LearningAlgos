# ==================== Imports ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from datetime import datetime
from tqdm import tqdm
from tabulate import tabulate
import warnings
# import talib  # Not used in this code
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
import os
import argparse
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings('ignore')

AVAILABLE_FEATURES = {
    'Open': 'Open price',
    'High': 'High price',
    'Low': 'Low price',
    'Close': 'Close price',
    'Volume': 'Trading volume',
    'MA_5': '5-day moving average of close',
    'MA_10': '10-day moving average of close',
    'MA_20': '20-day moving average of close',
}
DEFAULT_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# ==================== Random Seed Setup ====================
def setup_seed(seed):
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA convolution results

# ==================== Data Preprocessing Class ====================
class Preprocessor:
    """Preprocess raw market data and compute feature columns."""
    def computeIndicator(self, df, start_date, end_date, selected_features=None):
        """Compute indicators and return processed features with price trend data."""
        df = df[df['Volume'] != 0]  # Filter out zero-volume rows

        # Type conversion
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime format

        # Filter data by time range
        df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        df.reset_index(drop=True, inplace=True)  # Reset index

        # Compute technical indicators
        df['MA_5'] = df['Close'].rolling(5, min_periods=1).mean()   # 5-day moving average
        df['MA_10'] = df['Close'].rolling(10, min_periods=1).mean()  # 10-day moving average
        df['MA_20'] = df['Close'].rolling(20, min_periods=1).mean()  # 20-day moving average
        df.ffill(inplace=True)  # Forward-fill missing values

        # Extract features and price trend
        if selected_features is None:
            selected_features = DEFAULT_FEATURES
        available = [name for name in selected_features if name in df.columns]
        if not available:
            raise ValueError('No valid selected features were found in data.')

        preprocessed_data = df[available].copy()
        trend = df[['Date', 'Open', 'Close']].copy()
        return preprocessed_data, trend

# ==================== Trading Environment Class ====================
class TradingEnv:
    """Trading environment for policy-based reinforcement learning experiments."""
    def __init__(self, df, startingDate, endingDate, init_money=500000, selected_features=None):
        """Initialize the trading environment instance."""
        self.n_actions = 3       # 0: hold, 1: buy/open long, 2: sell/close long
        self.window_size = 20    # State observation window size
        self.init_money = init_money  # Initial capital
        self.transitionCost = 0.003  # Transaction cost
        self.startingDate = startingDate  # Start date
        self.endingDate = endingDate    # End date
        self.raw_df = df.copy()
        self.selected_features = selected_features or DEFAULT_FEATURES

        # Initialize market data
        self.preprocessed_market_data = None
        self.trend = None
        self.account = None
        self.input_size = None
        self.terminal_date = None
        self.t = 0

    def init_market_data(self):
        preprocessor = Preprocessor()
        preprocessed_data, trend_df = preprocessor.computeIndicator(
            self.raw_df, self.startingDate, self.endingDate, selected_features=self.selected_features)
        self.preprocessed_market_data = preprocessed_data
        self.trend = trend_df['Close']

        if len(self.trend) <= self.window_size + 1:
            raise ValueError('Not enough data points for selected date range/window size.')

        # Account table
        self.account = trend_df.copy()
        self.account['Position'] = 0.0  # 1: holding, 0: flat
        self.account['Action'] = 0.0
        self.account['Q_Action'] = 0.0
        self.account['Holdings'] = 0.0
        self.account['Cash'] = float(self.init_money)
        self.account['Capitals'] = self.account['Holdings'] + self.account['Cash'] # Total capital
        self.account['Returns'] = 0.0

        self.input_size = self.preprocessed_market_data.shape[1]
        self.terminal_date = len(self.trend) - 2


    def reset(self, startingPoint=1):
        if self.preprocessed_market_data is None:
            self.init_market_data()
        # Validate starting point

        upper_bound = max(self.window_size - 1, self.terminal_date)
        self.t = int(np.clip(startingPoint, self.window_size - 1, upper_bound))
        # Reset account
        self.account['Position'] = 0.0 # 0: flat, 1: long
        self.account['Action'] = 0.0 # 0: hold, 1: buy, -1: sell
        self.account['Q_Action'] = 0.0 # 0: hold, 1: buy, -1: sell
        self.account['Holdings'] = 0.0 # Holding quantity
        self.account['Cash'] = float(self.init_money)
        self.account['Capitals'] = float(self.init_money)
        self.account['Returns'] = 0.0

        # Position information
        self.holdingNum = 0 # Position size
        self.reward = 0.0 # Current reward

        return self.get_state(self.t)


    def get_state(self, t):
        data_slice = self.preprocessed_market_data.iloc[t + 1 - self.window_size:t + 1, :]
        # Normalize each feature independently to avoid scale mismatch in the window
        state = (data_slice - data_slice.mean(axis=0)) / (data_slice.std(axis=0) + 1e-8)
        #state = (data_slice - data_slice.mean()) / (data_slice.std() + 1e-8)
        return np.ravel(state.values)


    def buy_stock(self, idx):
        """Execute a buy action and update position state."""
        # Compute maximum buyable quantity
        max_pos = int(self.account.loc[idx - 1, 'Cash'] / (self.account.loc[idx, 'Open'] * (1 + self.transitionCost)))
        self.holdingNum = max_pos
        self.account.loc[idx, 'Cash'] = self.account.loc[idx - 1, 'Cash'] - self.holdingNum * self.account.loc[idx, 'Open'] * (1 + self.transitionCost)
        # Update position state
        self.account.loc[idx, 'Position'] = 1.0
        self.account.loc[idx, 'Action'] = 1.0

    def sell_stock(self, idx):
        """Execute a sell action and update cash/position state."""

        self.account.loc[idx, 'Cash'] = self.account.loc[idx - 1, 'Cash'] + self.holdingNum * self.account.loc[idx, 'Open'] * (1 - self.transitionCost)

        self.holdingNum = 0
        self.account.loc[idx, 'Position'] = 0
        self.account.loc[idx, 'Action'] = -1.0

            
    def riskControl(self, idx):
        self.account.loc[idx, 'Cash'] = self.account.loc[idx - 1, 'Cash']
        self.account.loc[idx, 'Action'] = 0.0
        self.account.loc[idx, 'Position'] = self.account.loc[idx - 1, 'Position']
            

    def step(self, action):
        # Decide at t, execute at next bar open (t + 1) to avoid same-bar look-ahead
        execute_idx = self.t + 1

        # Execute action
        if (action == 1) and (self.account.loc[execute_idx - 1, 'Position'] == 0):
            self.account.loc[execute_idx, 'Q_Action'] = 1.0
            self.buy_stock(execute_idx)
        elif action == 2 and (self.account.loc[execute_idx - 1, 'Position'] == 1):
            self.account.loc[execute_idx, 'Q_Action'] = -1.0
            self.sell_stock(execute_idx)
        else: # No operation
            self.account.loc[execute_idx, 'Q_Action'] = 0
            self.riskControl(execute_idx)
        
        # Update account information
        self.account.loc[execute_idx, 'Holdings'] = self.account.loc[execute_idx, 'Position'] * self.holdingNum * self.account.loc[execute_idx, 'Close']
        self.account.loc[execute_idx, 'Capitals'] = self.account.loc[execute_idx, 'Cash'] + self.account.loc[execute_idx, 'Holdings']
        # Compute return rate
        self.account.loc[execute_idx, 'Returns'] = (
            self.account.loc[execute_idx, 'Capitals'] - self.account.loc[execute_idx - 1, 'Capitals']
        ) / self.account.loc[execute_idx - 1, 'Capitals']
        
        # Compute reward - use immediate portfolio return
        current_capital = self.account.loc[execute_idx, 'Cash'] + self.holdingNum * self.account.loc[execute_idx, 'Close']
        prev_capital = self.account.loc[execute_idx - 1, 'Capitals']
        
        # Reward = portfolio return (percentage change)
        portfolio_return = (current_capital - prev_capital) / prev_capital
        
        # Add small penalty for inaction to encourage exploration
        if action == 0:
            self.reward = portfolio_return - 0.0001
        else:
            self.reward = portfolio_return

        # Update time step
        self.t = execute_idx
        
        done = self.t >= self.terminal_date
        if done:
            # Clean invalid data
            self.account = self.account.drop(index=(self.account.loc[(self.account.index > self.t)].index))
        
        next_state = self.get_state(self.t)
        return next_state, self.reward, done

# ==================== PPO Algorithm ====================
class PolicyNet(nn.Module):
    """Policy (actor) network mapping states to action probabilities."""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(nn.Module):
    """Value (critic) network estimating state value."""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_advantage(gamma, lmbda, td_delta, device):
    """Compute generalized advantage estimates (GAE)."""
    td_delta = td_delta.detach().cpu().numpy().flatten()
    advantage = 0.0
    advantages = []
    for delta in reversed(td_delta):
        advantage = delta + gamma * lmbda * advantage
        advantages.append(advantage)
    advantages = torch.tensor(advantages[::-1], dtype=torch.float, device=device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages.view(-1, 1)

class PPO:
    """Proximal Policy Optimization agent implementation."""
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # Discount factor
        self.lmbda = lmbda # GAE parameter
        self.epochs = epochs
        self.eps = eps  # PPO clipping range
        self.device = device

    def take_action(self, state, greedy=False):
        """Sample or select an action from the current policy."""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        if greedy:
            return torch.argmax(probs, dim=1).item()
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item()
        
    def update(self, transition_dict):
        """Update model parameters from collected transitions."""
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Compute TD target and advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta, self.device)


        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()


        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # Add entropy bonus for exploration
            probs = self.actor(states)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
            entropy_bonus = 0.01 * entropy

            actor_loss = torch.mean(-torch.min(surr1, surr2) - entropy_bonus)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        transition_dict.clear()


def cleanup_result_files(output_dir):
    """Remove generated result artifacts from an output folder."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    patterns = [
        'PPO_Best_*.pth',
        'best_account_ep*.csv',
        'train_account.csv',
        'backtest_results.csv',
        'action_events.csv',
        'backtest_summary.csv',
        'IXIC_PPO_*.png',
        'IXIC_PPO_*.eps',
        '*_PPO_*.png',
        '*_PPO_*.eps',
    ]

    removed_files = []
    for pattern in patterns:
        for file_path in output_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                removed_files.append(str(file_path))
    return removed_files


def add_buy_hold_benchmark(account_df, window_size=20):
    """Add Buy & Hold capital/return series using evaluation start as baseline."""
    account = account_df.copy()
    if account.empty or 'Close' not in account.columns:
        return account

    valid_start = min(max(window_size - 1, 0), len(account) - 1)
    start_close = float(account['Close'].iloc[valid_start])
    start_capital = float(account['Capitals'].iloc[valid_start])

    account['BuyHold_Capitals'] = np.nan
    account['BuyHold_Returns'] = 0.0

    if start_close == 0:
        account.loc[account.index[valid_start]:, 'BuyHold_Capitals'] = start_capital
    else:
        scaled_close = account.loc[account.index[valid_start]:, 'Close'] / start_close
        account.loc[account.index[valid_start]:, 'BuyHold_Capitals'] = start_capital * scaled_close

    account.loc[account.index[valid_start]:, 'BuyHold_Returns'] = (
        account.loc[account.index[valid_start]:, 'BuyHold_Capitals']
        .pct_change()
        .fillna(0.0)
    )
    return account


def compute_buy_hold_performance(account_df, window_size=20):
    """Compute Buy & Hold performance table from benchmark series."""
    account = add_buy_hold_benchmark(account_df, window_size)
    account = account.iloc[window_size - 1:].copy()
    if account.empty:
        return [], {}

    pnl = account['BuyHold_Capitals'].iloc[-1] - account['BuyHold_Capitals'].iloc[0]
    cr = ((account['BuyHold_Capitals'].iloc[-1] - account['BuyHold_Capitals'].iloc[0]) / account['BuyHold_Capitals'].iloc[0]) * 100

    start_date = pd.to_datetime(account['Date'].iloc[0])
    end_date = pd.to_datetime(account['Date'].iloc[-1])
    days = max((end_date - start_date).days, 1)
    total_return = (account['BuyHold_Capitals'].iloc[-1] - account['BuyHold_Capitals'].iloc[0]) / account['BuyHold_Capitals'].iloc[0]
    annualized_return = 100 * ((1 + total_return) ** (365 / days) - 1)

    annualized_volatility = 100 * np.sqrt(252) * account['BuyHold_Returns'].std()
    expected_return = account['BuyHold_Returns'].mean()
    volatility = account['BuyHold_Returns'].std()
    sharpe_ratio = 0 if volatility == 0 else np.sqrt(252) * expected_return / volatility

    downside_returns = account['BuyHold_Returns'][account['BuyHold_Returns'] < 0]
    downside_std = downside_returns.std()
    sortino_ratio = 0 if downside_std == 0 else np.sqrt(252) * expected_return / downside_std

    capitals = account['BuyHold_Capitals'].values
    through = np.argmax(np.maximum.accumulate(capitals) - capitals)
    if through == 0:
        max_dd, max_ddd = 0, 0
    else:
        peak = np.argmax(capitals[:through])
        max_dd = 100 * (capitals[peak] - capitals[through]) / capitals[peak]
        max_ddd = through - peak

    metrics_dict = {
        'PnL': float(pnl),
        'Cummulated Return': f'{cr:.2f}%',
        'Annualized Return': f'{annualized_return:.2f}%',
        'Annualized Volatility': f'{annualized_volatility:.2f}%',
        'Sharpe Ratio': float(sharpe_ratio),
        'Sortino Ratio': float(sortino_ratio),
        'Max Drawdown': f'{max_dd:.2f}%',
        'Max DD Duration': f'{max_ddd} days',
    }
    table = [
        ['PnL', f'{pnl:.0f}'],
        ['Cummulated Return', f'{cr:.2f}%'],
        ['Annualized Return', f'{annualized_return:.2f}%'],
        ['Annualized Volatility', f'{annualized_volatility:.2f}%'],
        ['Sharpe Ratio', f'{sharpe_ratio:.3f}'],
        ['Sortino Ratio', f'{sortino_ratio:.3f}'],
        ['Max Drawdown', f'{max_dd:.2f}%'],
        ['Max DD Duration', f'{max_ddd} days'],
    ]
    return table, metrics_dict

# ==================== Performance Evaluation & Visualization ====================
class PerformanceEstimator:
    """Performance Estimator class implementation."""
    def __init__(self, account_df, window_size=20):
        self.window_size = window_size
        self.valid_start = window_size - 1  # Valid data start index
        self.account = account_df.iloc[self.valid_start:]

    def computePnL(self):
        """Compute profit and loss over the evaluation period."""
        start_date = pd.to_datetime(self.account['Date'].iloc[0])
        end_date = pd.to_datetime(self.account['Date'].iloc[-1])
        self.PnL = self.account['Capitals'].iloc[- 1] - self.account['Capitals'].iloc[0]
        return self.PnL

    def computeCummulatedReturn(self):
        """Compute cumulative return over the evaluation period."""
        self.CR = ((self.account['Capitals'].iloc[- 1] - self.account['Capitals'].iloc[0]) / self.account['Capitals'].iloc[0]) * 100
        return self.CR

    def computeAnnualizedReturn(self):
        """Compute annualized return."""
        initial_capital = self.account['Capitals'].iloc[0]
        final_capital = self.account['Capitals'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        start_date = pd.to_datetime(self.account['Date'].iloc[0])
        end_date = pd.to_datetime(self.account['Date'].iloc[-1])
        days = (end_date - start_date).days
        self.annualizedReturn = 0.0 if days == 0 else 100 * ((1 + total_return) ** (365 / days) - 1)
        return self.annualizedReturn

    def computeAnnualizedVolatility(self):
        """Compute annualized volatility."""
        self.annualizedVolatility = 100 * np.sqrt(252) * self.account['Returns'].std()
        return self.annualizedVolatility

    def computeSharpeRatio(self, riskFreeRate=0):
        """Compute Sharpe ratio."""
        expectedReturn = self.account['Returns'].mean()
        volatility = self.account['Returns'].std()
        self.sharpeRatio = 0 if volatility == 0 else np.sqrt(252) * (expectedReturn - riskFreeRate) / volatility
        return self.sharpeRatio

    def computeMaxDrawdown(self):
        """Compute maximum drawdown and duration."""
        capital = self.account['Capitals'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through == 0:
            self.maxDD, self.maxDDD = 0, 0
        else:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through]) / capital[peak]
            self.maxDDD = through - peak
        return self.maxDD, self.maxDDD

    def computeSortinoRatio(self, riskFreeRate=0):
        """Compute Sortino ratio."""
        returns = self.account['Returns']
        expectedReturn = returns.mean() - riskFreeRate
        downside_returns = returns[returns < 0]

        downside_std = downside_returns.std()
        if downside_std == 0:
            self.sortinoRatio = 0
        else:
            self.sortinoRatio = np.sqrt(252) * expectedReturn / downside_std
        return self.sortinoRatio

    def computePerformance(self):
        """Compute all performance metrics and return them as a table."""
        self.computePnL()
        self.computeCummulatedReturn()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        table = [
            ["PnL", f"{self.PnL:.0f}"],
            ["Cummulated Return", f"{self.CR:.2f}%"],
            ["Annualized Return", f"{self.annualizedReturn:.2f}%"],
            ["Annualized Volatility", f"{self.annualizedVolatility:.2f}%"],
            ["Sharpe Ratio", f"{self.sharpeRatio:.3f}"],
            ["Sortino Ratio", f"{self.sortinoRatio:.3f}"],
            ["Max Drawdown", f"{self.maxDD:.2f}%"],
            ["Max DD Duration", f"{self.maxDDD} days"],
        ]
        return table

class Visualizer:
    """Initialize the visualizer with account data."""
    def __init__(self, account_df, window_size=20, stock_name='SPY'):
        self.window_size = window_size
        self.valid_start = window_size - 1  # Valid data start index
        self.account = add_buy_hold_benchmark(account_df, window_size).iloc[self.valid_start:]
        self.stock_name = stock_name
    def draw_final(self):
        """Plot and save the equity curve."""
        plt.clf()
        x_axis = pd.to_datetime(self.account['Date'])
        plt.plot(x_axis, self.account['Capitals'], label='PPO Capitals')
        if 'BuyHold_Capitals' in self.account.columns:
            plt.plot(x_axis, self.account['BuyHold_Capitals'], label='Buy & Hold Capitals', linestyle='--')

        buy_mask = self.account['Action'] == 1.0
        sell_mask = self.account['Action'] == -1.0
        plt.scatter(x_axis[buy_mask], self.account.loc[buy_mask, 'Capitals'], marker='^', s=36, color='green', label='Buy')
        plt.scatter(x_axis[sell_mask], self.account.loc[sell_mask, 'Capitals'], marker='v', s=36, color='red', label='Sell')

        plt.grid()
        plt.xlabel('Date')
        plt.ylabel('Capitals')
        plt.xticks(rotation=30)
        plt.legend()
        plt.savefig(f'{self.stock_name}/IXIC_PPO_Capitals.eps', format='eps', dpi=1000, bbox_inches='tight')
        plt.savefig(f'{self.stock_name}/IXIC_PPO_Capitals.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def draw(self):
        """Plot and save buy/sell trading signals."""
        fig, ax1 = plt.subplots(figsize=(12, 5))
        x_axis = pd.to_datetime(self.account['Date'])
        ax1.plot(x_axis, self.account['Close'], color='royalblue', lw=0.8, label='Market Price')
        ax1.plot(x_axis[self.account['Action'] == 1.0],
                 self.account.loc[self.account['Action'] == 1.0, 'Close'], '^', markersize=6, color='green', label='Buy')
        ax1.plot(x_axis[self.account['Action'] == -1.0],
                 self.account.loc[self.account['Action'] == -1.0, 'Close'], 'v', markersize=6, color='red', label='Sell')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price')
        ax1.tick_params(axis='x', rotation=30)
        ax1.legend(loc='upper center', ncol=3, frameon=False)
        plt.savefig(f'{self.stock_name}/IXIC_PPO_Actions.eps', format='eps', dpi=1000, bbox_inches='tight')
        plt.savefig(f'{self.stock_name}/IXIC_PPO_Actions.png', dpi=1000, bbox_inches='tight')
        plt.show()

# ==================== Training Pipeline ====================
def train_with_validation(train_env, val_env, agent, num_episodes=800, test_interval=10, stock_name='SPY', draw_plots=True):
    """Train the agent and periodically validate on the test environment."""
    history = {
        'train_returns': [],
        'test_pnl': [],
        'test_returns': [],
        'test_sharpe': [],
        'test_max_dd': []
    }
    best_pnl = -np.inf # Track best test PnL
    bestModelDir = []
    best_buy_hold_perf = {}

    for i_episode in tqdm(range(num_episodes), desc='Training'):
        # Training phase
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = train_env.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = train_env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(float(done))
            state = next_state
            episode_return += reward
        
        # Add debugging info (before update clears transition_dict)
        if (i_episode + 1) % 10 == 0:
            buy_count = sum(1 for a in transition_dict['actions'] if a == 1)
            sell_count = sum(1 for a in transition_dict['actions'] if a == 2)
            avg_reward = np.mean(transition_dict['rewards']) if transition_dict['rewards'] else 0
            print(f"\nEpisode {i_episode+1} Training Stats:")
            print(f"  Buy actions: {buy_count}")
            print(f"  Sell actions: {sell_count}")
            print(f"  Avg reward: {avg_reward:.6f}")
            print(f"  Episode return: {episode_return:.6f}")
        
        agent.update(transition_dict)
        history['train_returns'].append(episode_return)

        # Periodic validation
        if (i_episode + 1) >= test_interval:
            val_env.init_market_data()  # Ensure each validation run uses an independent environment
            state = val_env.reset()
            done = False
            while not done:
                action = agent.take_action(state, greedy=True)
                next_state, _, done = val_env.step(action)
                state = next_state

            # Performance evaluation
            est = PerformanceEstimator(val_env.account, val_env.window_size)
            est.computePerformance()
            history['test_pnl'].append(est.PnL)
            history['test_returns'].append(est.CR)
            history['test_sharpe'].append(est.sharpeRatio)
            history['test_max_dd'].append(est.maxDD)
            print("episode",i_episode + 1,"test_pnl",history['test_pnl'][ -1],"test_returns",history['test_returns'][ -1],"test_sharpe",history['test_sharpe'][ - 1])
            # Save best model
            if est.PnL > best_pnl:
                best_pnl = est.PnL
                best_perf = est.computePerformance()
                _, best_buy_hold_perf = compute_buy_hold_performance(val_env.account, val_env.window_size)
                torch.save(agent.actor.state_dict(), '{}/PPO_Best_{}.pth'.format(stock_name, i_episode+1))
                bestModelDir.append('{}/PPO_Best_{}.pth'.format(stock_name, i_episode+1))
                best_account = add_buy_hold_benchmark(val_env.account, val_env.window_size)
                best_account['Buy_Marker'] = (best_account['Action'] == 1.0).astype(int)
                best_account['Sell_Marker'] = (best_account['Action'] == -1.0).astype(int)
                best_account.to_csv(f'{stock_name}/best_account_ep{i_episode+1}_pnl{best_pnl:.0f}.csv', index=False)
    if draw_plots:
        dra = Visualizer(val_env.account, val_env.window_size, stock_name)
        dra.draw()
    train_account = add_buy_hold_benchmark(train_env.account, train_env.window_size)
    train_account['Buy_Marker'] = (train_account['Action'] == 1.0).astype(int)
    train_account['Sell_Marker'] = (train_account['Action'] == -1.0).astype(int)
    train_account.to_csv(f'{stock_name}/train_account.csv', index=False)
    print(f'Training complete. Best test-set PnL: {best_pnl:.2f}')
    print("\n" + "="*40 + " Validation Backtest Results " + "="*40)
    output_table = tabulate(best_perf, headers=['Metric', 'Value'], tablefmt='fancy_grid')
    print(output_table)
    if best_buy_hold_perf:
        print("\n" + "="*40 + " Buy & Hold (Validation Start Baseline) " + "="*40)
        buy_hold_output = tabulate([[k, v] for k, v in best_buy_hold_perf.items()], headers=['Metric', 'Value'], tablefmt='fancy_grid')
        print(buy_hold_output)
    # Write to txt file
    # with open('output.txt', 'w', encoding='utf-8') as f:
    #     f.write(output_table)
    file_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    performance_dict = dict(best_perf)
    df1 = pd.DataFrame([performance_dict])
    df1['model_name'] = file_name_without_extension
    df1['stock_name'] = stock_name
    save_csv_name = f'best_performance_{file_name_without_extension}.csv'
    if os.path.exists(save_csv_name):
        df1.to_csv(save_csv_name, index=False, mode='a', header=False)
    else:
        df1.to_csv(save_csv_name, index=False)
    return history, bestModelDir[-1], performance_dict, best_buy_hold_perf

# ==================== Backtest Function ====================
def backtest(
    model_path,
    env,
    greedy,
    hidden_dim,
    actor_lr,
    critic_lr,
    lmbda,
    epochs,
    eps,
    gamma,
    device,
    stock_name='SPY',
    visualize=True
):
    """Run backtest using a saved model and report metrics."""
    # Initialize PPO agent
    agent = PPO(state_dim=env.input_size * env.window_size,
                hidden_dim=hidden_dim,
                action_dim=env.n_actions,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                lmbda=lmbda,
                epochs=epochs,
                eps=eps,
                gamma=gamma,
                device=device)
    
    # Load best model
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()
    
    # Run backtest
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state, greedy=greedy)
        next_state, _, done = env.step(action)
        state = next_state
    
    # Performance evaluation
    env.account = add_buy_hold_benchmark(env.account, env.window_size)
    env.account['Buy_Marker'] = (env.account['Action'] == 1.0).astype(int)
    env.account['Sell_Marker'] = (env.account['Action'] == -1.0).astype(int)

    estimator = PerformanceEstimator(env.account, env.window_size)
    estimator.computePerformance()
    perf_table = estimator.computePerformance()
    buy_hold_table, buy_hold_metrics = compute_buy_hold_performance(env.account, env.window_size)
    print("\n" + "="*40 + " Backtest Results " + "="*40)
    print(tabulate(perf_table, headers=['Metric', 'Value'], tablefmt='fancy_grid'))
    print("\n" + "="*40 + " Buy & Hold Comparison (Backtest Start Baseline) " + "="*40)
    print(tabulate(buy_hold_table, headers=['Metric', 'Value'], tablefmt='fancy_grid'))
    
    # Visualize results
    if visualize:
        visualizer = Visualizer(env.account, env.window_size, stock_name)
        visualizer.draw_final()
        visualizer.draw()
    
    # Save detailed trade records
    env.account.to_csv(f'{stock_name}/backtest_results.csv', index=False)
    action_events = env.account.loc[env.account['Action'] != 0, ['Date', 'Close', 'Action', 'Capitals']].copy()
    action_events['Action_Label'] = action_events['Action'].map({1.0: 'BUY', -1.0: 'SELL'})
    action_events.to_csv(f'{stock_name}/action_events.csv', index=False)

    summary_df = pd.DataFrame({
        'Metric': [k for k, _ in perf_table] + [f'BuyHold {k}' for k, _ in buy_hold_table],
        'Value': [v for _, v in perf_table] + [v for _, v in buy_hold_table]
    })
    summary_df.to_csv(f'{stock_name}/backtest_summary.csv', index=False)

    # Additional analysis
    print("\n" + "="*40 + " Trade Analysis " + "="*40)
    print(f"Total number of trades: {len(env.account[env.account['Action'] != 0])}")
    print(f"Average holding days: {env.account[env.account['Position'] == 1].shape[0]/len(env.account['Position'].unique()):.1f}")
    print(f"Maximum single-day loss: {env.account['Returns'].min()*100:.2f}%")
    print(f"Winning trade ratio: {len(env.account[env.account['Returns'] > 0])/len(env.account)*100:.1f}%")
    
    return env.account, perf_table, buy_hold_metrics
    
def run_ppo_experiment(
    stock_name='SPY',
    data_file_path=None,
    train_start='2012-01-01',
    train_end='2023-12-31',
    test_start='2024-01-01',
    test_end='2025-03-31',
    init_money=500000,
    hidden_dim=64,
    actor_lr=3e-4,
    critic_lr=1e-3,
    lmbda=0.95,
    epochs=4,
    eps=0.1,
    gamma=0.99,
    test_interval=5,
    num_episodes=100,
    seed=1,
    use_mlflow=False,
    mlflow_tracking_uri='sqlite:///mlflow.db',
    mlflow_experiment='rl-learningalgos',
    draw_plots=True,
    clean_output_dir=True,
    selected_features=None
):
    setup_seed(seed)
    data_path = data_file_path or f'Data/{stock_name}.csv'
    df_raw = pd.read_csv(data_path)
    df_raw = df_raw.dropna()
    if 'Adj Close' in df_raw.columns:
        df_raw = df_raw.drop(['Adj Close'], axis=1)

    if not os.path.exists(stock_name):
        os.mkdir(stock_name)
    elif clean_output_dir:
        removed = cleanup_result_files(stock_name)
        if removed:
            print(f'Cleaned {len(removed)} previous result files from {stock_name}.')

    train_startingDate = datetime.strptime(train_start, '%Y-%m-%d')
    train_endingDate = datetime.strptime(train_end, '%Y-%m-%d')
    test_startingDate = datetime.strptime(test_start, '%Y-%m-%d')
    test_endingDate = datetime.strptime(test_end, '%Y-%m-%d')

    if selected_features is None:
        selected_features = DEFAULT_FEATURES
    selected_features = [feature for feature in selected_features if feature in AVAILABLE_FEATURES]
    if not selected_features:
        raise ValueError('At least one valid feature must be selected.')

    train_window_df = df_raw.loc[
        (pd.to_datetime(df_raw['Date']) >= train_startingDate)
        & (pd.to_datetime(df_raw['Date']) <= train_endingDate)
    ].copy()
    if len(train_window_df) < 60:
        raise ValueError('Training window too small to create train/validation split.')

    split_idx = int(len(train_window_df) * 0.8)
    split_idx = min(max(split_idx, 40), len(train_window_df) - 20)
    split_date = pd.to_datetime(train_window_df['Date'].iloc[split_idx]).to_pydatetime()
    train_endingDate_effective = pd.to_datetime(train_window_df['Date'].iloc[split_idx - 1]).to_pydatetime()

    print(
        f'Bias-safe split enabled: train={train_startingDate.date()}..{train_endingDate_effective.date()}, '
        f'validation={split_date.date()}..{train_endingDate.date()}, test={test_startingDate.date()}..{test_endingDate.date()}'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_train = TradingEnv(
        df_raw,
        train_startingDate,
        train_endingDate_effective,
        init_money=init_money,
        selected_features=selected_features
    )
    env_train.init_market_data()
    env_val = TradingEnv(
        df_raw,
        split_date,
        train_endingDate,
        init_money=init_money,
        selected_features=selected_features
    )
    env_val.init_market_data()
    env_test = TradingEnv(
        df_raw,
        test_startingDate,
        test_endingDate,
        init_money=init_money,
        selected_features=selected_features
    )
    env_test.init_market_data()

    agent = PPO(
        state_dim=env_train.input_size * env_train.window_size,
        hidden_dim=hidden_dim,
        action_dim=env_train.n_actions,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        lmbda=lmbda,
        epochs=epochs,
        eps=eps,
        gamma=gamma,
        device=device
    )

    mlflow_run = None
    mlflow_client = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment)
            mlflow_client = mlflow
            mlflow_run = mlflow.start_run(run_name=f'PPO_{stock_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            mlflow.log_params({
                'strategy': 'PPO',
                'stock_name': stock_name,
                'data_path': data_path,
                'train_start': train_start,
                'train_end': train_end,
                'train_end_effective': train_endingDate_effective.strftime('%Y-%m-%d'),
                'val_start': split_date.strftime('%Y-%m-%d'),
                'val_end': train_endingDate.strftime('%Y-%m-%d'),
                'test_start': test_start,
                'test_end': test_end,
                'init_money': init_money,
                'hidden_dim': hidden_dim,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
                'lmbda': lmbda,
                'epochs': epochs,
                'eps': eps,
                'gamma': gamma,
                'test_interval': test_interval,
                'num_episodes': num_episodes,
                'seed': seed,
                'device': str(device),
                'selected_features': ','.join(selected_features)
            })
        except Exception as exc:
            print(f'[WARN] MLflow initialization failed: {exc}')

    try:
        history, best_model_path, best_perf_dict, best_buy_hold_perf = train_with_validation(
            env_train,
            env_val,
            agent,
            num_episodes=num_episodes,
            test_interval=test_interval,
            stock_name=stock_name,
            draw_plots=draw_plots
        )

        account_df, performance, backtest_buy_hold_performance = backtest(
            best_model_path,
            env_test,
            greedy=True,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            gamma=gamma,
            device=device,
            stock_name=stock_name,
            visualize=draw_plots
        )

        if mlflow_client is not None:
            mlflow_client.log_metrics({
                'best_pnl': float(best_perf_dict.get('PnL', 0)),
                'best_cum_return_pct': float(str(best_perf_dict.get('Cummulated Return', '0')).replace('%', '')),
                'best_annualized_return_pct': float(str(best_perf_dict.get('Annualized Return', '0')).replace('%', '')),
                'best_annualized_volatility_pct': float(str(best_perf_dict.get('Annualized Volatility', '0')).replace('%', '')),
                'best_sharpe_ratio': float(best_perf_dict.get('Sharpe Ratio', 0)),
                'best_sortino_ratio': float(best_perf_dict.get('Sortino Ratio', 0)),
                'best_max_drawdown_pct': float(str(best_perf_dict.get('Max Drawdown', '0')).replace('%', '')),
            })
            mlflow_client.log_artifact(best_model_path)
            mlflow_client.log_artifact(f'{stock_name}/train_account.csv')
            mlflow_client.log_artifact(f'{stock_name}/backtest_results.csv')
            mlflow_client.log_artifact(f'{stock_name}/action_events.csv')
            mlflow_client.log_artifact(f'{stock_name}/backtest_summary.csv')

        return {
            'stock_name': stock_name,
            'data_path': data_path,
            'best_model_path': best_model_path,
            'history': history,
            'backtest_account': account_df,
            'performance_table': performance,
            'best_performance': best_perf_dict,
            'best_buy_hold_performance': best_buy_hold_perf,
            'backtest_buy_hold_performance': backtest_buy_hold_performance,
            'selected_features': selected_features,
            'validation_range': {
                'start': split_date.strftime('%Y-%m-%d'),
                'end': train_endingDate.strftime('%Y-%m-%d')
            }
        }
    finally:
        if mlflow_run is not None:
            try:
                mlflow_client.end_run()
            except Exception:
                pass


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='PPO stock trading')
    parser.add_argument('--stocks', default='SPY', help='Stock ticker name used for output folder naming')
    parser.add_argument('--data_path', default='', help='Path to stock CSV. If empty, Data/{stocks}.csv is used')
    parser.add_argument('--train_start', default='2012-01-01')
    parser.add_argument('--train_end', default='2023-12-31')
    parser.add_argument('--test_start', default='2024-01-01')
    parser.add_argument('--test_end', default='2025-03-31')
    parser.add_argument('--init_money', type=float, default=500000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_mlflow', action='store_true', help='Enable MLflow tracking')
    parser.add_argument('--mlflow_uri', default='sqlite:///mlflow.db', help='MLflow tracking URI')
    parser.add_argument('--mlflow_experiment', default='rl-learningalgos', help='MLflow experiment name')
    parser.add_argument('--no_plots', action='store_true', help='Disable matplotlib display and plotting')
    parser.add_argument('--keep_results', action='store_true', help='Keep existing result artifacts in output folder')
    parser.add_argument('--features', default=','.join(DEFAULT_FEATURES), help='Comma-separated feature list')
    return parser


# ==================== Main Program ====================
if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_ppo_experiment(
        stock_name=args.stocks,
        data_file_path=args.data_path if args.data_path else None,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        init_money=args.init_money,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        lmbda=args.lmbda,
        epochs=args.epochs,
        eps=args.eps,
        gamma=args.gamma,
        test_interval=args.test_interval,
        num_episodes=args.num_episodes,
        seed=args.seed,
        use_mlflow=args.use_mlflow,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        draw_plots=not args.no_plots,
        clean_output_dir=not args.keep_results,
        selected_features=[feature.strip() for feature in args.features.split(',') if feature.strip()]
    )
