# A-Comparative-Study-of-Policy-Based-Reinforcement-Learning-Algorithms
A Comparative Study of Policy-Based Reinforcement Learning Algorithms for Chinese Single-Stock Trading

## Streamlit UI + MLflow Tracking

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Launch the app

```bash
streamlit run streamlit_app.py
```

In the sidebar you can choose:
- Strategy (currently PPO)
- Data source CSV
- Train/test date ranges
- PPO hyperparameters
- MLflow tracking on/off and experiment settings

### 3) Optional: open MLflow dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 to compare runs, parameters, metrics, and artifacts.

## CLI training (PPO)

You can also run from command line:

```bash
python PPO.py --stocks SPY --num_episodes 100 --use_mlflow --no_plots
```

Useful flags:
- `--data_path Data/SPY.csv`
- `--train_start 2012-01-01 --train_end 2023-12-31`
- `--test_start 2024-01-01 --test_end 2025-03-31`
- `--actor_lr 0.0003 --critic_lr 0.001 --gamma 0.99 --eps 0.1`
- `--mlflow_uri sqlite:///mlflow.db --mlflow_experiment rl-learningalgos`
