import os
from pathlib import Path

import pandas as pd
import streamlit as st

from PPO import run_ppo_experiment


st.set_page_config(page_title="RL Strategy Runner", layout="wide")
st.title("RL Strategy Runner")


def _list_data_files(data_dir: Path):
    if not data_dir.exists():
        return []
    return sorted(str(p) for p in data_dir.glob("*.csv"))


def _infer_stock_name(selected_path: str):
    try:
        return Path(selected_path).stem
    except Exception:
        return "SPY"


def _safe_metric_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


with st.sidebar:
    st.header("Configuration")

    strategy = st.selectbox("Strategy", ["PPO"], index=0)

    data_files = _list_data_files(Path("Data"))
    default_idx = data_files.index("Data/SPY.csv") if "Data/SPY.csv" in data_files else 0
    data_path = st.selectbox("Data source", data_files, index=default_idx if data_files else None)
    custom_data_path = st.text_input("Or custom CSV path", value="")

    stock_name = st.text_input("Output folder / symbol", value=_infer_stock_name(data_path) if data_files else "SPY")

    st.subheader("Date ranges")
    train_start = st.text_input("Train start", value="2012-01-01")
    train_end = st.text_input("Train end", value="2023-12-31")
    test_start = st.text_input("Test start", value="2024-01-01")
    test_end = st.text_input("Test end", value="2025-03-31")

    st.subheader("Training params")
    num_episodes = st.number_input("Episodes", min_value=1, value=100, step=1)
    test_interval = st.number_input("Test interval", min_value=1, value=5, step=1)
    seed = st.number_input("Seed", min_value=0, value=1, step=1)

    st.subheader("Model params")
    init_money = st.number_input("Initial capital", min_value=1000.0, value=500000.0, step=1000.0)
    hidden_dim = st.number_input("Hidden dim", min_value=8, value=64, step=8)
    actor_lr = st.number_input("Actor LR", min_value=1e-6, value=3e-4, step=1e-5, format="%.6f")
    critic_lr = st.number_input("Critic LR", min_value=1e-6, value=1e-3, step=1e-5, format="%.6f")
    lmbda = st.slider("Lambda (GAE)", min_value=0.0, max_value=1.0, value=0.95)
    epochs = st.number_input("PPO epochs", min_value=1, value=4, step=1)
    eps = st.slider("Clip eps", min_value=0.01, max_value=0.4, value=0.1)
    gamma = st.slider("Gamma", min_value=0.8, max_value=0.999, value=0.99)

    st.subheader("Tracking")
    use_mlflow = st.checkbox("Enable MLflow", value=True)
    mlflow_uri = st.text_input("MLflow URI", value="sqlite:///mlflow.db")
    mlflow_experiment = st.text_input("MLflow experiment", value="rl-learningalgos")

    run_clicked = st.button("Run train + backtest", type="primary", use_container_width=True)

if strategy != "PPO":
    st.warning("Only PPO is currently implemented.")

if run_clicked:
    selected_data_path = custom_data_path.strip() or data_path
    if not selected_data_path:
        st.error("No data source selected. Add a file under Data/ or provide custom path.")
        st.stop()

    if not os.path.exists(selected_data_path):
        st.error(f"Data file not found: {selected_data_path}")
        st.stop()

    with st.spinner("Training and backtesting... this can take a while"):
        try:
            results = run_ppo_experiment(
                stock_name=stock_name,
                data_file_path=selected_data_path,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                init_money=float(init_money),
                hidden_dim=int(hidden_dim),
                actor_lr=float(actor_lr),
                critic_lr=float(critic_lr),
                lmbda=float(lmbda),
                epochs=int(epochs),
                eps=float(eps),
                gamma=float(gamma),
                test_interval=int(test_interval),
                num_episodes=int(num_episodes),
                seed=int(seed),
                use_mlflow=bool(use_mlflow),
                mlflow_tracking_uri=mlflow_uri,
                mlflow_experiment=mlflow_experiment,
                draw_plots=False,
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.success("Run completed")

    st.subheader("Run summary")
    st.write({
        "strategy": strategy,
        "stock_name": results["stock_name"],
        "data_path": results["data_path"],
        "best_model_path": results["best_model_path"],
    })

    perf = results.get("best_performance", {})
    if perf:
        st.subheader("Best validation metrics")
        metric_cols = st.columns(min(4, len(perf)))
        for idx, (k, v) in enumerate(perf.items()):
            metric_cols[idx % len(metric_cols)].metric(k, _safe_metric_value(v))

    hist = results.get("history", {})
    if hist and hist.get("train_returns"):
        st.subheader("Training returns")
        train_df = pd.DataFrame({"episode": list(range(1, len(hist["train_returns"]) + 1)), "return": hist["train_returns"]})
        st.line_chart(train_df.set_index("episode"))

    account_df = results.get("backtest_account")
    if isinstance(account_df, pd.DataFrame) and not account_df.empty:
        st.subheader("Backtest outputs")

        if "Capitals" in account_df.columns:
            cap_df = account_df[["Date", "Capitals"]].copy()
            st.line_chart(cap_df.set_index("Date"))

        action_col = "Action" if "Action" in account_df.columns else None
        if action_col is not None:
            action_counts = account_df[action_col].value_counts().rename_axis("Action").reset_index(name="Count")
            st.dataframe(action_counts, use_container_width=True)

        output_csv = Path(results["stock_name"]) / "backtest_results.csv"
        if output_csv.exists():
            st.download_button(
                "Download backtest_results.csv",
                data=output_csv.read_bytes(),
                file_name=output_csv.name,
                mime="text/csv",
            )

    if use_mlflow:
        st.info("MLflow tracked this run. To inspect locally: `mlflow ui --backend-store-uri sqlite:///mlflow.db`")
else:
    st.caption("Configure parameters in the sidebar and click 'Run train + backtest'.")
