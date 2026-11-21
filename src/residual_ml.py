import pickle
import sqlite3
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGB = False

DB_PATH = Path("data/soccer.db")
FEATURES_PATH = Path("data/five_dim_features.csv")
MODEL_OUT_PATH = Path("data/residual_features_enriched.csv")
PKL_MODEL_PATH = Path("data/lgb_residual_model.pkl")

FEATURE_COLS = [
    "elo_diff",
    "xg_lam_diff",
    "sot_efficiency_diff",
    "dc_prob_home",
    "mkt_prob_home",
    "steam_slope_h",
    "steam_slope_a",
    "steam_net_diff",
    "ah_line",
    "ah_eu_divergence",
    "market_overround",
    "rest_days_diff",
]

FEATURE_NAMES_CN = {
    "elo_diff": "動態 Elo 實力差 (elo_diff)",
    "xg_lam_diff": "時間衰減 xG 預期入球差 (xg_lam_diff)",
    "sot_efficiency_diff": "射正轉化率差值 (sot_efficiency_diff)",
    "dc_prob_home": "Dixon-Coles 基準主勝率 (dc_prob_home)",
    "mkt_prob_home": "市場去水隱含主勝率 (mkt_prob_home)",
    "steam_slope_h": "主勝初盤->臨場水位斜率 (steam_slope_h)",
    "steam_slope_a": "客勝初盤->臨場水位斜率 (steam_slope_a)",
    "steam_net_diff": "主客臨場資金淨流向差 (steam_net_diff)",
    "ah_line": "亞洲讓球盤口 (ah_line)",
    "ah_eu_divergence": "亞盤與歐賠背離度 (ah_eu_divergence)",
    "market_overround": "莊家抽水率 (market_overround)",
    "rest_days_diff": "雙方休息天數差 (rest_days_diff)",
}


def enrich_from_sqlite(base_df: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT o.match_id, o.home_odd, o.draw_odd, o.away_odd,
           o.open_home_odd, o.open_away_odd,
           o.asian_line, o.home_ah_odd, o.away_ah_odd,
           s.home_shots, s.away_shots, s.home_sot, s.away_sot
    FROM odds_history o
    LEFT JOIN match_stats s ON o.match_id = s.match_id
    """
    db_df = pd.read_sql_query(query, conn)
    conn.close()

    rows = []
    for _, r in db_df.iterrows():
        open_h, close_h = r["open_home_odd"], r["home_odd"]
        if pd.notna(open_h) and pd.notna(close_h) and open_h > 1.0 and close_h > 1.0:
            p_oh, p_ch = 1.0 / open_h, 1.0 / close_h
            steam_slope_h = (p_ch - p_oh) / p_oh
        else:
            steam_slope_h = 0.0

        open_a, close_a = r["open_away_odd"], r["away_odd"]
        if pd.notna(open_a) and pd.notna(close_a) and open_a > 1.0 and close_a > 1.0:
            p_oa, p_ca = 1.0 / open_a, 1.0 / close_a
            steam_slope_a = (p_ca - p_oa) / p_oa
        else:
            steam_slope_a = 0.0

        ah_line = float(r["asian_line"]) if pd.notna(r["asian_line"]) else 0.0
        h_ah_odd = float(r["home_ah_odd"]) if pd.notna(r["home_ah_odd"]) else 1.90
        a_ah_odd = float(r["away_ah_odd"]) if pd.notna(r["away_ah_odd"]) else 1.90

        ah_implied_adv = -ah_line + (0.5 / h_ah_odd - 0.5 / a_ah_odd)
        if pd.notna(close_h) and pd.notna(close_a) and close_h > 1.0 and close_a > 1.0:
            eu_implied_adv = (1.0 / close_h) - (1.0 / close_a)
        else:
            eu_implied_adv = 0.0
        ah_eu_divergence = ah_implied_adv - (eu_implied_adv * 1.5)

        hs, hst = r["home_shots"], r["home_sot"]
        as_, ast = r["away_shots"], r["away_sot"]
        h_sot_ratio = (float(hst) / float(hs)) if (pd.notna(hs) and hs > 0 and pd.notna(hst)) else 0.33
        a_sot_ratio = (float(ast) / float(as_)) if (pd.notna(as_) and as_ > 0 and pd.notna(ast)) else 0.33

        rows.append({
            "match_id": r["match_id"],
            "steam_slope_h": round(steam_slope_h, 4),
            "steam_slope_a": round(steam_slope_a, 4),
            "steam_net_diff": round(steam_slope_h - steam_slope_a, 4),
            "ah_line": ah_line,
            "ah_eu_divergence": round(ah_eu_divergence, 4),
            "sot_efficiency_diff": round(h_sot_ratio - a_sot_ratio, 4),
        })

    enrich_df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"])
    return pd.merge(base_df, enrich_df, on="match_id", how="left").fillna(0.0)


def _build_regressor(seed: int = 42):
    if HAS_LGB:
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=80,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=15,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=seed,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.03,
        max_leaf_nodes=7,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=seed,
    )


def train_and_evaluate_residual_model():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("找不到 data/five_dim_features.csv，請先執行 `python src/backtest.py`。")

    base_df = pd.read_csv(FEATURES_PATH)
    df = enrich_from_sqlite(base_df)

    df["dt"] = pd.to_datetime(df["match_id"].str.split("_").str[0], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)
    df.to_csv(MODEL_OUT_PATH, index=False)

    X = df[FEATURE_COLS].astype(float)
    y_residual = df["residual_home"].astype(float)
    y_actual = (df["residual_home"] + df["dc_prob_home"]).round().astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    oof_residual_preds = np.zeros(len(df))
    valid_mask = np.zeros(len(df), dtype=bool)
    importances = np.zeros(len(FEATURE_COLS))

    engine_name = "LightGBM (LGBMRegressor)" if HAS_LGB else "HistGradientBoostingRegressor"
    print(f"啟動第二層殘差樹模型訓練 ({engine_name})，共 {len(df)} 筆樣本、{len(FEATURE_COLS)} 維特徵...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = _build_regressor(seed=42 + fold)
        model.fit(X.iloc[train_idx], y_residual.iloc[train_idx])
        oof_residual_preds[val_idx] = model.predict(X.iloc[val_idx])
        if HAS_LGB:
            importances += model.feature_importances_ / 5.0
        valid_mask[val_idx] = True

    dc_base_val = df.loc[valid_mask, "dc_prob_home"].values
    mkt_base_val = df.loc[valid_mask, "mkt_prob_home"].values
    y_true_val = y_actual.loc[valid_mask].values
    corrected_probs = np.clip(dc_base_val + 0.65 * oof_residual_preds[valid_mask], 0.02, 0.98)

    brier_dc = brier_score_loss(y_true_val, dc_base_val)
    brier_ml = brier_score_loss(y_true_val, corrected_probs)
    brier_mkt = brier_score_loss(y_true_val, mkt_base_val)

    ll_dc = log_loss(y_true_val, dc_base_val)
    ll_ml = log_loss(y_true_val, corrected_probs)
    ll_mkt = log_loss(y_true_val, mkt_base_val)

    print("=" * 72)
    print("【第二層樹模型殘差修正：樣本外時間序列驗證結果 (Out-Of-Fold)】")
    print(f"1. 純 Dixon-Coles 基準模型   -> Brier Score: {brier_dc:.4f} | Log-Loss: {ll_dc:.4f}")
    print(f"2. Dixon-Coles + 殘差樹模型 -> Brier Score: {brier_ml:.4f} | Log-Loss: {ll_ml:.4f}  (誤差變化: {brier_ml - brier_dc:+.4f})")
    print(f"3. 頂級莊家市場收盤基準      -> Brier Score: {brier_mkt:.4f} | Log-Loss: {ll_mkt:.4f}")
    print("=" * 72)

    if HAS_LGB and importances.sum() > 0:
        imp_pct = (importances / importances.sum()) * 100.0
        imp_df = pd.DataFrame({
            "排名": range(1, len(FEATURE_COLS) + 1),
            "五維特徵名稱": [FEATURE_NAMES_CN[c] for c in FEATURE_COLS],
            "分裂貢獻度 (%)": np.round(imp_pct, 2),
        }).sort_values("分裂貢獻度 (%)", ascending=False).reset_index(drop=True)
        imp_df["排名"] = range(1, len(imp_df) + 1)

        print("\n【五維殘差修正模型：特徵重要性排行榜 (Feature Importance)】")
        print(imp_df.to_string(index=False))
        print("=" * 72)

    # 使用全量歷史樣本訓練最終正式環境模型並保存至 data/lgb_residual_model.pkl
    final_model = _build_regressor(seed=42)
    final_model.fit(X, y_residual)
    with open(PKL_MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "features": FEATURE_COLS}, f)
    print(f"正式環境殘差修正模型已匯出至：{PKL_MODEL_PATH.resolve()}")


def predict_home_residual_adjustment(feature_dict: dict) -> float:
    """供 app.py 即時呼叫：輸入單場五維特徵，回傳主勝機率的殘差修正值 (ΔP_home)"""
    if not PKL_MODEL_PATH.exists():
        return 0.0
    try:
        with open(PKL_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        cols = bundle["features"]
        row_df = pd.DataFrame([{c: float(feature_dict.get(c, 0.0)) for c in cols}])
        raw_res = float(model.predict(row_df)[0])
        return round(0.65 * raw_res, 4)
    except Exception:
        return 0.0


if __name__ == "__main__":
    train_and_evaluate_residual_model()