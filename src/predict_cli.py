import numpy as np
from model import PoissonMatchPredictor

LEAGUE_MENU = {
    "1": "ENG-Premier League",
    "2": "ESP-La Liga",
    "3": "GER-Bundesliga",
    "4": "ITA-Serie A",
    "5": "FRA-Ligue 1",
}

def remove_vig_1x2(h_odd: float, d_odd: float, a_odd: float):
    inv_h, inv_d, inv_a = 1.0 / h_odd, 1.0 / d_odd, 1.0 / a_odd
    total = inv_h + inv_d + inv_a
    return inv_h / total, inv_d / total, inv_a / total

def interactive_analyze():
    print("\n=== HKJC 個人足球量化分析系統 (Phase 3 校準版) ===")
    print("請選擇聯賽：1.英超  2.西甲  3.德甲  4.意甲  5.法甲")
    choice = input("輸入編號 (預設 1): ").strip() or "1"
    league = LEAGUE_MENU.get(choice, "ENG-Premier League")

    model = PoissonMatchPredictor(league=league, goal_weight=0.4, half_life_days=45.0)
    print(f"\n[{league}] 可用球隊清單：")
    print(", ".join(model.list_teams()))

    home_team = input("\n輸入主隊名稱 (完全匹配上方清單): ").strip()
    away_team = input("輸入客隊名稱 (完全匹配上方清單): ").strip()

    pred = model.predict_match(home_team, away_team)

    print("\n請輸入 HKJC 足智彩賠率（直接按 Enter 可跳過該項）：")
    h_odd = float(input("  主客和 - 主勝賠率: ").strip() or 0)
    d_odd = float(input("  主客和 - 和局賠率: ").strip() or 0)
    a_odd = float(input("  主客和 - 客勝賠率: ").strip() or 0)
    o_odd = float(input("  入球大細 (2.5) - 大賠率: ").strip() or 0)
    u_odd = float(input("  入球大細 (2.5) - 細賠率: ").strip() or 0)
    bankroll = float(input("  總本金 HKD (預設 10000): ").strip() or 10000)

    print(f"\n{'='*66}")
    print(f"【實戰分析報告】{home_team} vs {away_team} ({league})")
    print(f"純模型預期入球 (xG)：主隊 {pred['lambda_home']}  vs  客隊 {pred['lambda_away']}")
    top_cs = ", ".join([f"{s} ({p*100:.1f}%)" for s, p in pred["top_scores"]])
    print(f"最高機率波膽前三：{top_cs}")
    print(f"{'-'*66}")
    print(f"{'玩法選項':<12} | {'校準勝率':<8} | {'HKJC賠率':<8} | {'期望值(EV)':<10} | {'風控建議'}")
    print(f"{'-'*66}")

    candidates = []
    if h_odd > 1 and d_odd > 1 and a_odd > 1:
        mkt_h, mkt_d, mkt_a = remove_vig_1x2(h_odd, d_odd, a_odd)
        candidates.extend([
            ("主勝", 0.70 * mkt_h + 0.30 * pred["prob_home"], h_odd),
            ("和局", 0.70 * mkt_d + 0.30 * pred["prob_draw"], d_odd),
            ("客勝", 0.70 * mkt_a + 0.30 * pred["prob_away"], a_odd),
        ])
    else:
        if h_odd > 1: candidates.append(("主勝(未去水)", pred["prob_home"], h_odd))
        if d_odd > 1: candidates.append(("和局(未去水)", pred["prob_draw"], d_odd))
        if a_odd > 1: candidates.append(("客勝(未去水)", pred["prob_away"], a_odd))

    if o_odd > 1 and u_odd > 1:
        inv_o, inv_u = 1.0 / o_odd, 1.0 / u_odd
        mkt_o = inv_o / (inv_o + inv_u)
        cal_o = 0.70 * mkt_o + 0.30 * pred["prob_over_25"]
        candidates.extend([
            ("入球大(2.5)", cal_o, o_odd),
            ("入球細(2.5)", 1.0 - cal_o, u_odd),
        ])

    for name, prob, odd in candidates:
        ev = prob * odd - 1.0
        in_range = 1.55 <= odd <= 3.10
        valid_ev = 0.015 < ev <= 0.12

        if valid_ev and in_range:
            full_k = ev / (odd - 1.0)
            q_k = min(0.03, full_k * 0.25)
            stake = round(bankroll * q_k, 0)
            advice = f"★ 推薦買入 ${stake:.0f} ({q_k*100:.1f}%)"
        elif ev > 0.12:
            advice = "⚠ 分歧過大(先查傷停)"
        elif ev > 0 and not in_range:
            advice = "⚠ 賠率超出風控區間"
        else:
            advice = "跳過 (無正期望值)"

        print(f"{name:<12} | {prob*100:>6.1f}% | {odd:>8.2f} | {ev*100:>+7.1f}% | {advice}")
    print(f"{'='*66}\n")

if __name__ == "__main__":
    interactive_analyze()