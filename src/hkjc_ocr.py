import re
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    _OCR_ENGINE = RapidOCR()
    HAS_OCR = True
except Exception:
    _OCR_ENGINE = None
    HAS_OCR = False


def parse_hkjc_screenshot(uploaded_file) -> dict:
    """
    接收 Streamlit 上傳的 HKJC 足智彩手機截圖，利用文字座標空間分組自動提取：
    - 主客和 (home_odd, draw_odd, away_odd)
    - 讓球主客和 (hhad_line, hh_odd, hd_odd, ha_odd)
    - 亞洲讓球 (asian_line, h_ah_odd, a_ah_odd)
    - 入球大細 (goal_line, over_g_odd, under_g_odd)
    - 角球大細 (corner_line, over_c_odd, under_c_odd)
    """
    if not HAS_OCR or _OCR_ENGINE is None:
        return {"error": "尚未安裝 rapidocr-onnxruntime，請先在終端機執行 `pip install rapidocr-onnxruntime opencv-python-headless`"}

    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    result, _ = _OCR_ENGINE(img_np)

    if not result:
        return {"error": "未能從圖片中辨識出任何文字，請確認截圖清晰且包含賠率數字。"}

    # 將辨識結果整理為包含中心點 (cx, cy) 的項目列表
    items = []
    for box, text, conf in result:
        if conf < 0.45:
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append({
            "text": str(text).strip(),
            "cx": sum(xs) / 4.0,
            "cy": sum(ys) / 4.0,
            "h": max(ys) - min(ys),
        })

    # 依垂直高度 cy 聚類成同一列 (Row)
    items.sort(key=lambda x: x["cy"])
    rows = []
    for it in items:
        placed = False
        for r in rows:
            avg_y = sum(x["cy"] for x in r) / len(r)
            tol = max(14.0, it["h"] * 0.75)
            if abs(it["cy"] - avg_y) <= tol:
                r.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    for r in rows:
        r.sort(key=lambda x: x["cx"])

    parsed = {}
    current_section = "HAD"
    raw_lines_preview = []

    odd_pattern = re.compile(r"^\d{1,2}\.\d{2}$")
    handicap_pattern = re.compile(r"^[+-]\d+(?:/[+-]?\d+|\.\d+)?$")
    line_bracket_pattern = re.compile(r"\[([+-]?\d+(?:/[+-]?\d+|\.\d+)?)\]")

    for r in rows:
        row_texts = [x["text"] for x in r]
        joined = " ".join(row_texts)
        raw_lines_preview.append(joined)

        # 偵測目前所屬玩法區塊標題
        if any(k in joined for k in ["讓球主客和", "HHAD"]):
            current_section = "HHAD"
        elif any(k in joined for k in ["主客和", "HAD"]) and "讓球" not in joined:
            current_section = "HAD"
        elif any(k in joined for k in ["讓球", "HDC", "亞洲讓球"]) and "主客和" not in joined:
            current_section = "HDC"
        elif any(k in joined for k in ["角球大細", "角球", "Corners", "CHL"]):
            current_section = "CORNER"
        elif any(k in joined for k in ["入球大細", "大細", "HiLo", "Total"]):
            current_section = "HILO"

        # 擷取該列中的所有標準兩位小數賠率
        odds_in_row = []
        for t in row_texts:
            clean_t = t.replace(",", ".").strip()
            if odd_pattern.match(clean_t):
                val = float(clean_t)
                if 1.01 <= val <= 50.0:
                    odds_in_row.append(val)

        # 擷取讓球或球線盤口數字
        m_br = line_bracket_pattern.search(joined)
        found_line = None
        if m_br:
            raw_l = m_br.group(1)
            if "/" in raw_l:
                parts = [float(p) for p in raw_l.split("/")]
                if parts[0] < 0 and parts[1] > 0:
                    parts[1] = -parts[1]
                found_line = sum(parts) / 2.0
            else:
                found_line = float(raw_l)
        else:
            for t in row_texts:
                if handicap_pattern.match(t):
                    if "/" in t:
                        parts = [float(p) for p in t.split("/")]
                        if parts[0] < 0 and parts[1] > 0:
                            parts[1] = -parts[1]
                        found_line = sum(parts) / 2.0
                    else:
                        found_line = float(t)
                    break

        # 依區塊與賠率數量映射至欄位
        if current_section == "HAD" and len(odds_in_row) >= 3 and "home_odd" not in parsed:
            parsed["home_odd"], parsed["draw_odd"], parsed["away_odd"] = odds_in_row[:3]
            current_section = "UNKNOWN"

        elif current_section == "HHAD" and len(odds_in_row) >= 3 and "hh_odd" not in parsed:
            parsed["hh_odd"], parsed["hd_odd"], parsed["ha_odd"] = odds_in_row[:3]
            if found_line is not None and int(found_line) in [-2, -1, 1, 2]:
                parsed["hhad_line"] = int(found_line)
            current_section = "UNKNOWN"

        elif current_section == "HDC" and len(odds_in_row) >= 2 and "h_ah_odd" not in parsed:
            parsed["h_ah_odd"], parsed["a_ah_odd"] = odds_in_row[:2]
            if found_line is not None:
                parsed["asian_line"] = found_line
            current_section = "UNKNOWN"

        elif current_section == "HILO" and len(odds_in_row) >= 2 and "over_g_odd" not in parsed:
            parsed["over_g_odd"], parsed["under_g_odd"] = odds_in_row[:2]
            m_gl = re.search(r"(1\.5|2\.5|3\.5|4\.5)", joined)
            if m_gl:
                parsed["goal_line"] = float(m_gl.group(1))
            current_section = "UNKNOWN"

        elif current_section == "CORNER" and len(odds_in_row) >= 2 and "over_c_odd" not in parsed:
            parsed["over_c_odd"], parsed["under_c_odd"] = odds_in_row[:2]
            m_cl = re.search(r"(7\.5|8\.5|9\.5|10\.5|11\.5|12\.5)", joined)
            if m_cl:
                parsed["corner_line"] = float(m_cl.group(1))
            current_section = "UNKNOWN"

        # 備援規則：若標題未被抓到，依該列出現 3 個賠率自動推斷主客和
        elif "home_odd" not in parsed and len(odds_in_row) == 3 and found_line is None:
            parsed["home_odd"], parsed["draw_odd"], parsed["away_odd"] = odds_in_row

    parsed["raw_preview"] = "\n".join(raw_lines_preview[:25])
    return parsed