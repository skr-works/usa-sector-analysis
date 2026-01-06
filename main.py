import yfinance as yf
import pandas as pd
import numpy as np
import json
import datetime
import os
import requests
import base64
import random
from concurrent.futures import ThreadPoolExecutor

# --- 設定: 米国11セクター ETFリスト (GICS分類) ---
SECTOR_ETFS = {
    "XLC": "通信サービス",
    "XLY": "一般消費財",
    "XLP": "生活必需品",
    "XLE": "エネルギー",
    "XLF": "金融",
    "XLV": "ヘルスケア",
    "XLI": "資本財",
    "XLB": "素材",
    "XLRE": "不動産",
    "XLK": "テクノロジー",
    "XLU": "公益事業"
}

def calculate_technical_indicators(df):
    """データフレーム全体に対してテクニカル指標を一括計算する"""
    df = df.copy()
    
    # 1. 移動平均乖離率
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma25'] = df['Close'].rolling(window=25).mean()
    df['ma75'] = df['Close'].rolling(window=75).mean()
    
    df['diff_short'] = ((df['Close'] - df['ma5']) / df['ma5']) * 100
    df['diff_mid'] = ((df['Close'] - df['ma25']) / df['ma25']) * 100
    df['diff_long'] = ((df['Close'] - df['ma75']) / df['ma75']) * 100

    # 2. RSI (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 3. ボリンジャーバンド %B (20日, 2σ)
    df['bb_ma'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_up'] = df['bb_ma'] + (df['bb_std'] * 2)
    df['bb_low'] = df['bb_ma'] - (df['bb_std'] * 2)
    
    bb_range = df['bb_up'] - df['bb_low']
    df['bb_pct_b'] = np.where(bb_range == 0, 0, (df['Close'] - df['bb_low']) / bb_range)

    # 4. 出来高倍率 (直近5日平均との比較)
    df['vol_ma5'] = df['Volume'].rolling(window=5).mean()
    df['vol_ratio'] = np.where(df['vol_ma5'] == 0, 0, df['Volume'] / df['vol_ma5'])

    # 5. 前日比
    df['change_pct'] = df['Close'].pct_change() * 100

    return df

def get_sector_data(code, name):
    """指定銘柄のデータを取得・計算し、辞書のリストとして返す"""
    try:
        stock = yf.Ticker(code)
        # 過去2年分取得
        hist = stock.history(period="2y")
        
        if hist.empty:
            return []

        # 指標計算
        df = calculate_technical_indicators(hist)
        
        # NaNを除去し、直近300営業日分に絞る（チャート用）
        df = df.dropna().tail(300) 

        # 行データ作成用ヘルパー関数
        def make_row(date_idx, row):
            return {
                "コード": code,
                "セクター名": name,
                "日付": date_idx.strftime('%Y-%m-%d'),
                "現在値": round(row['Close'], 2),
                "前日比(%)": round(row['change_pct'], 2),
                "短期(5日乖離)": round(row['diff_short'], 2),
                "中期(25日乖離)": round(row['diff_mid'], 2),
                "長期(75日乖離)": round(row['diff_long'], 2),
                "RSI": round(row['rsi'], 1),
                "BB%B(過熱)": round(row['bb_pct_b'], 2),
                "出来高倍率": round(row['vol_ratio'], 2),
            }

        results = []
        for date_idx, row in df.iterrows():
            results.append(make_row(date_idx, row))
            
        return results

    except Exception as e:
        print(f"Error {code}: {e}")
        return []

def process_data_for_chart(all_rows):
    """取得した生データをチャートとパネル用に加工"""
    if not all_rows:
        return None, None, None, None

    df = pd.DataFrame(all_rows)
    df['日付'] = pd.to_datetime(df['日付'])
    
    # 重複排除
    df = df.sort_values(['日付', 'コード'])
    df = df.drop_duplicates(subset=['日付', 'コード'], keep='last')

    # 1. 最新データの抽出 (パネル用)
    latest_df = df.sort_values('日付').groupby('コード').tail(1).copy()
    
    latest_df['sort_key'] = latest_df['コード'].apply(lambda x: list(SECTOR_ETFS.keys()).index(x) if x in SECTOR_ETFS else 99)
    latest_df = latest_df.sort_values('sort_key')

    # 2. 時系列データの作成 (チャート用 - 起点100)
    pivot_df = df.pivot(index='日付', columns='セクター名', values='現在値')
    
    if not pivot_df.empty:
        base_prices = pivot_df.iloc[0]
        normalized_df = pivot_df.div(base_prices).mul(100).round(2)
    else:
        normalized_df = pivot_df

    # 3. 過熱ランキングTop3
    overheated_sectors = []
    if not latest_df.empty and not normalized_df.empty:
        for _, row in latest_df.iterrows():
            sector = row['セクター名']
            rsi = float(row['RSI'])
            bb = float(row['BB%B(過熱)'])
            
            if rsi >= 70 or bb > 1.0:
                current_index_val = 0
                if sector in normalized_df.columns:
                    current_index_val = normalized_df[sector].iloc[-1]
                
                overheated_sectors.append({
                    'sector': sector,
                    'index_val': current_index_val,
                    'rsi': rsi
                })
        
        overheated_sectors.sort(key=lambda x: x['rsi'], reverse=True)
        overheated_top3 = overheated_sectors[:3]
    else:
        overheated_top3 = []

    # Chart.js用データ
    chart_labels = normalized_df.index.strftime('%Y/%m/%d').tolist()
    chart_datasets = []
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79'
    ]
    
    for i, column in enumerate(normalized_df.columns):
        color = colors[i % len(colors)]
        dataset = {
            "label": column,
            "data": normalized_df[column].ffill().tolist(),
            "borderColor": color,
            "backgroundColor": color,
            "borderWidth": 2,
            "pointRadius": 0,
            "pointHoverRadius": 4,
            "fill": False,
            "tension": 0.1
        }
        chart_datasets.append(dataset)

    return latest_df, chart_labels, chart_datasets, overheated_top3

def generate_html_content(latest_df, chart_labels, chart_datasets, overheated_top3):
    """HTMLコンテンツ生成"""
    if latest_df is None or latest_df.empty:
        return "<p>データ取得に失敗しました。</p>"

    last_update_str = latest_df['日付'].max().strftime('%Y-%m-%d')
    chart_id = f"sectorChart_{random.randint(1000, 9999)}"
    
    # グリッドスタイルを修正: 1fr 1fr の2列固定
    style_grid = "display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px;"

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto;">
        <p style="text-align: right; font-size: 0.8rem; color: #666; margin-bottom: 10px;">データ更新日: {last_update_str}</p>
        
        <h3 style="font-size: 1.1rem; margin-bottom: 15px; color: #333;">短期の過熱割安判定パネル</h3>

        <div style="{style_grid}">
    """

    for _, row in latest_df.iterrows():
        sector = row['セクター名']
        # code = row['コード'] # 米国版コードは非表示（参考画像に合わせる）
        change = float(row['前日比(%)'])
        rsi = float(row['RSI'])
        bb = float(row['BB%B(過熱)'])
        
        # デフォルト設定 (通常)
        status_text = "通常"
        status_style = "color: #aaa; font-size: 0.7rem; background: #f7f7f7; padding: 2px 6px; border-radius: 4px; display: inline-block;"
        
        card_bg = "#fff"
        card_border = "1px solid #eee"

        # 過熱 (赤系)
        if rsi >= 70 or bb > 1.0:
            status_text = "過熱"
            status_style = (
                "color: #fff; font-weight: 900; font-size: 1.1rem; "
                "background: #d32f2f; padding: 6px 12px; border-radius: 6px; "
                "box-shadow: 0 3px 6px rgba(211, 47, 47, 0.4); "
                "display: inline-block; transform: scale(1.05);"
            )
            card_bg = "#ffebee" 
            card_border = "2px solid #ef5350"
            
        # 割安 (青系)
        elif rsi <= 30 or bb < 0:
            status_text = "割安"
            status_style = (
                "color: #fff; font-weight: 900; font-size: 1.1rem; "
                "background: #1976d2; padding: 6px 12px; border-radius: 6px; "
                "box-shadow: 0 3px 6px rgba(25, 118, 210, 0.4); "
                "display: inline-block; transform: scale(1.05);"
            )
            card_bg = "#e3f2fd"
            card_border = "2px solid #42a5f5"

        change_color = "#d32f2f" if change > 0 else ("#1976d2" if change < 0 else "#333")
        sign = "+" if change > 0 else ""
        
        # パネルHTML生成
        html += f"""
        <div style="padding: 12px; border-radius: 6px; background: {card_bg}; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: {card_border};">
            <div style="font-weight: bold; font-size: 0.95rem; color: #333; margin-bottom: 8px;">{sector}</div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <div style="font-size: 0.7rem; color: #888; margin-bottom: 2px;">ETF価格前日比</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: {change_color}; line-height: 1;">
                        {sign}{change}<span style="font-size: 0.8rem;">%</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="{status_style}">{status_text}</div>
                </div>
            </div>
            
            <div style="font-size: 0.75rem; color: #666; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 6px; display: flex; justify-content: space-between;">
                <span>RSI(14): <strong>{rsi:.1f}</strong></span>
                <span>BB: <strong>{bb:.2f}</strong></span>
            </div>
        </div>
        """

    # パネル下の説明エリア
    html += """
        </div>
        <div style="font-size: 0.8rem; color: #666; background: #f9f9f9; padding: 12px; border-radius: 6px; margin-bottom: 40px; border: 1px solid #eee;">
            <strong>【パネルの見方・判定条件】</strong><br>
            <ul style="margin: 5px 0 0 20px; padding: 0;">
                <li><strong>ETF価格前日比</strong>：米国セクターETF終値の前日比です。</li>
                <li><strong>過熱</strong>：RSI(14日)が70以上、またはボリンジャーバンド(20日/2σ)の%Bが1.0(バンド上限)を超えた場合。</li>
                <li><strong>割安</strong>：RSI(14日)が30以下、またはボリンジャーバンド(20日/2σ)の%Bが0(バンド下限)を下回った場合。</li>
                <li><strong>BB</strong>：ボリンジャーバンド%B値。1.0以上でバンド上限突破、0以下でバンド下限割れを示唆します。</li>
            </ul>
        </div>
    """

    # Top3 表示 (日本語化)
    top3_html = ""
    if overheated_top3:
        top3_html += '<div style="background: #fff3e0; padding: 12px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #ffe0b2;">'
        top3_html += '<div style="font-weight:bold; color:#e65100; margin-bottom:8px; font-size:0.95rem;">上昇トレンド × 過熱シグナル発生中 (Top 3)</div>'
        top3_html += '<ul style="margin: 0; padding-left: 20px; color: #333; font-size: 0.9rem;">'
        for item in overheated_top3:
            idx_val = round(item['index_val'], 1)
            top3_html += f"<li><strong>{item['sector']}</strong> <span style='color:#666; font-size:0.85rem;'>(300日指数: {idx_val} / RSI: {item['rsi']})</span></li>"
        top3_html += '</ul></div>'

    json_labels = json.dumps(chart_labels)
    json_datasets = json.dumps(chart_datasets)

    html += f"""
        <h3 style="font-size: 1.1rem; margin-top: 40px; margin-bottom: 15px; color: #333;">長期の過熱割安判定チャート(起点100)</h3>
        
        {top3_html}
        
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        
        <div style="position: relative; width: 100%; height: 500px; border: 1px solid #eee; border-radius: 4px; padding: 5px;">
            <canvas id="{chart_id}"></canvas>
        </div>
        
        <div style="font-size: 0.8rem; color: #666; background: #f9f9f9; padding: 12px; border-radius: 6px; margin-top: 15px; border: 1px solid #eee;">
            <strong>【チャートの仕様】</strong><br>
            <ul style="margin: 5px 0 0 20px; padding: 0;">
                <li>直近300営業日前の終値を「100」として指数化したパフォーマンス推移です。</li>
                <li>グラフ上の凡例の四角(●)をタップすると、その業種の表示/非表示を切り替えられます。</li>
                <li>チャート上の点をタップすると、詳細な日付と指数値が表示されます。</li>
            </ul>
        </div>
        
        <script>
        (function() {{
            const ctx = document.getElementById('{chart_id}').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json_labels},
                    datasets: {json_datasets}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    plugins: {{
                        legend: {{ 
                            position: 'bottom', 
                            labels: {{ 
                                usePointStyle: true,
                                boxWidth: 8, 
                                padding: 15,
                                font: {{ size: 11 }}
                            }} 
                        }},
                        tooltip: {{ position: 'nearest' }}
                    }},
                    scales: {{
                        y: {{ 
                            title: {{ display: true, text: '指数' }},
                            grid: {{ color: '#f0f0f0' }} 
                        }},
                        x: {{ 
                            grid: {{ display: false }}, 
                            ticks: {{ maxTicksLimit: 10 }} 
                        }}
                    }},
                    elements: {{ 
                        point: {{ 
                            radius: 0, 
                            hitRadius: 10, 
                            hoverRadius: 5 
                        }} 
                    }}
                }}
            }});
        }})();
        </script>
    </div>
    """
    return html

def parse_config_from_env():
    """1つの環境変数から設定を解析する"""
    raw_config = os.environ.get("CORE_SYSTEM_CONFIG", "")
    config = {}
    
    # 行ごとに分割して処理
    for line in raw_config.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
            
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        config[key] = value
        
    return config

def sync_remote_node(content):
    """リモートのエンドポイントへコンテンツを同期"""
    
    # 設定をパース
    config = parse_config_from_env()
    
    api_endpoint = config.get("API_ENDPOINT_V1")
    client_ref = config.get("CLIENT_ID_REF")
    secret_key = config.get("APP_SECRET_KEY")
    target_node = config.get("TARGET_NODE_ID")

    # デバッグ情報 (値は隠す)
    debug_info = {k: "Set" if v else "MISSING" for k, v in 
                 {"API": api_endpoint, "USER": client_ref, "KEY": secret_key, "ID": target_node}.items()}

    if not all([api_endpoint, client_ref, secret_key, target_node]):
        print("Skipping sync: Missing configuration in CORE_SYSTEM_CONFIG.")
        print(f"Debug Config: {debug_info}")
        return

    target_url = f"{api_endpoint.rstrip('/')}/wp-json/wp/v2/pages/{target_node}"
    
    credentials = f"{client_ref}:{secret_key}"
    token = base64.b64encode(credentials.encode()).decode("utf-8")
    
    headers = {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json'
    }
    
    payload = {'content': content}
    
    print(f"Syncing data to remote node: {target_node}...")
    try:
        response = requests.post(target_url, headers=headers, json=payload)
        if response.status_code == 200:
            print("Sync successful.")
        else:
            print(f"Sync failed. Status: {response.status_code}")
            print(response.text[:200])
    except Exception as e:
        print(f"Connection error: {e}")

def main():
    print("Starting US Sector Analysis...")
    
    all_rows = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_sector_data, code, name) for code, name in SECTOR_ETFS.items()]
        for future in futures:
            res = future.result()
            if res:
                all_rows.extend(res)
    
    if not all_rows:
        print("No data retrieved.")
        return

    print("Processing data...")
    latest, labels, datasets, top3 = process_data_for_chart(all_rows)
    
    print("Generating view...")
    html_content = generate_html_content(latest, labels, datasets, top3)
    
    print("Initiating remote sync...")
    sync_remote_node(html_content)
    
    print("Done.")

if __name__ == "__main__":
    main()
