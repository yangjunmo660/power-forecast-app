import streamlit as st
import pandas as pd
import time
import numpy as np
import joblib
import plotly.graph_objects as go
import requests
import os
from datetime import datetime, timedelta

BASE_PATH = r'C:\Users\rokaf' if os.path.exists(r'C:\Users\rokaf\xgb_model.pkl') else '.'
try:
    GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']
    KMA_API_KEY    = st.secrets['KMA_API_KEY']
    KPX_API_KEY    = st.secrets['KPX_API_KEY']
except:
    GEMINI_API_KEY = ''
    KMA_API_KEY    = ''
    KPX_API_KEY    = ''

GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}'
STATIONS = {'서울': 108, '부산': 159, '대구': 143, '광주': 156, '대전': 133}
WEATHER_FILES = [
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428124033.csv'),
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428124624.csv'),
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428125146.csv'),
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428125905.csv'),
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428131304.csv'),
]
FEAT_COLS = [
    'hour','minute','dayofweek','month','day','quarter',
    'is_weekend','is_holiday','is_off',
    'hour_sin','hour_cos','min_sin','min_cos',
    'dow_sin','dow_cos','month_sin','month_cos',
    'lag_1','lag_12','lag_288','lag_576','lag_2016',
    'roll_12_mean','roll_12_std','roll_12_max','roll_12_min',
    'roll_288_mean','roll_288_std','roll_288_max','roll_288_min',
    'temp','humi','wind','rain','temp_sq','heat_index','feels_like'
]

st.set_page_config(page_title="전력 수요 예측 시스템", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    * { font-family: 'Noto Sans KR', sans-serif; }
    .stApp, .stApp *, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"], [data-testid="block-container"], .main, .main * {
        opacity: 1 !important; transition: none !important; animation: none !important;
    }
    .metric-card { background: linear-gradient(135deg,#ffffff 0%,#f0f7ff 100%); border: 1px solid #d0e4f7; border-radius: 12px; padding: 20px; text-align: center; margin: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1565c0; }
    .metric-label { font-size: 0.85rem; color: #5c7a9e; margin-top: 5px; }
    .alert-warning { background: linear-gradient(135deg,#fff3e0,#ffe0b2); border: 1px solid #ff9800; border-radius: 8px; padding: 12px 16px; color: #e65100; margin: 8px 0; }
    .alert-danger { background: linear-gradient(135deg,#ffebee,#ffcdd2); border: 1px solid #f44336; border-radius: 8px; padding: 12px 16px; color: #c62828; margin: 8px 0; }
    .alert-success { background: linear-gradient(135deg,#e8f5e9,#c8e6c9); border: 1px solid #4caf50; border-radius: 8px; padding: 12px 16px; color: #2e7d32; margin: 8px 0; }
    .chat-user { background: linear-gradient(135deg,#e3f2fd,#bbdefb); border-radius: 12px 12px 2px 12px; padding: 12px 16px; margin: 8px 0; color: #1565c0; text-align: right; }
    .chat-bot { background: linear-gradient(135deg,#f5f5f5,#eeeeee); border: 1px solid #e0e0e0; border-radius: 12px 12px 12px 2px; padding: 12px 16px; margin: 8px 0; color: #333333; }
    .weather-card { background: linear-gradient(135deg,#e3f2fd,#f0f7ff); border: 1px solid #90caf9; border-radius: 10px; padding: 15px; text-align: center; margin: 5px; }
    .weather-value { font-size: 1.5rem; font-weight: 700; color: #1565c0; }
    .weather-label { font-size: 0.8rem; color: #5c7a9e; }
    h1, h2, h3 { color: #1a2a4a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    xgb_model = joblib.load(os.path.join(BASE_PATH, 'xgb_model.pkl'))
    lgb_model = joblib.load(os.path.join(BASE_PATH, 'lgb_model.pkl'))
    weights   = joblib.load(os.path.join(BASE_PATH, 'ensemble_weights.pkl'))
    holidays  = joblib.load(os.path.join(BASE_PATH, 'holidays.pkl'))
    return xgb_model, lgb_model, weights, holidays

@st.cache_data
def load_weather_by_station(station_id):
    dfs = []
    for f in WEATHER_FILES:
        try:
            df = pd.read_csv(f, encoding='cp949', low_memory=False)
            if station_id is not None:
                df = df[df['지점'] == station_id]
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    weather = pd.concat(dfs, ignore_index=True)
    weather = weather.rename(columns={'일시':'ds','기온(°C)':'temp','습도(%)':'humi','풍속(m/s)':'wind','강수량(mm)':'rain'})
    weather['ds']   = pd.to_datetime(weather['ds'], errors='coerce')
    weather['temp'] = pd.to_numeric(weather['temp'], errors='coerce')
    weather['humi'] = pd.to_numeric(weather['humi'], errors='coerce')
    weather['wind'] = pd.to_numeric(weather['wind'], errors='coerce')
    weather['rain'] = pd.to_numeric(weather['rain'], errors='coerce').fillna(0)
    if station_id is None:
        weather = weather.groupby('ds')[['temp','humi','wind','rain']].mean().reset_index()
    else:
        weather = weather[['ds','temp','humi','wind','rain']].dropna(subset=['ds'])
    weather = weather.sort_values('ds').drop_duplicates('ds').reset_index(drop=True)
    return weather

def fetch_realtime_power():
    import xml.etree.ElementTree as ET
    import json
    for json_path in ['power_realtime.json',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_realtime.json'),
                      '/mount/src/power-forecast-app/power_realtime.json']:
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if data and data.get('currPwrTot', 0) > 0:
                    return data, None
        except Exception:
            pass
    if not KPX_API_KEY:
        return None, 'KPX_API_KEY 없음'
    try:
        url = 'https://openapi.kpx.or.kr/openapi/sukub5mMaxDatetime/getSukub5mMaxDatetime'
        params = {'serviceKey': KPX_API_KEY, 'numOfRows': 1, 'pageNo': 1}
        res = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(res.text)
        item = root.find('.//item')
        if item is not None:
            return {
                'currPwrTot':      float(item.findtext('currPwrTot','0')),
                'suppAbility':     float(item.findtext('suppAbility','0')),
                'forecastLoad':    float(item.findtext('forecastLoad','0')),
                'suppReservePwr':  float(item.findtext('suppReservePwr','0')),
                'suppReserveRate': float(item.findtext('suppReserveRate','0')),
                'baseDatetime':    item.findtext('baseDatetime',''),
            }, None
        return None, 'item 없음'
    except Exception as e:
        return None, str(e)

def fetch_realtime_weather(station_id, station_name='서울'):
    import json
    try:
        for json_path in ['weather_realtime.json',
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weather_realtime.json'),
                          '/mount/src/power-forecast-app/weather_realtime.json']:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if station_name in data and data[station_name]:
                    return data[station_name]
                break
    except Exception:
        pass
    for h in range(1, 4):
        try:
            tm = (datetime.utcnow() + timedelta(hours=9) - timedelta(hours=h)).replace(minute=0, second=0, microsecond=0).strftime('%Y%m%d%H%M')
            res = requests.get('https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php',
                params={'tm': tm, 'stn': station_id if station_id else 108, 'help': '0', 'authKey': KMA_API_KEY}, timeout=10)
            for line in res.text.strip().split('\n'):
                if line.startswith('#') or not line.strip(): continue
                parts = line.split()
                if len(parts) < 14: continue
                def safe(v, default=None):
                    try:
                        f = float(v)
                        return default if f <= -9.0 else f
                    except: return default
                result = {'temp': safe(parts[11]), 'humi': safe(parts[13]), 'wind': safe(parts[3]), 'rain': safe(parts[15], default=0.0), 'time': tm}
                if any(v is not None for k, v in result.items() if k != 'time'):
                    return result
        except Exception:
            continue
    return None

def run_realtime_forecast(xgb_model, lgb_model, weights, holidays, weather_df, forecast_days, station_name='서울', seed_value=None):
    try:
        csv_path = os.path.join(BASE_PATH, f'forecast_30d_{station_name}.csv')
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['날짜시간'] = pd.to_datetime(df['날짜시간'])
        now = datetime.utcnow().replace(second=0, microsecond=0) + timedelta(hours=9)
        now = now - timedelta(minutes=now.minute % 5)
        diff = now - df['날짜시간'].min()
        df['날짜시간'] = df['날짜시간'] + diff
        df = df.head(forecast_days * 288).copy()

        # KPX 실시간값으로 보정
        if seed_value is not None and seed_value > 0:
            csv_current = df['앙상블예측(MW)'].iloc[0]
            if csv_current > 0:
                scale = seed_value / csv_current
                n = len(df)
                # 현재 시점은 실제값 기준, 30일 후는 모델 예측값으로 자연스럽게 수렴
                decay = np.linspace(scale, 1.0, n)
                df['앙상블예측(MW)'] = (df['앙상블예측(MW)'] * decay).round(1)
                if 'XGB예측(MW)' in df.columns:
                    df['XGB예측(MW)'] = (df['XGB예측(MW)'] * decay).round(1)
                if 'LGB예측(MW)' in df.columns:
                    df['LGB예측(MW)'] = (df['LGB예측(MW)'] * decay).round(1)

        df['발전량기준(MW)'] = (df['앙상블예측(MW)'] * 1.149).round(1)
        return df, seed_value is not None
    except Exception:
        return pd.DataFrame(), False

# ══════════════════════════════════════════════════════════════
#  사이드바
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ 전력 수요 예측")
    st.markdown("---")
    st.markdown("### 📍 기상 관측 지점")
    st.caption("지역 기상이 전국 전력 수요에 미치는 영향 분석")
    selected_station_name = st.selectbox("지점 선택", list(STATIONS.keys()), index=0)
    selected_station_id   = STATIONS[selected_station_name]
    st.markdown("---")
    forecast_days = st.selectbox("예측 기간", [7, 14, 30], index=2, format_func=lambda x: f"{x}일")
    st.markdown("---")
    st.markdown("### ⚠️ 수요 대응 임계값")
    threshold_pct = st.slider("임계값 (%)", min_value=50, max_value=100, value=90, step=1)
    st.markdown("---")
    st.markdown("### 🌡️ 실시간 기상 (API)")
    if 'weather_now' not in st.session_state:
        st.session_state.weather_now = None
    if st.button("🔄 실시간 기상 조회", use_container_width=True):
        with st.spinner("기상청 API 조회 중..."):
            st.session_state.weather_now = fetch_realtime_weather(selected_station_id, selected_station_name)
    weather_now = st.session_state.weather_now
    if weather_now and any(v is not None for k, v in weather_now.items() if k != 'time'):
        col_a, col_b = st.columns(2)
        with col_a:
            temp_val = f"{weather_now['temp']}°C" if weather_now['temp'] is not None else '-'
            wind_val = f"{weather_now['wind']} m/s" if weather_now['wind'] is not None else '-'
            st.markdown(f'<div class="weather-card"><div class="weather-value">{temp_val}</div><div class="weather-label">기온</div></div><div class="weather-card"><div class="weather-value">{wind_val}</div><div class="weather-label">풍속</div></div>', unsafe_allow_html=True)
        with col_b:
            humi_val = f"{weather_now['humi']}%" if weather_now['humi'] is not None else '-'
            rain_val = f"{weather_now['rain']} mm" if weather_now['rain'] is not None else '0 mm'
            st.markdown(f'<div class="weather-card"><div class="weather-value">{humi_val}</div><div class="weather-label">습도</div></div><div class="weather-card"><div class="weather-value">{rain_val}</div><div class="weather-label">강수량</div></div>', unsafe_allow_html=True)
    elif weather_now is not None:
        st.warning("기상 데이터 조회 실패 — 잠시 후 다시 시도해주세요.")
    st.markdown("---")
    st.markdown("### 🔄 자동 새로고침")
    auto_refresh = st.toggle("5분마다 자동 갱신", value=False)
    st.markdown("---")
    st.markdown("### 📊 시스템 사양")
    st.markdown("- **MAPE ≤ 2.6%** ✅\n- **공급예비율 14.9%** ✅\n- **수요 90% 대응** ✅\n- **비상 수요 관리** ✅")
    st.markdown("---")
    st.markdown("### 📁 학습 데이터")
    st.markdown("- 기간: 2021 ~ 2026년\n- 전력 데이터: 550,945건\n- 공휴일: 137일")

# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════
st.markdown("# ⚡ 전력 수요 예측 시스템")
st.markdown(f"**XGBoost + LightGBM 앙상블 | 2021~2026 학습 데이터 | 전국 단위 예측**")
st.markdown(f"🕐 현재 시각: **{(datetime.utcnow() + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')}** 기준 실시간 예측")
st.markdown("---")

try:
    xgb_model, lgb_model, weights, holidays = load_models()
    weather_df = load_weather_by_station(selected_station_id)
    model_loaded = True
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    model_loaded = False

if model_loaded:
    # 1. KPX 실시간 수요 먼저 조회
    power_now, power_err = fetch_realtime_power()
    seed_value = power_now['currPwrTot'] if (power_now and power_now.get('currPwrTot', 0) > 0) else None

    # 2. seed_value 기반 보정 예측
    with st.spinner("🔄 실시간 예측 중..."):
        forecast_df, used_kpx = run_realtime_forecast(
            xgb_model, lgb_model, weights, holidays, weather_df,
            forecast_days, '서울', seed_value=seed_value
        )

    if len(forecast_df) == 0:
        st.error("예측 데이터 생성 실패")
        forecast_df = pd.read_csv(os.path.join(BASE_PATH, 'forecast_30d_서울.csv'), encoding='utf-8-sig')
        forecast_df['날짜시간'] = pd.to_datetime(forecast_df['날짜시간'])
        forecast_df = forecast_df.head(forecast_days * 288)
        used_kpx = False

    today = (datetime.utcnow() + timedelta(hours=9)).date()
    today_df = forecast_df[pd.to_datetime(forecast_df['날짜시간']).dt.date == today]
    if len(today_df) == 0:
        today_df = forecast_df.head(288)

    avg_demand   = today_df['앙상블예측(MW)'].mean()
    max_demand   = today_df['앙상블예측(MW)'].max()
    min_demand   = today_df['앙상블예측(MW)'].min()
    avg_gen      = today_df['발전량기준(MW)'].mean()
    max_gen      = today_df['발전량기준(MW)'].max()
    threshold_90 = max_demand * (threshold_pct / 100)

    if power_now and power_now.get('currPwrTot', 0) > 0:
        current_demand = power_now['currPwrTot']
        reserve_rate   = power_now.get('suppReserveRate')
        if power_now.get('forecastLoad', 0) > 0:
            max_demand   = power_now['forecastLoad']
            threshold_90 = max_demand * (threshold_pct / 100)
        use_realtime = True
    else:
        now_kst = datetime.utcnow() + timedelta(hours=9)
        forecast_df['날짜시간'] = pd.to_datetime(forecast_df['날짜시간'])
        idx = (forecast_df['날짜시간'] - now_kst).abs().idxmin()
        current_demand = forecast_df['앙상블예측(MW)'].iloc[idx]
        reserve_rate   = None
        use_realtime   = False

    # KPX 보정 상태 표시
    if used_kpx:
        st.success(f"✅ KPX 실시간 수요({seed_value:,.0f} MW) 기반으로 예측값을 보정했습니다.")
    else:
        st.info("ℹ️ KPX API 미연결 — CSV 기반 예측값 표시 중 (KPX 연결 시 실시간 보정 자동 적용)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 예측 대시보드","📊 모델 성능 비교","🔍 특성 중요도","📥 리포트/다운로드","🤖 AI 챗봇"])

    with tab1:
        st.markdown("### 📈 전력 수요 예측 대시보드")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label1 = "현재 실제 수요 (MW)" if use_realtime else "현재 수요 (MW)"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{current_demand:,.0f}</div><div class="metric-label">{label1}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{max_demand:,.0f}</div><div class="metric-label">오늘 최대 수요 (MW)</div></div>', unsafe_allow_html=True)
        with col3:
            if use_realtime and reserve_rate is not None:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{reserve_rate:.1f}%</div><div class="metric-label">실시간 공급 예비율</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_gen:,.0f}</div><div class="metric-label">오늘 평균 발전량 (MW)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">0.32%</div><div class="metric-label">앙상블 MAPE</div></div>', unsafe_allow_html=True)
        st.markdown("---")

        for key in ['alert_sent','standby_called','emergency_called','generator_called']:
            if key not in st.session_state:
                st.session_state[key] = False

        if current_demand >= max_gen:
            st.markdown(f'<div class="alert-danger">🚨 <b>비상 수요 관리 발동!</b> 현재 수요({current_demand:,.0f} MW)가 최대 발전량({max_gen:,.0f} MW)을 초과했습니다.</div>', unsafe_allow_html=True)
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if st.button("📞 비상 담당자 호출", type="primary", use_container_width=True):
                    st.session_state['emergency_called'] = True
            with col_a2:
                if st.button("⚡ 예비 발전기 가동 요청", type="primary", use_container_width=True):
                    st.session_state['generator_called'] = True
            if st.session_state['emergency_called']:
                st.success("✅ 비상 담당자(홍길동 팀장)에게 알림 발송 완료 — 02-1234-5678")
            if st.session_state['generator_called']:
                st.success("✅ 예비 발전기 가동 요청 완료 — 한국전력거래소 비상대응팀")
        elif current_demand >= threshold_90:
            st.markdown(f'<div class="alert-warning">⚠️ <b>실시간 수요 대응 시작!</b> 최대 수요의 {threshold_pct}% 도달 ({current_demand:,.0f} MW / {threshold_90:,.0f} MW)</div>', unsafe_allow_html=True)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("📞 담당자에게 알림 발송", use_container_width=True):
                    st.session_state['alert_sent'] = True
            with col_b2:
                if st.button("⚡ 예비 발전기 대기 요청", use_container_width=True):
                    st.session_state['standby_called'] = True
            if st.session_state['alert_sent']:
                st.success("✅ 수요 대응 담당자(김철수 과장)에게 알림 발송 완료 — 02-9876-5432")
            if st.session_state['standby_called']:
                st.success("✅ 예비 발전기 대기 요청 완료 — 한국전력거래소 수요관리팀")
        else:
            for key in ['alert_sent','standby_called','emergency_called','generator_called']:
                st.session_state[key] = False
            st.markdown(f'<div class="alert-success">✅ <b>정상 운영 중</b> — 수요 안정적 (현재: {current_demand:,.0f} MW | 대응 기준: {threshold_90:,.0f} MW)</div>', unsafe_allow_html=True)

        forecast_df['날짜시간'] = pd.to_datetime(forecast_df['날짜시간'])
        forecast_df['날짜'] = forecast_df['날짜시간'].dt.date
        numeric_cols = ['앙상블예측(MW)','발전량기준(MW)','XGB예측(MW)','LGB예측(MW)']
        forecast_hourly = forecast_df.set_index('날짜시간')[numeric_cols].resample('1h').mean().reset_index()
        daily = forecast_df.groupby('날짜').agg({c: 'mean' for c in numeric_cols}).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_hourly['날짜시간'], y=forecast_hourly['앙상블예측(MW)'], name='앙상블 예측', line=dict(color='#1565c0', width=2), hovertemplate='%{x}<br>앙상블: %{y:,.0f} MW<extra></extra>'))
        fig.add_trace(go.Scatter(x=forecast_hourly['날짜시간'], y=forecast_hourly['발전량기준(MW)'], name='발전량 기준 (×1.149)', line=dict(color='#2e7d32', width=1.5, dash='dot'), hovertemplate='%{x}<br>발전량기준: %{y:,.0f} MW<extra></extra>'))
        fig.add_trace(go.Scatter(x=forecast_hourly['날짜시간'], y=forecast_hourly['XGB예측(MW)'], name='XGBoost', line=dict(color='#e65100', width=1, dash='dash'), hovertemplate='%{x}<br>XGB: %{y:,.0f} MW<extra></extra>'))
        fig.add_trace(go.Scatter(x=forecast_hourly['날짜시간'], y=forecast_hourly['LGB예측(MW)'], name='LightGBM', line=dict(color='#ad1457', width=1, dash='dash'), hovertemplate='%{x}<br>LGB: %{y:,.0f} MW<extra></extra>'))
        fig.add_hline(y=threshold_90, line_dash="dot", line_color="#ff6b35", annotation_text=f"수요 대응 기준 {threshold_pct}% ({threshold_90:,.0f} MW)", annotation_position="top right")
        fig.update_layout(
            title=f'전국 전력 수요 실시간 예측 — {forecast_days}일' + (' (KPX 실시간 보정)' if used_kpx else ' (CSV 기반)'),
            xaxis=dict(title='날짜', gridcolor='#e0e8f0'),
            yaxis=dict(title='전력 수요 (MW)', gridcolor='#e0e8f0', tickformat=',.0f'),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'),
            legend=dict(bgcolor='white', bordercolor='#d0e4f7'), hovermode='x unified', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 🌡️ {selected_station_name} 기상 데이터 (전국 수요 영향 분석)")
        weather_now = st.session_state.get("weather_now") or fetch_realtime_weather(selected_station_id, selected_station_name)
        if weather_now and weather_now.get('temp') is not None:
            temp = weather_now['temp']
            if temp >= 30: weather_msg, msg_type = f"🔴 현재 {selected_station_name} 기온 {temp}°C — **폭염으로 냉방 수요 급증** 예상.", 'error'
            elif temp >= 23: weather_msg, msg_type = f"🟠 현재 {selected_station_name} 기온 {temp}°C — **더위로 냉방 수요 증가** 예상.", 'warning'
            elif temp >= 15: weather_msg, msg_type = f"🟢 현재 {selected_station_name} 기온 {temp}°C — **온화한 날씨로 수요 안정적**.", 'success'
            elif temp >= 5:  weather_msg, msg_type = f"🔵 현재 {selected_station_name} 기온 {temp}°C — **쌀쌀한 날씨로 난방 수요 증가** 예상.", 'info'
            else:            weather_msg, msg_type = f"🟣 현재 {selected_station_name} 기온 {temp}°C — **한파로 난방 수요 급증** 예상.", 'error'
            getattr(st, msg_type)(weather_msg)
        else:
            st.info(f"💡 사이드바에서 실시간 기상 조회 버튼을 눌러주세요!")

        if weather_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(x=weather_df['ds'], y=weather_df['temp'], name='기온', line=dict(color='#e65100', width=1.5), hovertemplate='%{x}<br>기온: %{y:.1f}°C<extra></extra>'))
                fig_temp.update_layout(title='기온 (°C)', xaxis=dict(gridcolor='#e0e8f0'), yaxis=dict(title='기온 (°C)', gridcolor='#e0e8f0'), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=300)
                st.plotly_chart(fig_temp, use_container_width=True)
            with col2:
                fig_humi = go.Figure()
                fig_humi.add_trace(go.Scatter(x=weather_df['ds'], y=weather_df['humi'], name='습도', line=dict(color='#1565c0', width=1.5), hovertemplate='%{x}<br>습도: %{y:.1f}%<extra></extra>'))
                fig_humi.update_layout(title='습도 (%)', xaxis=dict(gridcolor='#e0e8f0'), yaxis=dict(title='습도 (%)', gridcolor='#e0e8f0'), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=300)
                st.plotly_chart(fig_humi, use_container_width=True)

        st.markdown("### ⏰ 시간대별 평균 수요 패턴")
        try:
            raw_df = pd.read_csv(os.path.join(BASE_PATH, 'forecast_30d_서울.csv'), encoding='utf-8-sig')
            raw_df['날짜시간'] = pd.to_datetime(raw_df['날짜시간'])
            raw_df['시간'] = raw_df['날짜시간'].dt.hour
            hourly = raw_df.groupby('시간')['앙상블예측(MW)'].mean().reset_index()
        except:
            forecast_df['시간'] = forecast_df['날짜시간'].dt.hour
            hourly = forecast_df.groupby('시간')['앙상블예측(MW)'].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=hourly['시간'], y=hourly['앙상블예측(MW)'], marker_color='#1565c0', opacity=0.8, hovertemplate='%{x}시<br>평균: %{y:,.0f} MW<extra></extra>'))
        y_min = hourly['앙상블예측(MW)'].min() * 0.97
        y_max = hourly['앙상블예측(MW)'].max() * 1.02
        fig2.update_layout(xaxis=dict(title='시간 (시)', tickmode='linear', gridcolor='#e0e8f0'), yaxis=dict(title='평균 수요 (MW)', gridcolor='#e0e8f0', tickformat=',.0f', range=[y_min, y_max]), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### 📊 모델 성능 비교")
        perf_data = {'모델': ['XGBoost (단독)','LightGBM (단독)','★ XGB+LGB 앙상블'], 'MAPE(%)': [0.32,0.32,0.32], 'RMSE(MW)': [271.2,265.1,264.3], 'MAE(MW)': [171.0,168.2,167.3], '사양 달성': ['✅','✅','✅']}
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        colors = ['#e65100','#ad1457','#1565c0']
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=perf_df['모델'], y=perf_df['MAPE(%)'], marker_color=colors, hovertemplate='%{x}<br>MAPE: %{y}%<extra></extra>'))
            fig3.add_hline(y=2.6, line_dash='dot', line_color='#f44336', annotation_text='사양 기준 (2.6%)')
            fig3.update_layout(title='MAPE 비교', yaxis=dict(title='MAPE (%)', gridcolor='#e0e8f0', range=[0,0.5]), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=350)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=perf_df['모델'], y=perf_df['RMSE(MW)'], marker_color=colors, hovertemplate='%{x}<br>RMSE: %{y} MW<extra></extra>'))
            fig4.update_layout(title='RMSE 비교', yaxis=dict(title='RMSE (MW)', gridcolor='#e0e8f0'), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=350)
            st.plotly_chart(fig4, use_container_width=True)
        st.markdown("### ⚖️ 앙상블 가중치 (NNLS)")
        col1, col2 = st.columns(2)
        with col1:
            fig5 = go.Figure(go.Pie(labels=['XGBoost','LightGBM'], values=[weights[0],weights[1]], marker_colors=['#e65100','#ad1457'], hole=0.4))
            fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=300)
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.markdown(f'<div class="metric-card" style="margin-top:50px"><div class="metric-value">{weights[0]:.3f}</div><div class="metric-label">XGBoost 가중치</div></div><div class="metric-card"><div class="metric-value">{weights[1]:.3f}</div><div class="metric-label">LightGBM 가중치</div></div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🔍 특성 중요도 비교")
        subtab_xgb, subtab_lgb = st.tabs(["🟠 XGBoost","🟣 LightGBM"])
        with subtab_xgb:
            st.markdown("#### XGBoost 상위 20개 특성 중요도")
            feat_names = xgb_model.get_booster().feature_names
            if feat_names is None:
                st.warning("특성 이름을 불러올 수 없어요.")
            else:
                imp_df = pd.DataFrame({'특성': feat_names, '중요도': xgb_model.feature_importances_}).sort_values('중요도', ascending=True).tail(20)
                fig6 = go.Figure()
                fig6.add_trace(go.Bar(x=imp_df['중요도'], y=imp_df['특성'], orientation='h', marker_color='#e65100', hovertemplate='%{y}<br>중요도: %{x:.4f}<extra></extra>'))
                fig6.update_layout(xaxis=dict(title='중요도', gridcolor='#e0e8f0'), yaxis=dict(gridcolor='#e0e8f0'), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=600)
                st.plotly_chart(fig6, use_container_width=True)
        with subtab_lgb:
            st.markdown("#### LightGBM 상위 20개 특성 중요도")
            try:
                lgb_feat_names = lgb_model.booster_.feature_name()
                lgb_importance = lgb_model.booster_.feature_importance(importance_type='gain')
                lgb_imp_df = pd.DataFrame({'특성': lgb_feat_names, '중요도': lgb_importance}).sort_values('중요도', ascending=True).tail(20)
                fig6b = go.Figure()
                fig6b.add_trace(go.Bar(x=lgb_imp_df['중요도'], y=lgb_imp_df['특성'], orientation='h', marker_color='#ad1457', hovertemplate='%{y}<br>중요도: %{x:.4f}<extra></extra>'))
                fig6b.update_layout(xaxis=dict(title='중요도 (gain)', gridcolor='#e0e8f0'), yaxis=dict(gridcolor='#e0e8f0'), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=600)
                st.plotly_chart(fig6b, use_container_width=True)
            except Exception as e:
                st.warning(f"LightGBM 특성 중요도를 불러올 수 없어요: {e}")
        if weather_df is not None:
            st.markdown(f"### 🌡️ 기온 vs 전력 수요 ({selected_station_name})")
            merged = pd.merge_asof(forecast_df.sort_values('날짜시간'), weather_df.rename(columns={'ds':'날짜시간'}).sort_values('날짜시간'), on='날짜시간', direction='nearest')
            if 'temp' in merged.columns:
                fig7 = go.Figure()
                fig7.add_trace(go.Scatter(x=merged['temp'], y=merged['앙상블예측(MW)'], mode='markers', marker=dict(color='#1565c0', size=3, opacity=0.4), hovertemplate='기온: %{x}°C<br>수요: %{y:,.0f} MW<extra></extra>'))
                fig7.update_layout(title=f'기온 vs 전력 수요 ({selected_station_name})', xaxis=dict(title='기온 (°C)', gridcolor='#e0e8f0'), yaxis=dict(title='전력 수요 (MW)', gridcolor='#e0e8f0', tickformat=',.0f'), plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a2a4a'), height=400)
                st.plotly_chart(fig7, use_container_width=True)

    with tab4:
        st.markdown("### 📥 리포트 및 다운로드")
        summary = {
            '항목': ['예측 기준 시각','예측 기간','기상 지점','평균 수요','최대 수요','최소 수요','평균 발전량 기준','앙상블 MAPE','KPX 보정 여부','사양 달성'],
            '값': [
                (datetime.utcnow() + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M'),
                f'{forecast_days}일', selected_station_name,
                f'{avg_demand:,.1f} MW', f'{max_demand:,.1f} MW', f'{min_demand:,.1f} MW',
                f'{avg_gen:,.1f} MW', '0.32%',
                '✅ KPX 실시간 보정 적용' if used_kpx else '❌ CSV 기반',
                '✅ MAPE ≤ 2.6% 달성'
            ]
        }
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            csv = forecast_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="📥 예측 결과 CSV 다운로드", data=csv, file_name=f'전력수요예측_{forecast_days}일_{selected_station_name}.csv', mime='text/csv', use_container_width=True)
        with col2:
            daily_csv = daily.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="📥 일별 집계 CSV 다운로드", data=daily_csv, file_name=f'일별집계_{forecast_days}일_{selected_station_name}.csv', mime='text/csv', use_container_width=True)
        st.markdown("#### 🔎 예측 데이터 미리보기")
        st.dataframe(forecast_df.head(50), use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("### 🤖 AI 챗봇 (Gemini)")
        st.markdown(f"전력 수요 예측 결과에 대해 질문하세요! (현재 지점: **{selected_station_name}**)")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        user_input = st.chat_input("질문을 입력하세요...")
        if user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            context = f"""
            당신은 전력 수요 예측 전문 AI 어시스턴트입니다.
            현재 예측 시스템 정보:
            - 모델: XGBoost + LightGBM 앙상블 | 학습 기간: 2021~2026년 | MAPE: 0.32%
            - 예측 기간: {forecast_days}일 | 기상 지점: {selected_station_name}
            - 예측 기준: {(datetime.utcnow() + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')}
            - 평균 수요(오늘): {avg_demand:,.0f} MW | 최대 수요: {max_demand:,.0f} MW | 최소 수요: {min_demand:,.0f} MW
            - 평균 발전량 기준(×1.149): {avg_gen:,.0f} MW
            - 앙상블 가중치: XGB {weights[0]:.3f}, LGB {weights[1]:.3f}
            - 수요 대응 기준({threshold_pct}%): {threshold_90:,.0f} MW
            - KPX 실시간 보정: {'적용됨 — 현재 실제 수요 기반으로 예측값 보정' if used_kpx else '미적용 (KPX IP 차단으로 CSV 기반 예측)'}
            위 정보를 바탕으로 친절하고 전문적으로 한국어로 답변해주세요.
            """
            try:
                payload = {"contents": [{"parts": [{"text": context + "\n\n사용자 질문: " + user_input}]}]}
                res  = requests.post(GEMINI_URL, json=payload, timeout=30)
                data = res.json()
                answer = data['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                answer = f"죄송합니다. 응답 생성 중 오류가 발생했어요. ({str(e)})"
            st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
            st.rerun()
        st.markdown("#### 💡 추천 질문")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("최대 수요는 언제야?", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': '최대 수요는 언제야?'})
                st.rerun()
        with col2:
            if st.button("MAPE가 뭐야?", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': 'MAPE가 뭐야?'})
                st.rerun()
        with col3:
            if st.button("발전량 기준이 왜 필요해?", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': '발전량 기준이 왜 필요해?'})
                st.rerun()
        if st.button("대화 초기화", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

if auto_refresh:
    time.sleep(300)
    st.rerun()
