"""
전력 수요 예측 시스템
XGBoost + LightGBM 앙상블
2021~2026 데이터 + 지역별 예측 지원
"""

import os
import glob
import requests
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import nnls
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════

BASE_PATH       = r'C:\Users\rokaf'
HOLIDAY_API_KEY = 'ebcb301e908f641ad9bd126aa4ba50acf9c2e5738f40c94ef9d6e608cba359d9'

# 5개 주요 도시 지점번호
STATIONS = {
    '서울': 108,
    '부산': 159,
    '대구': 143,
    '광주': 156,
    '대전': 133,
}

# 전력 데이터 파일 (2021~2025 + 2026)
POWER_FILES_OLD = sorted(glob.glob(os.path.join(BASE_PATH, 'sukub (*).csv')))
POWER_FILES_NEW = sorted(glob.glob(os.path.join(BASE_PATH, 'sukub 2026 (*).csv')))
POWER_FILES_NEW += [f for f in [os.path.join(BASE_PATH, 'sukub2026(3).csv')] if os.path.exists(f)]
POWER_FILES     = POWER_FILES_OLD + POWER_FILES_NEW

# 기상 CSV 파일 (2021~2025 + 2026)
WEATHER_FILES_OLD = [
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428124033.csv'),  # 2021
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428124624.csv'),  # 2022
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428125146.csv'),  # 2023
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428125905.csv'),  # 2024
    os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260428131304.csv'),  # 2025
]
WEATHER_FILE_2026 = os.path.join(BASE_PATH, 'OBS_ASOS_TIM_20260504170216.csv')
WEATHER_FILES_ALL = WEATHER_FILES_OLD + [WEATHER_FILE_2026]

FEAT_COLS = [
    'hour', 'minute', 'dayofweek', 'month', 'day', 'quarter',
    'is_weekend', 'is_holiday', 'is_off',
    'hour_sin', 'hour_cos', 'min_sin', 'min_cos',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'lag_1', 'lag_12', 'lag_288', 'lag_576', 'lag_2016',
    'roll_12_mean', 'roll_12_std', 'roll_12_max', 'roll_12_min',
    'roll_288_mean', 'roll_288_std', 'roll_288_max', 'roll_288_min',
    'temp', 'humi', 'wind', 'rain', 'temp_sq', 'heat_index', 'feels_like'
]


# ══════════════════════════════════════════════════════════════
#  1. 전력 데이터 로드
# ══════════════════════════════════════════════════════════════

def load_power_data():
    print("[1/5] 전력 데이터 로드 중...")
    dfs = []
    for f in POWER_FILES:
        try:
            df = pd.read_csv(f, encoding='cp949', low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"      {os.path.basename(f)} 로드 실패: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={'기준일시': 'ds', '현재수요(MW)': 'y'})
    df['ds'] = pd.to_datetime(df['ds'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
    df['y']  = pd.to_numeric(df['y'], errors='coerce')
    df = df[['ds', 'y']].dropna().sort_values('ds').drop_duplicates('ds').reset_index(drop=True)

    print(f"      기간: {df['ds'].min()} ~ {df['ds'].max()}")
    print(f"      총 {len(df):,}건 로드 완료")
    return df


# ══════════════════════════════════════════════════════════════
#  2. 공휴일 API
# ══════════════════════════════════════════════════════════════

def fetch_holidays(years):
    print("[2/5] 공휴일 데이터 수집 중...")
    holidays = set()
    for year in years:
        for month in range(1, 13):
            url = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
            params = {
                'serviceKey': HOLIDAY_API_KEY,
                'solYear':    str(year),
                'solMonth':   f'{month:02d}',
                'numOfRows':  '50',
                '_type':      'json'
            }
            try:
                res = requests.get(url, params=params, timeout=10)
                data = res.json()
                items = data.get('response', {}).get('body', {}).get('items', {})
                if not items:
                    continue
                item_list = items.get('item', [])
                if isinstance(item_list, dict):
                    item_list = [item_list]
                for item in item_list:
                    date_str = str(item.get('locdate', ''))
                    if len(date_str) == 8:
                        holidays.add(pd.to_datetime(date_str, format='%Y%m%d').date())
            except Exception:
                continue
    print(f"      공휴일 {len(holidays)}일 수집 완료")
    return holidays


# ══════════════════════════════════════════════════════════════
#  3. 기상 CSV 로드 (지역별)
# ══════════════════════════════════════════════════════════════

def load_weather_by_station(station_id=None):
    """station_id=None이면 전국 평균"""
    print(f"[3/5] 기상 데이터 로드 중 (지점: {station_id})...")
    dfs = []
    for f in WEATHER_FILES_ALL:
        try:
            df = pd.read_csv(f, encoding='cp949', low_memory=False)
            if station_id is not None:
                df = df[df['지점'] == station_id]
            dfs.append(df)
        except Exception as e:
            print(f"      {os.path.basename(f)} 로드 실패: {e}")

    if not dfs:
        return None

    weather = pd.concat(dfs, ignore_index=True)
    weather = weather.rename(columns={
        '일시': 'ds', '기온(°C)': 'temp',
        '습도(%)': 'humi', '풍속(m/s)': 'wind', '강수량(mm)': 'rain'
    })
    weather['ds']   = pd.to_datetime(weather['ds'], errors='coerce')
    weather['temp'] = pd.to_numeric(weather['temp'], errors='coerce')
    weather['humi'] = pd.to_numeric(weather['humi'], errors='coerce')
    weather['wind'] = pd.to_numeric(weather['wind'], errors='coerce')
    weather['rain'] = pd.to_numeric(weather['rain'], errors='coerce').fillna(0)

    if station_id is None:
        weather = weather.groupby('ds')[['temp', 'humi', 'wind', 'rain']].mean().reset_index()
    else:
        weather = weather[['ds', 'temp', 'humi', 'wind', 'rain']].dropna(subset=['ds'])

    weather = weather.sort_values('ds').drop_duplicates('ds').reset_index(drop=True)
    weather['temp'] = weather['temp'].fillna(weather['temp'].median())
    weather['humi'] = weather['humi'].fillna(weather['humi'].median())
    weather['wind'] = weather['wind'].fillna(weather['wind'].median())
    weather['rain'] = weather['rain'].fillna(0)

    print(f"      기상 데이터 {len(weather):,}건 준비 완료")
    return weather


# ══════════════════════════════════════════════════════════════
#  4. 피처 엔지니어링
# ══════════════════════════════════════════════════════════════

def make_features(df, holidays, weather_df=None):
    d = df.copy()
    d['hour']       = d['ds'].dt.hour
    d['minute']     = d['ds'].dt.minute
    d['dayofweek']  = d['ds'].dt.dayofweek
    d['month']      = d['ds'].dt.month
    d['day']        = d['ds'].dt.day
    d['quarter']    = d['ds'].dt.quarter
    d['is_weekend'] = (d['ds'].dt.dayofweek >= 5).astype(int)
    d['is_holiday'] = d['ds'].dt.date.apply(lambda x: 1 if x in holidays else 0)
    d['is_off']     = ((d['is_weekend'] == 1) | (d['is_holiday'] == 1)).astype(int)
    d['hour_sin']   = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos']   = np.cos(2 * np.pi * d['hour'] / 24)
    d['min_sin']    = np.sin(2 * np.pi * d['minute'] / 60)
    d['min_cos']    = np.cos(2 * np.pi * d['minute'] / 60)
    d['dow_sin']    = np.sin(2 * np.pi * d['dayofweek'] / 7)
    d['dow_cos']    = np.cos(2 * np.pi * d['dayofweek'] / 7)
    d['month_sin']  = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos']  = np.cos(2 * np.pi * d['month'] / 12)

    for lag in [1, 12, 288, 576, 2016]:
        d[f'lag_{lag}'] = d['y'].shift(lag)
    for w in [12, 288]:
        d[f'roll_{w}_mean'] = d['y'].shift(1).rolling(w).mean()
        d[f'roll_{w}_std']  = d['y'].shift(1).rolling(w).std()
        d[f'roll_{w}_max']  = d['y'].shift(1).rolling(w).max()
        d[f'roll_{w}_min']  = d['y'].shift(1).rolling(w).min()

    if weather_df is not None:
        d['ds_hour'] = d['ds'].dt.floor('h')
        w = weather_df.copy()
        w['ds'] = pd.to_datetime(w['ds']).dt.floor('h')
        d = d.merge(w, left_on='ds_hour', right_on='ds', how='left', suffixes=('', '_w'))
        d = d.drop(columns=['ds_hour', 'ds_w'], errors='ignore')
        for col in ['temp', 'humi', 'wind', 'rain']:
            if col in d.columns:
                d[col] = d[col].fillna(d[col].median())
    else:
        d['temp'] = 15.0
        d['humi'] = 60.0
        d['wind'] = 2.0
        d['rain'] = 0.0

    d['temp_sq']    = d['temp'] ** 2
    d['heat_index'] = d['temp'] * d['humi'] / 100
    d['feels_like'] = d['temp'] - 0.4 * (d['temp'] - 10) * (1 - d['humi'] / 100)
    return d


# ══════════════════════════════════════════════════════════════
#  5. XGB + LGB 앙상블
# ══════════════════════════════════════════════════════════════

def train_xgb(X_tr, y_tr, X_val, y_val):
    m = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42, verbosity=0,
        early_stopping_rounds=30,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def train_lgb(X_tr, y_tr, X_val, y_val):
    from lightgbm import early_stopping, log_evaluation
    m = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)])
    return m


def train_ensemble(df_feat):
    split     = int(len(df_feat) * 0.8)
    train_df  = df_feat.iloc[:split].copy()
    test_df   = df_feat.iloc[split:].copy()
    val_split = int(len(train_df) * 0.85)
    tr  = train_df.iloc[:val_split]
    val = train_df.iloc[val_split:]

    X_tr,  y_tr  = tr[FEAT_COLS],  tr['y']
    X_val, y_val = val[FEAT_COLS], val['y']
    X_test = test_df[FEAT_COLS]
    y_test = test_df['y'].values

    print("      XGBoost 학습 중...", end=' ', flush=True)
    xgb_model = train_xgb(X_tr, y_tr, X_val, y_val)
    print("완료!")

    print("      LightGBM 학습 중...", end=' ', flush=True)
    lgb_model = train_lgb(X_tr, y_tr, X_val, y_val)
    print("완료!")

    xgb_val_pred = xgb_model.predict(X_val)
    lgb_val_pred = lgb_model.predict(X_val)
    weights, _   = nnls(np.column_stack([xgb_val_pred, lgb_val_pred]), y_val.values)
    weights      = weights / weights.sum()
    print(f"      앙상블 가중치 — XGB: {weights[0]:.3f} | LGB: {weights[1]:.3f}")

    xgb_pred      = xgb_model.predict(X_test)
    lgb_pred      = lgb_model.predict(X_test)
    ensemble_pred = weights[0] * xgb_pred + weights[1] * lgb_pred

    return xgb_model, lgb_model, weights, test_df, xgb_pred, lgb_pred, ensemble_pred, y_test


def evaluate(y_true, y_pred, name):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {'모델': name, 'MAPE(%)': round(mape, 2), 'RMSE(MW)': round(rmse, 1), 'MAE(MW)': round(mae, 1)}


# ══════════════════════════════════════════════════════════════
#  6. 지역별 미래 30일 예측
# ══════════════════════════════════════════════════════════════

def forecast_by_station(df_feat, xgb_model, lgb_model, weights, holidays, station_name, station_id):
    """특정 지역 기상 데이터로 미래 30일 예측"""
    last_ds   = df_feat['ds'].max()
    future_ds = pd.date_range(last_ds + pd.Timedelta(minutes=5), periods=288*30, freq='5min')

    # 해당 지역 기상 데이터 로드
    weather_df = load_weather_by_station(station_id)

    # 미래 기상 → 과거 같은 달/시간 평균으로 대체
    if weather_df is not None:
        future_weather = []
        for dt in future_ds:
            same = weather_df[
                (weather_df['ds'].dt.month == dt.month) &
                (weather_df['ds'].dt.hour  == dt.hour)
            ]
            future_weather.append({
                'ds':   dt,
                'temp': same['temp'].mean() if len(same) > 0 else 15.0,
                'humi': same['humi'].mean() if len(same) > 0 else 60.0,
                'wind': same['wind'].mean() if len(same) > 0 else 2.0,
                'rain': 0.0,
            })
        future_weather_df = pd.DataFrame(future_weather)
    else:
        future_weather_df = None

    # 순차 예측 (5분 단위, 래그 피처 실시간 계산)
    y_series = df_feat['y'].tolist()

    # 기상 데이터 딕셔너리로 변환 (빠른 조회)
    weather_dict = {}
    if future_weather_df is not None:
        for _, row in future_weather_df.iterrows():
            weather_dict[row['ds']] = row

    results = []
    for dt in future_ds:
        # 기상값 조회
        w = weather_dict.get(dt, None)
        temp = w['temp'] if w is not None else 15.0
        humi = w['humi'] if w is not None else 60.0
        wind = w['wind'] if w is not None else 2.0

        is_holiday = 1 if dt.date() in holidays else 0
        is_weekend = int(dt.dayofweek >= 5)

        row = {
            'hour': dt.hour, 'minute': dt.minute,
            'dayofweek': dt.dayofweek, 'month': dt.month,
            'day': dt.day, 'quarter': dt.quarter,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'is_off': int(is_weekend == 1 or is_holiday == 1),
            'hour_sin': np.sin(2*np.pi*dt.hour/24),
            'hour_cos': np.cos(2*np.pi*dt.hour/24),
            'min_sin':  np.sin(2*np.pi*dt.minute/60),
            'min_cos':  np.cos(2*np.pi*dt.minute/60),
            'dow_sin':  np.sin(2*np.pi*dt.dayofweek/7),
            'dow_cos':  np.cos(2*np.pi*dt.dayofweek/7),
            'month_sin': np.sin(2*np.pi*dt.month/12),
            'month_cos': np.cos(2*np.pi*dt.month/12),
            'lag_1':    y_series[-1],
            'lag_12':   y_series[-12],
            'lag_288':  y_series[-288]  if len(y_series) >= 288  else y_series[0],
            'lag_576':  y_series[-576]  if len(y_series) >= 576  else y_series[0],
            'lag_2016': y_series[-2016] if len(y_series) >= 2016 else y_series[0],
            'roll_12_mean': np.mean(y_series[-12:]),
            'roll_12_std':  np.std(y_series[-12:]),
            'roll_12_max':  np.max(y_series[-12:]),
            'roll_12_min':  np.min(y_series[-12:]),
            'roll_288_mean': np.mean(y_series[-288:]),
            'roll_288_std':  np.std(y_series[-288:]),
            'roll_288_max':  np.max(y_series[-288:]),
            'roll_288_min':  np.min(y_series[-288:]),
            'temp': temp, 'humi': humi, 'wind': wind, 'rain': 0.0,
            'temp_sq':    temp ** 2,
            'heat_index': temp * humi / 100,
            'feels_like': temp - 0.4 * (temp - 10) * (1 - humi / 100),
        }
        X = pd.DataFrame([row])[FEAT_COLS]
        xgb_p = xgb_model.predict(X)[0]
        lgb_p = lgb_model.predict(X)[0]
        pred  = weights[0] * xgb_p + weights[1] * lgb_p
        y_series.append(pred)
        results.append({
            '날짜시간':       dt,
            '앙상블예측(MW)': round(pred, 1),
            '발전량기준(MW)': round(pred * 1.149, 1),
            'XGB예측(MW)':   round(xgb_p, 1),
            'LGB예측(MW)':   round(lgb_p, 1),
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  전력 수요 예측 시스템 (XGB + LGB 앙상블) - 2021~2026")
    print("=" * 60)

    # 1. 전력 데이터
    df_raw = load_power_data()

    # 2. 공휴일
    years    = list(range(df_raw['ds'].dt.year.min(), df_raw['ds'].dt.year.max() + 2))
    holidays = fetch_holidays(years)

    # 3. 기상 (전국 평균으로 학습)
    weather_df = load_weather_by_station(None)

    # 4. 피처 엔지니어링
    print("[4/5] 피처 엔지니어링 중...")
    df_feat = make_features(df_raw, holidays, weather_df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"      피처 생성 완료: {len(df_feat):,}건, {len(df_feat.columns)}개 컬럼")

    # 5. 모델 학습 & 평가
    print("[5/5] 모델 학습 & 평가 중...")
    xgb_model, lgb_model, weights, test_df, xgb_pred, lgb_pred, ensemble_pred, y_test = train_ensemble(df_feat)

    results = [
        evaluate(y_test, xgb_pred,     'XGBoost (단독)'),
        evaluate(y_test, lgb_pred,      'LightGBM (단독)'),
        evaluate(y_test, ensemble_pred, '★ XGB+LGB 앙상블'),
    ]

    print()
    print(f"{'모델':<22} {'MAPE(%)':>10} {'RMSE(MW)':>10} {'MAE(MW)':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['모델']:<22} {r['MAPE(%)']:>10} {r['RMSE(MW)']:>10} {r['MAE(MW)']:>10}")
    print("-" * 55)

    ensemble_mape = results[-1]['MAPE(%)']
    status = "✅ 달성" if ensemble_mape <= 2.6 else "❌ 미달"
    print(f"\n  사양 검증 (MAPE ≤ 2.6%): {ensemble_mape}% → {status}")

    # 모델 저장
    print("\n  모델 저장 중...", end=' ', flush=True)
    joblib.dump(xgb_model,  os.path.join(BASE_PATH, 'xgb_model.pkl'))
    joblib.dump(lgb_model,  os.path.join(BASE_PATH, 'lgb_model.pkl'))
    joblib.dump(weights,    os.path.join(BASE_PATH, 'ensemble_weights.pkl'))
    joblib.dump(holidays,   os.path.join(BASE_PATH, 'holidays.pkl'))
    print("완료!")

    # 지역별 미래 30일 예측 저장
    print("\n  지역별 미래 30일 예측 생성 중...")
    for station_name, station_id in STATIONS.items():
        print(f"    {station_name}...", end=' ', flush=True)
        forecast_df = forecast_by_station(df_feat, xgb_model, lgb_model, weights, holidays, station_name, station_id)
        save_path = os.path.join(BASE_PATH, f'forecast_30d_{station_name}.csv')
        forecast_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"완료! ({len(forecast_df)}건)")

    print("\n" + "=" * 60)
    print("  전체 완료!")
    print("=" * 60)

    return xgb_model, lgb_model, weights, df_feat, holidays


if __name__ == '__main__':
    xgb_model, lgb_model, weights, df_feat, holidays = main()
