import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import xlsxwriter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ==============================================
# Optional / heavy deps guarded behind try/except
# ==============================================
from prophet import Prophet

try:
    from neuralprophet import NeuralProphet
    _HAS_NEURALPROPHET = True
except ImportError:
    _HAS_NEURALPROPHET = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from xgboost import XGBRegressor
    import sklearn  # xgboost needs sklearn API
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# ---- Torch 2.6 safe-load fix for NeuralProphet checkpoints ----
# PyTorch 2.6 switched torch.load(weights_only=True) by default. NeuralProphet/Lightning
# may load checkpoints internally and blow up unless we allowlist its config classes or wrap calls
# with torch.serialization.safe_globals().
if _HAS_NEURALPROPHET:
    try:
        import torch
        import torch.serialization as _ts
        from torch.serialization import safe_globals as _safe_globals_ctx
        from neuralprophet.configure import (
            ConfigSeasonality, ConfigEvents, ConfigTrend, ConfigCountryHolidays,
            ConfigAR, ConfigLaggedRegressor, ConfigCustomSeasonality
        )
        _NP_SAFE_GLOBALS = [
            ConfigSeasonality, ConfigEvents, ConfigTrend, ConfigCountryHolidays,
            ConfigAR, ConfigLaggedRegressor, ConfigCustomSeasonality
        ]
        # make them globally safe for any internal torch.load
        _ts.add_safe_globals(_NP_SAFE_GLOBALS)
    except Exception:
        _safe_globals_ctx = None
        _NP_SAFE_GLOBALS = []
        pass

# ==============================================
# Streamlit Page Config
# ==============================================
st.set_page_config(page_title="NEST Multi‑Model Forecast App", layout="wide")
st.title("📈 SEB NEST Forecast App – Multi‑Model Compare")

# Global constants
WEEK_FREQ = 'W-SAT'
PI_Z_90 = 1.645  # for ~90% prediction interval

# ---- Session state keys for run / abort ----
if 'run_forecast' not in st.session_state:
    st.session_state['run_forecast'] = False
if 'abort_forecast' not in st.session_state:
    st.session_state['abort_forecast'] = False

# ==============================================
# Helpers
# ==============================================
@st.cache_data
def get_custom_holidays() -> pd.DataFrame:
    holiday_week_dates = [
        ('fiesta', pd.date_range('2023-04-20', '2023-04-30')),
        ('fiesta', pd.date_range('2024-04-18', '2024-04-28')),
        ('fiesta', pd.date_range('2025-04-21', '2025-05-04')),
        ('rodeo', pd.date_range('2023-02-09', '2023-02-26')),
        ('rodeo', pd.date_range('2024-02-08', '2024-02-25')),
        ('rodeo', pd.date_range('2025-02-12', '2025-03-01')),
        ('christmas', ['2023-12-25', '2024-12-25', '2025-12-25']),
        ('christmas_eve', ['2023-12-24', '2024-12-24', '2025-12-24']),
        ('memorial_day', ['2023-05-29', '2024-05-27', '2025-05-26']),
        ('cinco_de_mayo', ['2023-05-05', '2024-05-05', '2025-05-05']),
        ('fourth_of_july', ['2023-07-04', '2024-07-04', '2025-07-04']),
        ('labor_day', ['2023-09-04', '2024-09-02', '2025-09-01']),
        ('thanksgiving', ['2023-11-23', '2024-11-28', '2025-11-27']),
    ]
    records = []
    for name, dates in holiday_week_dates:
        for date in pd.to_datetime(dates):
            records.append({
                'holiday': str(name),
                'ds': pd.Timestamp(date).normalize(),
                'lower_window': -3,
                'upper_window': 3
            })
    return pd.DataFrame(records)

def _add_event_flags(df_dates: pd.DataFrame, holidays_df: pd.DataFrame, event_names=None) -> pd.DataFrame:
    """Create 0/1 columns for each event listed in holidays_df."""
    if event_names is None:
        event_names = holidays_df['holiday'].astype(str).unique().tolist()
    df_dates = df_dates.copy()
    for h in event_names:
        mask = df_dates['ds'].isin(holidays_df.loc[holidays_df['holiday'] == h, 'ds'])
        df_dates[h] = mask.astype(int)
    return df_dates

def safe_accuracy(y, yhat):
    if pd.isna(y) or y == 0 or pd.isna(yhat):
        return np.nan
    return round(1 - abs((y - yhat) / y), 3)

def next_saturday_from(date: pd.Timestamp) -> pd.Timestamp:
    # Monday=0 .. Saturday=5
    days_ahead = (5 - date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # ensure *next* Saturday
    return date + pd.Timedelta(days=days_ahead)

def build_future_saturdays(last_date: pd.Timestamp, horizon_weeks: int) -> pd.DatetimeIndex:
    first_sat = next_saturday_from(last_date)
    return pd.date_range(start=first_sat, periods=horizon_weeks, freq=WEEK_FREQ)

# ==============================================
# Models
# ==============================================
# -------------------- Prophet --------------------
def run_prophet(df_sku: pd.DataFrame, holidays_df: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        holidays=holidays_df,
        interval_width=0.90
    )
    m.fit(df_sku[['ds', 'y']])

    last_date = df_sku['ds'].max()
    future_sats = build_future_saturdays(last_date, horizon_weeks)

    future = pd.concat([
        df_sku[['ds']],
        pd.DataFrame({'ds': future_sats})
    ], ignore_index=True).drop_duplicates('ds')

    fcst = m.predict(future)
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    extra_cols = [c for c in ['trend', 'weekly', 'yearly', 'holidays'] if c in fcst.columns]
    cols.extend(extra_cols)

    out = fcst[cols].copy()
    out.rename(columns={
        'yhat': 'prophet_yhat',
        'yhat_lower': 'prophet_yhat_lower',
        'yhat_upper': 'prophet_yhat_upper'
    }, inplace=True)
    return out

# ---------------- NeuralProphet ------------------
def run_neuralprophet(df_sku: pd.DataFrame, holidays_df: pd.DataFrame, horizon_weeks: int, debug: bool = False) -> pd.DataFrame:
    """NeuralProphet wrapper with:
    - event one-hot flags
    - torch 2.6 safe load guards
    - ensures 'y' exists in full_future (NP can require it)
    """
    if not _HAS_NEURALPROPHET:
        raise ImportError("neuralprophet is not installed. pip install neuralprophet")

    event_names = holidays_df['holiday'].astype(str).unique().tolist()

    m = NeuralProphet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        quantiles=[0.05, 0.95]
    )

    for ev in event_names:
        m.add_events(ev, lower_window=-3, upper_window=3)

    # ---------- TRAIN -------------
    train_df = df_sku[['ds', 'y']].copy()
    train_df = _add_event_flags(train_df, holidays_df, event_names)

    missing_train = [ev for ev in event_names if ev not in train_df.columns]
    if missing_train:
        raise ValueError(f"Missing event cols in train_df: {missing_train}")

    def _fit_model():
        try:
            return m.fit(train_df, freq=WEEK_FREQ, trainer_config={'enable_checkpointing': False, 'logger': False})
        except TypeError:
            return m.fit(train_df, freq=WEEK_FREQ)

    if '_safe_globals_ctx' in globals() and _safe_globals_ctx:
        with _safe_globals_ctx(_NP_SAFE_GLOBALS):
            _fit_model()
    else:
        _fit_model()

    # ---------- FUTURE ------------
    last_date = df_sku['ds'].max()
    future_sats = build_future_saturdays(last_date, horizon_weeks)
    future = pd.DataFrame({'ds': future_sats})
    future = _add_event_flags(future, holidays_df, event_names)

    full_future = pd.concat([
        train_df[['ds'] + event_names],
        future[['ds'] + event_names]
    ], ignore_index=True).drop_duplicates('ds').sort_values('ds')

    # >>> 2-line fix <<<
    if 'y' not in full_future.columns:
        full_future['y'] = np.nan

    missing_future = [ev for ev in event_names if ev not in full_future.columns]
    if missing_future:
        raise ValueError(f"Missing event cols in full_future: {missing_future}")

    def _predict_model():
        return m.predict(full_future)

    if '_safe_globals_ctx' in globals() and _safe_globals_ctx:
        with _safe_globals_ctx(_NP_SAFE_GLOBALS):
            forecast = _predict_model()
    else:
        forecast = _predict_model()
        
    # catch-all for whatever names NP gave you
    q_cols = [c for c in forecast.columns if c.startswith('yhat1') and c not in ('yhat1',)]
    lower_col = next((c for c in q_cols if '0.05' in c or 'lower' in c), None)
    upper_col = next((c for c in q_cols if '0.95' in c or 'upper' in c), None)

    # fallback to NaN if NP didn’t produce them
    l_ser = forecast[lower_col] if lower_col in forecast else np.nan
    u_ser = forecast[upper_col] if upper_col in forecast else np.nan

    out = forecast[['ds', 'yhat1']].rename(columns={'yhat1': 'neural_yhat'})
    out['neural_yhat_lower'] = forecast.get('yhat1 5.0%', np.nan)
    out['neural_yhat_upper'] = forecast.get('yhat1 95.0%', np.nan)
    return out

# ----------------- SARIMAX ----------------------- -----------------------
def run_sarimax(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index('ds')['y'].asfreq(WEEK_FREQ)

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)

    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    steps = horizon_weeks
    fcst = res.get_forecast(steps=steps)
    pred_mean = fcst.predicted_mean
    conf = fcst.conf_int(alpha=0.10)
    # Try to rename whatever StatsModels gives us to standard names
    conf_cols = conf.columns.str.lower().tolist()
    lower_col = conf.columns[0] if 'lower' in conf_cols[0] else conf.columns[1]
    upper_col = conf.columns[1] if 'upper' in conf_cols[1] else conf.columns[0]
    conf = conf.rename(columns={lower_col: 'sarimax_yhat_lower', upper_col: 'sarimax_yhat_upper'})

    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=steps, freq=WEEK_FREQ)

    hist_df = pd.DataFrame({'ds': ts.index, 'sarimax_yhat': res.fittedvalues})
    fut_df = pd.DataFrame({'ds': future_index, 'sarimax_yhat': pred_mean.values})
    fut_df = pd.concat([fut_df.reset_index(drop=True), conf.reset_index(drop=True)], axis=1)

    out = pd.concat([hist_df, fut_df], ignore_index=True)
    out['sarimax_yhat_lower'] = out.get('sarimax_yhat_lower', np.nan)
    out['sarimax_yhat_upper'] = out.get('sarimax_yhat_upper', np.nan)
    return out

# --------------- XGBoost Regressor ---------------
def make_lag_features(y: pd.Series, lags=(1,2,3,4,5,6,7,12,26,52), roll_windows=(4,12)) -> pd.DataFrame:
    df_feat = pd.DataFrame({'y': y})
    for l in lags:
        df_feat[f'lag_{l}'] = y.shift(l)
    for w in roll_windows:
        df_feat[f'roll_mean_{w}'] = y.shift(1).rolling(window=w).mean()
    return df_feat

def add_date_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        'weekofyear': idx.isocalendar().week.astype(int),
        'month': idx.month,
        'quarter': idx.quarter,
        'year': idx.year
    }, index=idx)

def run_xgboost(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    if not _HAS_XGBOOST:
        raise ImportError("xgboost / scikit-learn not installed. pip install xgboost scikit-learn")

    ts = df_sku.set_index('ds')['y'].asfreq(WEEK_FREQ)
    features = make_lag_features(ts)
    date_feats = add_date_features(features.index)
    X = pd.concat([features.drop(columns=['y']), date_feats], axis=1)
    y = features['y']

    mask = X.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)

    preds_in = model.predict(X_train)
    resid_std = np.std(y_train - preds_in)

    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    full_y = ts.copy()
    rows = []

    for ds in future_index:
        feats = make_lag_features(full_y).iloc[-1:].drop(columns=['y'])
        feats = feats.assign(**add_date_features(pd.DatetimeIndex([ds])).iloc[0].to_dict())
        pred = model.predict(feats)[0]
        rows.append({
            'ds': ds,
            'xgb_yhat': pred,
            'xgb_yhat_lower': pred - PI_Z_90 * resid_std,
            'xgb_yhat_upper': pred + PI_Z_90 * resid_std
        })
        full_y.loc[ds] = pred

    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({'ds': X.index, 'xgb_yhat': hist_pred})
    hist_df['xgb_yhat_lower'] = np.nan
    hist_df['xgb_yhat_upper'] = np.nan

    fut_df = pd.DataFrame(rows)
    out = pd.concat([hist_df, fut_df], ignore_index=True)
    return out

# ==============================================
# UI Controls
# ==============================================
uploaded = st.file_uploader("📤 Upload your 'NEST Forecast Template.csv' file", type=["csv"])

colA, colB, colC, colD = st.columns(4)
with colA:
    horizon_weeks = st.slider("Forecast horizon (weeks)", 4, 52, 12, 1)
with colB:
    min_points = st.slider("Minimum rows per SKU", 20, 300, 50, 5)
with colC:
    run_prophet_flag = st.checkbox("Prophet", value=True)
    run_neural_flag = st.checkbox("NeuralProphet", value=False, disabled=not _HAS_NEURALPROPHET)
with colD:
    run_sarimax_flag = st.checkbox("SARIMAX", value=True)
    run_xgb_flag = st.checkbox("XGBoost", value=False, disabled=not _HAS_XGBOOST)

# Optional debug output
show_debug = st.checkbox("Show debug columns", value=False)

# Control buttons
start_clicked = st.button("🚀 Start forecasting", disabled=not bool(uploaded))
stop_placeholder = st.empty()

if start_clicked:
    st.session_state['run_forecast'] = True
    st.session_state['abort_forecast'] = False

# Show stop button only while running
if st.session_state['run_forecast'] and not st.session_state['abort_forecast']:
    if stop_placeholder.button("🛑 Stop forecasting", type="secondary"):
        st.session_state['abort_forecast'] = True

# ----------------------------------------------
# Main processing block (only when run flag true)
# ----------------------------------------------
if uploaded and st.session_state['run_forecast'] and not st.session_state['abort_forecast']:
    with st.spinner("Processing your forecast..."):
        df = pd.read_csv(uploaded)
        required_cols = {"sku_id", "ds", "y"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Missing columns: {required_cols - set(df.columns)}")
            st.session_state['run_forecast'] = False
            st.stop()

        df['ds'] = pd.to_datetime(df['ds'])
        holidays_df = get_custom_holidays()
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

        sku_list = df['sku_id'].unique()
        total = len(sku_list)

        progress = st.progress(0.0)
        status = st.empty()

        detailed_rows = []
        aborted = False

        for idx, (sku, df_sku) in enumerate(df.groupby('sku_id'), start=1):
            # Check abort flag each loop
            if st.session_state.get('abort_forecast'):
                aborted = True
                break

            progress.progress(idx / total)
            status.text(f"Processing {idx}/{total} → {sku}")

            if len(df_sku) < min_points:
                continue

            try:
                base = pd.DataFrame({'ds': df_sku['ds'].unique()}).sort_values('ds')
                base = base.merge(df_sku[['ds', 'y']], on='ds', how='left')

                model_futures = {}
                with ThreadPoolExecutor(max_workers=6) as executor:
                    if run_prophet_flag:
                        model_futures['prophet'] = executor.submit(run_prophet, df_sku, holidays_df, horizon_weeks)
                    if run_neural_flag:
                        model_futures['neural'] = executor.submit(run_neuralprophet, df_sku, holidays_df, horizon_weeks)
                    if run_sarimax_flag:
                        model_futures['sarimax'] = executor.submit(run_sarimax, df_sku, horizon_weeks)
                    if run_xgb_flag:
                        model_futures['xgb'] = executor.submit(run_xgboost, df_sku, horizon_weeks)

                # Merge model outputs
                for model_name, future in model_futures.items():
                    try:
                        result_df = future.result()
                        base = base.merge(result_df, on='ds', how='outer', validate='one_to_one')
                    except Exception as e:
                        st.warning(f"{model_name} failed for {sku}: {e}")

                # Compute model-specific accuracy
                for model_prefix in ["prophet", "neural", "sarimax", "xgb"]:
                    pred_col = f"{model_prefix}_yhat"
                    if pred_col in base.columns:
                        base[f"{model_prefix}_acc"] = base.apply(
                            lambda r: safe_accuracy(r.get('y'), r.get(pred_col)), axis=1
                        )

                base['sku_id'] = sku
                detailed_rows.append(base)

            except Exception as e:
                st.warning(f"⚠️ Error processing {sku}: {e}")
                continue

        # After loop
        if aborted:
            st.warning("🚫 Forecasting aborted by user.")
            st.session_state['run_forecast'] = False
            st.stop()

        if not detailed_rows:
            st.error("❌ No forecasts were generated. Check your file or adjust filters.")
            st.session_state['run_forecast'] = False
            st.stop()

        df_out = pd.concat(detailed_rows, ignore_index=True).sort_values(['sku_id', 'ds'])

        st.success("✅ Forecasting complete!")
        st.subheader("📊 Forecast Preview")
        preview_cols = [c for c in df_out.columns if (not c.endswith('_acc') or show_debug)]
        st.dataframe(df_out[preview_cols].head(50))

        output = BytesIO()
        with pd.ExcelWriter(
                output,
                engine='xlsxwriter',
                date_format='mm/dd/yyyy',
                datetime_format='mm/dd/yyyy'
            ) as writer:
            # Write the DataFrame
            df_out.to_excel(writer, index=False, sheet_name='forecasts')

            # Grab workbook & worksheet objects
            workbook  = writer.book
            worksheet = writer.sheets['forecasts']
            n_rows, n_cols = df_out.shape

            # --- Define formats ---
            date_fmt    = workbook.add_format({'num_format': 'mm/dd/yyyy'})
            int_fmt     = workbook.add_format({'num_format': '0'})       # whole numbers
            pct_fmt     = workbook.add_format({'num_format': '0.00%'})  # percentages, 2 decimals
            # center & middle-align + wrap + bold
            wrap_center = {'bold': True, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter'}

            # Header formats with background colors
            hdr1 = workbook.add_format({**wrap_center, 'bg_color': '#E6B8B7'})  # ds, y, sku_id
            hdr2 = workbook.add_format({**wrap_center, 'bg_color': '#CCC0DA'})  # prophet group
            hdr3 = workbook.add_format({**wrap_center, 'bg_color': '#8DB4E2'})  # neural group
            hdr4 = workbook.add_format({**wrap_center, 'bg_color': '#FCD5B4'})  # sarimax group
            hdr5 = workbook.add_format({**wrap_center, 'bg_color': '#C6EFCE'})  # xgb group
            hdr6 = workbook.add_format({**wrap_center, 'bg_color': '#FFFF00'})  # new Final cols

            # Freeze the header row
            worksheet.freeze_panes(1, 0)

            # Make every column ~75px wide (≈11 chars)
            for col in range(n_cols + 2):
                px = 75
                width = (px - 5) / 7
                worksheet.set_column(col, col, width)

            # Map column names → zero-based index
            col_idx = {name: i for i, name in enumerate(df_out.columns)}

            # Groups of columns for formatting
            group_map = {
                hdr1: ['ds','y','sku_id'],
                hdr2: ['prophet_yhat','prophet_yhat_lower','prophet_yhat_upper','trend','weekly','yearly','holidays','prophet_acc'],
                hdr3: ['neural_yhat','neural_yhat_lower','neural_yhat_upper','neural_acc'],
                hdr4: ['sarimax_yhat','sarimax_yhat_lower','sarimax_yhat_upper','sarimax_acc'],
                hdr5: ['xgb_yhat','xgb_yhat_lower','xgb_yhat_upper','xgb_acc'],
            }

            # Apply header formats & set data formats
            for fmt, names in group_map.items():
                for name in names:
                    if name not in col_idx:
                        continue
                    c = col_idx[name]
                    # write header
                    worksheet.write(0, c, name, fmt)
                    # set data-cell format
                    if name == 'ds':
                        worksheet.set_column(c, c, 11, date_fmt)
                    elif name.endswith('_acc'):
                        worksheet.set_column(c, c, 11, pct_fmt)
                    elif (
                        name in ('y','trend','weekly','yearly','holidays')
                        or name.endswith('_yhat')
                        or name.endswith('_yhat_lower')
                        or name.endswith('_yhat_upper')
                    ):
                        worksheet.set_column(c, c, 11, int_fmt)

            # --- Add Final Forecast & Final Accy % in cols X (23) & Y (24) ---
            col_final, col_accy = 23, 24
            worksheet.write(0, col_final, 'Final Forecast', hdr6)
            worksheet.write(0, col_accy,  'Final Accy %',   hdr6)

            for row in range(1, n_rows + 1):
                # LET/HSTACK formula for Final Forecast, now IFERROR-wrapped
                base_f1 = (
                    "LET(vals, HSTACK(C{r},J{r},M{r},P{r}),"
                    "avgVal, AVERAGE(vals),"
                    "diffs, ABS(vals-avgVal),"
                    "outlier, INDEX(vals, MATCH(MAX(diffs), diffs, 0)),"
                    "AVERAGE(FILTER(vals, vals<>outlier)))"
                ).format(r=row+1)
                f1 = f'=IFERROR({base_f1}, "")'
                worksheet.write_formula(row, col_final, f1, int_fmt)

                # Accuracy formula for Final Accy %, IFERROR-wrapped
                base_f2 = (
                    "IF(ABS(X{r}-$B{r})/$B{r}>1,"
                    "ABS(X{r}-$B{r})/$B{r}-1,"
                    "1-ABS(X{r}-$B{r})/$B{r})"
                ).format(r=row+1)
                f2 = f'=IFERROR({base_f2}, "")'
                worksheet.write_formula(row, col_accy, f2, pct_fmt)

        # rewind buffer for download
        output.seek(0)

        st.download_button(
            label="📥 Download Forecast Excel",
            data=output,
            file_name="NEST_Forecasts_MultiModel.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        # reset run flag after completion
        st.session_state['run_forecast'] = False

elif uploaded and not st.session_state['run_forecast']:
    st.info("✅ File uploaded. Click **Start forecasting** when you're ready.")
else:
    st.info("👆 Upload your CSV to get started.")
