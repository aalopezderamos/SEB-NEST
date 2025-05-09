import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
import base64

st.set_page_config(page_title="NEST Forecast App", layout="wide")
st.title("📈 SEB NEST Forecast App")

# Upload section
uploaded_file = st.file_uploader("Upload your 'NEST Forecast Template.csv' file", type=["csv"])

# Define holidays
def get_custom_holidays():
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
                'holiday': name,
                'ds': date,
                'lower_window': -3,
                'upper_window': 3
            })

    return pd.DataFrame(records)

# Accuracy calc
def safe_accuracy(row):
    if pd.isna(row['y']) or row['y'] == 0:
        return None
    return round(1 - abs((row['y'] - row['yhat']) / row['y']), 3)

# Main logic
if uploaded_file:
    with st.spinner("Processing forecast..."):
        df = pd.read_csv(uploaded_file)
        df['ds'] = pd.to_datetime(df['ds'])
        sku_list = df['sku_id'].unique()
        custom_holidays = get_custom_holidays()
        detailed_forecasts = []

        for sku in sku_list:
            df_sku = df[df['sku_id'] == sku]
            if len(df_sku) < 50:
                continue

            try:
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    holidays=custom_holidays,
                    interval_width=0.90
                )
                m.fit(df_sku)

                last_date = df_sku['ds'].max()
                next_saturday = last_date + pd.DateOffset(days=(5 - last_date.weekday() + 7) % 7 + 1)
                future_saturdays = pd.date_range(start=next_saturday, periods=12, freq='W-SAT')
                future = pd.concat([df_sku[['ds']], pd.DataFrame({'ds': future_saturdays})], ignore_index=True)

                forecast = m.predict(future)
                df_actual = df_sku[['ds', 'y']]
                forecast = pd.merge(forecast, df_actual, on='ds', how='left')
                forecast['accuracy'] = forecast.apply(safe_accuracy, axis=1)
                forecast['sku_id'] = sku

                breakdown = forecast[['ds', 'sku_id', 'y', 'yhat', 'yhat_lower', 'yhat_upper',
                                      'accuracy', 'trend', 'weekly', 'yearly', 'holidays']]
                detailed_forecasts.append(breakdown)

            except Exception as e:
                st.warning(f"⚠️ Error processing SKU {sku}: {e}")

        if detailed_forecasts:
            df_detailed = pd.concat(detailed_forecasts)

            # Display preview
            st.success("✅ Forecasting complete!")
            st.subheader("📊 Forecast Preview")
            st.dataframe(df_detailed.head(20))

            # Export as Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_detailed.to_excel(writer, index=False)
            output.seek(0)

            b64 = base64.b64encode(output.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="NEST_Forecasts_Final.xlsx">📥 Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("❌ No valid forecasts generated. Check your data and try again.")
