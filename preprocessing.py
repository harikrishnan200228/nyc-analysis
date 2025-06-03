import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()
    df['OCCUR_DATE'] = pd.to_datetime(df['OCCUR_DATE'], errors='coerce')
    df = df.dropna(subset=['OCCUR_DATE', 'LAT', 'LON'])
    df['hour'] = df['OCCUR_DATE'].dt.hour
    df['day_of_week'] = df['OCCUR_DATE'].dt.dayofweek
    df['month'] = df['OCCUR_DATE'].dt.month
    nyc_lat, nyc_lon = 40.7128, -74.0060
    df['distance_from_center'] = ((df['LAT'] - nyc_lat)**2 + (df['LON'] - nyc_lon)**2) ** 0.5
    features = ['LAT', 'LON', 'hour', 'day_of_week', 'distance_from_center']
    df = df.dropna(subset=features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, df
