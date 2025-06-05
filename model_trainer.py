import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class XGBoostTrainer:
    def __init__(self, filepath):
        self.file_path = filepath
        self.models = {}
        self.interval_length = 300  # 5 minutes
        self.rolling_window = 2     # For rolling statistics

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data['start_time'] = data['start_time'] / 1e6
        data['end_time'] = data['end_time'] / 1e6
        data = data.sort_values(by='start_time')
        return data

    def process_interval(self, interval_data):
        return {
            'unique_collection_ids': interval_data['collection_id'].nunique(),
            'non_unique_collection_ids': interval_data['collection_id'].value_counts().gt(1).sum(),
            'unique_machine_ids': interval_data['machine_id'].nunique(),
            'sum_avg_cpu': interval_data['average_usage.cpus'].sum(),
            'sum_avg_memory': interval_data['average_usage.memory'].sum(),
            'sum_max_cpu': interval_data['maximum_usage.cpus'].sum(),
            'sum_max_memory': interval_data['maximum_usage.memory'].sum()
        }

    def prepare_features(self, data):
        results = []
        current_start = data['start_time'].min()
        end_time = data['start_time'].max()

        while current_start < end_time:
            current_end = current_start + self.interval_length
            interval_data = data[(data['start_time'] >= current_start) & (data['start_time'] < current_end)]
            if not interval_data.empty:
                row = self.process_interval(interval_data)
                row['interval_start'] = current_start
                results.append(row)
            current_start = current_end

        df = pd.DataFrame(results)

        # Lag features
        for lag in range(1, 3):
            for col in ['sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Rolling stats
        for col in ['sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=self.rolling_window).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=self.rolling_window).std()

        # Time features
        df['hour_of_day'] = pd.to_datetime(df['interval_start'], unit='s').dt.hour
        df['day_of_week'] = pd.to_datetime(df['interval_start'], unit='s').dt.dayofweek

        # Drop rows with NaNs from lag/rolling
        return df.dropna().reset_index(drop=True)

    def prepare_data(self):
        data = self.load_data()
        train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)
        train_df = self.prepare_features(train_data)
        test_df = self.prepare_features(test_data)

        features = [col for col in train_df.columns if col not in [
            'sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']]

        X_train = train_df[features]
        y_train = train_df[['sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']]
        X_test = test_df[features]
        y_test = test_df[['sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']]

        return X_train, y_train, X_test, y_test

    def train_and_evaluate(self):
        X_train, y_train, X_test, y_test = self.prepare_data()
        results = {}
        predictions = pd.DataFrame()

        for target in y_train.columns:
            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
            model.fit(X_train, y_train[target])
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test[target], y_pred)
            self.models[target] = model
            results[target] = mse
            predictions[target + '_pred'] = y_pred

        return results, self.models, predictions, y_test.reset_index(drop=True)

    def save_models(self, output_dir, client_id):
        for target, model in self.models.items():
            model_filename = f"{client_id}_{target}_xgb.json"
            model.save_model(os.path.join(output_dir, model_filename))

    def load_models(self, model_dir):
        for target in ['sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']:
            model = xgb.XGBRegressor()
            model.load_model(f"{model_dir}/{target}_xgb.json")
            self.models[target] = model