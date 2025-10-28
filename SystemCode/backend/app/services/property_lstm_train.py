import pandas as pd
import numpy as np
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class PropertyBehaviorLSTM:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.sequence_length = 10 
        
    def parse_public_facilities(self, facilities_str):
        """Parse the public_facilities field to extract number of facilities and average distance"""
        try:
            if pd.isna(facilities_str) or facilities_str == '[]':
                return 0, 0
            facilities = eval(facilities_str)
            num_facilities = len(facilities)
            distances = []
            for facility in facilities:
                for name, dist in facility.items():
                    try:
                        distances.append(float(dist))
                    except:
                        pass
            avg_distance = np.mean(distances) if distances else 0
            return num_facilities, avg_distance
        except:
            return 0, 0
    
    def preprocess_data(self, df):
        """Preprocess raw data"""
        print("Starting data preprocessing...")
        df[['num_facilities', 'avg_facility_distance']] = df['public_facilities'].apply(
            lambda x: pd.Series(self.parse_public_facilities(x))
        )
        df['update_time'] = pd.to_datetime(df['update_time'])
        df['hour'] = df['update_time'].dt.hour
        df['day_of_week'] = df['update_time'].dt.dayofweek
        df['favorite'] = df['favorite'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
        categorical_features = ['district', 'facility_type']
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        feature_columns = [
            'price', 'beds', 'baths', 'area', 'build_time',
            'time_to_school', 'distance_to_mrt', 'latitude', 'longitude',
            'costScore', 'commuteScore', 'neighborhoodScore',
            'dwell_time', 'favorite',
            'num_facilities', 'avg_facility_distance',
            'hour', 'day_of_week',
            'district_encoded', 'facility_type_encoded'
        ]
        self.feature_names = feature_columns
        return df[feature_columns + ['device_id', 'property_id']], df
    
    def calculate_user_preferences(self, user_df):
        """
        Calculate preference weights based on user behavior
        Use weighted method: dwell time and favorite as weights
        """
        max_dwell = user_df['dwell_time'].max()
        if max_dwell > 0:
            user_df['dwell_weight'] = user_df['dwell_time'] / max_dwell
        else:
            user_df['dwell_weight'] = 0
        user_df['behavior_weight'] = user_df['dwell_weight'] * 0.6 + user_df['favorite'] * 0.4
        total_weight = user_df['behavior_weight'].sum()
        if total_weight == 0:
            return np.array([0.33, 0.33, 0.34])
        weighted_cost = (user_df['costScore'] * user_df['behavior_weight']).sum()
        weighted_commute = (user_df['commuteScore'] * user_df['behavior_weight']).sum()
        weighted_neighbor = (user_df['neighborhoodScore'] * user_df['behavior_weight']).sum()
        total = weighted_cost + weighted_commute + weighted_neighbor
        if total > 0:
            omega_cost = weighted_cost / total
            omega_commute = weighted_commute / total
            omega_neighbor = weighted_neighbor / total
        else:
            omega_cost, omega_commute, omega_neighbor = 0.33, 0.33, 0.34
        return np.array([omega_cost, omega_commute, omega_neighbor])
    
    def create_sequences(self, df):
        """Create time series data"""
        print("Creating sequence data...")
        sequences = []
        targets = []
        device_ids = []
        for device_id, group in df.groupby('device_id'):
            group = group.sort_values('update_time')
            if len(group) < 3:
                continue
            omega = self.calculate_user_preferences(group)
            features = group[self.feature_names].values
            for i in range(len(features)):
                if i + 1 < len(features):
                    seq_len = min(i + 1, self.sequence_length)
                    seq = features[max(0, i + 1 - seq_len):i + 1]
                    if len(seq) < self.sequence_length:
                        padding = np.zeros((self.sequence_length - len(seq), features.shape[1]))
                        seq = np.vstack([padding, seq])
                    sequences.append(seq)
                    targets.append(omega)
                    device_ids.append(device_id)
        return np.array(sequences), np.array(targets), device_ids
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        print("Building LSTM model...")
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(32, dropout=0.2),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train(self, csv_path, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        print(f"Loading data: {csv_path}")
        df = pd.read_csv(csv_path)
        processed_df, original_df = self.preprocess_data(df)
        processed_df['update_time'] = original_df['update_time']
        X, y, device_ids = self.create_sequences(processed_df)
        print(f"Number of sequences: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        self.model = self.build_model((X.shape[1], X.shape[2]))
        print("\nModel summary:")
        self.model.summary()
        print("\nStart training...")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("\nModel evaluation:")
        val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print("\nPrediction samples (first 5):")
        predictions = self.model.predict(X_val[:5], verbose=0)
        for i in range(min(5, len(predictions))):
            print(f"True: ω_cost={y_val[i][0]:.3f}, ω_commute={y_val[i][1]:.3f}, ω_neighbor={y_val[i][2]:.3f}")
            print(f"Pred: ω_cost={predictions[i][0]:.3f}, ω_commute={predictions[i][1]:.3f}, ω_neighbor={predictions[i][2]:.3f}")
            print()
        return history
    
    def save_model(self, model_path='property_lstm_model.keras', metadata_path='model_metadata.pkl'):
        """Save model and metadata"""
        print(f"\nSaving model to: {model_path}")
        self.model.save(model_path)

        metadata = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata to: {metadata_path}")
        print("Model saved successfully!")

if __name__ == "__main__":
    lstm_model = PropertyBehaviorLSTM()
    base_dir = os.path.dirname(__file__)
    csv_file = os.path.abspath(os.path.join(base_dir, "../../../../Miscellaneous/behaviors.csv"))
    try:
        history = lstm_model.train(
            csv_path=csv_file,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        lstm_model.save_model(
            model_path='property_lstm_model.keras',
            metadata_path='property_model_metadata.pkl'
        )

        print("\n✅ Training completed!")
    except FileNotFoundError:
        print(f"❌ Error: File not found '{csv_file}'")
        print("Please make sure the CSV file exists and update the file path")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
