import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from ast import literal_eval
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
from typing import List

#用来测试的class，实际调用的时候直接改成实际的class就行
@dataclass
class UserBehaviorComplete:
    device_id: str
    property_id: int
    dwell_time: float
    favorite: bool
    update_time: datetime
    name: str
    district: str
    price: float
    beds: int
    baths: int
    area: int
    build_time: str
    location: str
    time_to_school: int
    distance_to_mrt: int
    latitude: float
    longitude: float
    costScore: float
    commuteScore: float
    neighborhoodScore: float
    public_facilities: List[Dict[str, int]]
    facility_type: str


_model = None
_scaler = None
_label_encoders = None
_feature_names = None
_sequence_length = None


def load_model_and_metadata(model_path='property_lstm_model.keras', metadata_path='property_model_metadata.pkl'):
    model = keras.models.load_model(model_path)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return model, metadata['scaler'], metadata['label_encoders'], metadata['feature_names'], metadata['sequence_length']

def parse_public_facilities(facilities):
    try:
        if not facilities or facilities == []:
            return 0, 0
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

def preprocess_single_behavior(behavior, label_encoders):
    num_facilities, avg_facility_distance = parse_public_facilities(behavior.public_facilities)
    hour = behavior.update_time.hour
    day_of_week = behavior.update_time.weekday()
    favorite = 1 if behavior.favorite else 0
    district_encoded = label_encoders['district'].transform([behavior.district])[0]
    facility_type_encoded = label_encoders['facility_type'].transform([behavior.facility_type])[0]
    features = [
        float(behavior.price.replace('$', '').replace(',', '')) if isinstance(behavior.price, str) else behavior.price,
        behavior.beds,
        behavior.baths,
        behavior.area,
        int(behavior.build_time) if behavior.build_time else 0,
        behavior.time_to_school,
        behavior.distance_to_mrt,
        behavior.latitude,
        behavior.longitude,
        behavior.costScore,
        behavior.commuteScore,
        behavior.neighborhoodScore,
        behavior.dwell_time,
        favorite,
        num_facilities,
        avg_facility_distance,
        hour,
        day_of_week,
        district_encoded,
        facility_type_encoded
    ]
    return features


def predict_omega(user_behaviors, model, scaler, label_encoders, feature_names, sequence_length):
    sorted_behaviors = sorted(user_behaviors, key=lambda x: x.update_time)
    features_list = [preprocess_single_behavior(b, label_encoders) for b in sorted_behaviors]
    features_array = np.array(features_list)

    seq_len = min(len(features_array), sequence_length)
    sequence = features_array[-seq_len:]
    if len(sequence) < sequence_length:
        padding = np.zeros((sequence_length - len(sequence), len(feature_names)))
        sequence = np.vstack([padding, sequence])
    sequence = sequence.reshape(1, sequence_length, len(feature_names))
    sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
    sequence_scaled = scaler.transform(sequence_reshaped).reshape(sequence.shape)
    prediction = model.predict(sequence_scaled, verbose=0)[0]
    prediction = prediction / prediction.sum()
    return tuple(prediction) #这是最终输出的omega值，定义完omega的格式之后替换成omega的真正格式就行



def predict_user_omega(user_behaviors: List[UserBehaviorComplete]) -> tuple:
    global _model, _scaler, _label_encoders, _feature_names, _sequence_length

    if _model is None or _scaler is None:
        _model, _scaler, _label_encoders, _feature_names, _sequence_length = load_model_and_metadata()

    return predict_omega(
        user_behaviors,
        _model,
        _scaler,
        _label_encoders,
        _feature_names,
        _sequence_length
    )



if __name__ == "__main__":
    
    #用来测试的数据，可删
    sample_records_raw = [
        {
            "device_id": "device_123",
            "property_id": 1,
            "dwell_time": 30.5,
            "favorite": True,
            "update_time": "2024-01-15 10:30:00",
            "name": "Sunrise Residences",
            "district": "Marina Bay", 
            "price": 1600,
            "beds": 1,
            "baths": 0,
            "area": 65,
            "build_time": "2018",
            "location": "123 Orchard Road",
            "time_to_school": 30,
            "distance_to_mrt": 300,
            "latitude": 1.3521,
            "longitude": 103.8198,
            "costScore": 0.44,
            "commuteScore": 0.85,
            "neighborhoodScore": 0.90,
            "public_facilities": '[{"ABC Supermarket": 200}, {"Happy Shopping Mall": 500}]',
            "facility_type": "Condo"
        }
    ]
    

    sample_records = []
    for record in sample_records_raw:
        record['update_time'] = datetime.fromisoformat(record['update_time'])
        record['public_facilities'] = (
            literal_eval(record['public_facilities']) 
            if isinstance(record['public_facilities'], str) 
            else record['public_facilities']
        )
        sample_records.append(UserBehaviorComplete(**record))

    try:
        omega_result = predict_user_omega(sample_records)  #调用的时候直接用这个方法就行
        print("\n✅ Prediction completed!")
        print(f"User preference weights: {omega_result}")
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
