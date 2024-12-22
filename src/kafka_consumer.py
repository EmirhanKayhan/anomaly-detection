from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
import numpy as np

class AnomalyDetector:
    def __init__(self, 
                 bootstrap_servers=['localhost:9092'], 
                 input_topic='network_data', 
                 anomaly_topic='anomalies',
                 normal_topic='normal_data'):
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Load pre-trained models
        self.rf_model = joblib.load('/home/emirhan/anomaly-detection/src/random_forest_model.joblib')
        self.label_encoder = joblib.load('/home/emirhan/anomaly-detection/src/label_encoder.joblib')
        
        self.input_topic = input_topic
        self.anomaly_topic = anomaly_topic
        self.normal_topic = normal_topic
    
    def detect_anomalies(self):
        """
        Consume messages from Kafka, detect anomalies using Random Forest
        """
        for message in self.consumer:
            data = message.value
            
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
            
            # Make prediction
            prediction = self.rf_model.predict(df)
            prediction_label = self.label_encoder.inverse_transform(prediction)[0]
            
            # Add timestamp and prediction to data
            data['timestamp'] = pd.Timestamp.now().isoformat()
            data['predicted_label'] = prediction_label
            
            # Classify and print results
            if prediction_label != 'Normal' or data.get('is_anomaly', False):
                print(f"\n🚨 ANOMALY DETECTED 🚨")
                print(f"Timestamp: {data['timestamp']}")
                print(f"Predicted Type: {prediction_label}")
                print(f"Key Indicators:")
                print(f"- Flow Duration: {data.get('flow_duration', 'N/A')}")
                print(f"- Packets per sec: {data.get('flow_pkts_per_sec', 'N/A')}")
                print(f"- Bytes per sec: {data.get('payload_bytes_per_second', 'N/A')}")
            else:
                print(f"Normal Traffic - {data['timestamp']}")

def main():
    detector = AnomalyDetector()
    print("Starting Anomaly Detection... Press Ctrl+C to stop.")
    try:
        detector.detect_anomalies()
    except KeyboardInterrupt:
        print("\nStopping Anomaly Detection...")

if __name__ == "__main__":
    main()
