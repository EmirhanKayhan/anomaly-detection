from kafka import KafkaProducer
import json
import pandas as pd
import time
import random

class AnomalyDataProducer:
    def __init__(self, bootstrap_servers=['localhost:9092'], topic_name='network_data'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic_name = topic_name
        self.data = self.load_data()
    
    def load_data(self):
        """Load preprocessed data"""
        return pd.read_csv('/home/emirhan/anomaly-detection/data/X_test.csv')
    
    def send_data_stream(self, num_messages=100, delay=1):
        """
        Send data stream to Kafka topic
        Simulate real-time data streaming
        """
        for _ in range(num_messages):
            # Randomly select a row
            row = self.data.sample(1).to_dict(orient='records')[0]
            
            # Simulate some random anomaly injection
            if random.random() < 0.1:  # 10% chance of anomaly
                row['is_anomaly'] = True
                # Introduce some extreme values
                for col in row.keys():
                    if isinstance(row[col], (int, float)):
                        row[col] *= random.uniform(2, 5)
            else:
                row['is_anomaly'] = False
            
            # Send to Kafka
            self.producer.send(self.topic_name, row)
            print(f"Sent: {row}")
            
            time.sleep(delay)
        
        self.producer.flush()
        self.producer.close()

def main():
    producer = AnomalyDataProducer()
    producer.send_data_stream()

if __name__ == "__main__":
    main()
