# kafka_service.py
import json
import time
import datetime

# Intentar importar kafka, si falla, usaremos un modo simulado
try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Kafka no está disponible - usando modo simulado")

class MovieKafkaProducer:
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='movie-recommendations-log'):
        if not KAFKA_AVAILABLE:
            self.connected = False
            print("Kafka no está disponible - usando modo simulado")
            return
            
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            self.topic = topic
            self.connected = True
            print(f"Kafka producer connected to {bootstrap_servers}")
        except Exception as e:
            print(f"Failed to connect to Kafka: {str(e)}")
            self.connected = False
    
    def log_recommendation(self, user_id, server, status, recommendations, response_time):
        """Log recommendation request to Kafka"""
        if not KAFKA_AVAILABLE or not self.connected:
            # En modo simulado, solo imprimimos el log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            message = f"{timestamp},{user_id},recommendation request {server}, status {status}, result: {recommendations}, {response_time}"
            print(f"SIMULATED LOG: {message}")
            return True
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            message = f"{timestamp},{user_id},recommendation request {server}, status {status}, result: {recommendations}, {response_time}"
            
            self.producer.send(self.topic, value={"log": message})
            return True
        except Exception as e:
            print(f"Error sending log to Kafka: {str(e)}")
            return False
    
    def close(self):
        """Close the Kafka producer"""
        if KAFKA_AVAILABLE and self.connected:
            self.producer.close()

class MovieKafkaConsumer:
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='movie-recommendations-log'):
        if not KAFKA_AVAILABLE:
            self.connected = False
            print("Kafka no está disponible - usando modo simulado")
            return
            
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            self.connected = True
            print(f"Kafka consumer connected to {bootstrap_servers}")
        except Exception as e:
            print(f"Failed to connect to Kafka: {str(e)}")
            self.connected = False
    
    def consume_logs(self):
        """Consume and print logs from Kafka"""
        if not KAFKA_AVAILABLE or not self.connected:
            print("Kafka no está disponible - usando modo simulado")
            print("En modo simulado, no hay logs para consumir")
            return
            
        print("Starting to consume logs from Kafka...")
        for message in self.consumer:
            log = message.value.get('log', '')
            print(f"LOG: {log}")
    
    def close(self):
        """Close the Kafka consumer"""
        if KAFKA_AVAILABLE and self.connected:
            self.consumer.close()