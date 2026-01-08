from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions
import json
import time
CONF = {
    'bootstrap.servers': '103.155.161.67:9093',
    'linger.ms': 10,  
    'socket.timeout.ms': 60000,   # Chờ lâu hơn tí cho chắc
    'client.id': 'server_producer_app',
}
class KafkaManager:
    def __init__(self, conf):
        if conf is None:
            conf = CONF
        self.admin = AdminClient(conf)

    def create_topic(self, topic_name, num_patition=1, replication_factor=1):
        new_topic = NewTopic(topic_name, num_patition, replication_factor)
        fs = self.admin.create_topics(new_topics=[new_topic])

        for topic, f in fs.items():
            try:
                f.result()
                print("Create topic kafka successfully!")
            except KafkaException as e:
                error_obj = e.args[0]
                if error_obj.code() == KafkaError.TOPIC_ALREADY_EXISTS:
                    print(f"⚠️ Topic '{topic}' existed. Its ok!")
                else:
                    raise Exception(f"❌ Can not create '{topic}': {e}")
            except Exception as e:
                raise Exception(f"System Error!: {f}")


class KafkaProducer:
    def __init__(self, conf):
        if conf is None:
            conf = CONF
        self.producer = Producer(conf)

    def _ack(self, err, msg):
        if err is not None:
            print(f"Send Error: {err}")
        else:
            print(
                f"Send successfully: {msg.value().decode('utf-8')} (Topic: {msg.topic()})"
            )

    def send_json(self, topic, data):
        json_data = json.dumps(data).encode("utf-8")
        self.producer.produce(topic, json_data, callback=self._ack)
        self.producer.poll(0)

    def flush(self):
        self.producer.flush()


class KafkaConsumer:
    def __init__(self, topic, group_id="Uniform"):

        conf = CONF.copy()
        conf.update(
            {
                "group.id": group_id,
                "auto.offset.reset": "earliest",  # Đọc từ đầu nếu là group mới
            }
        )
        self.consumer = Consumer(conf)
        self.consumer.subscribe([topic])

    def listen(self):
        print("🎧 Waiting msg... ")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"Lỗi Consumer: {msg.error()}")
                    continue

                raw_data = msg.value().decode("utf-8")
                data = json.loads(raw_data)
                print(f"📥 NHẬN ĐƯỢC: {data}")

        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()


# --- DEMO CÁCH DÙNG ---
# if __name__ == "__main__":
#     TOPIC = "camera_events"

#     # 1. Quản trị: Tạo topic
#     manager = KafkaManager()
#     manager.create_topic(TOPIC)

#     # 2. Producer: Bắn 5 tin giả lập
#     print("\n--- Bắt đầu gửi dữ liệu ---")
#     producer = KafkaProducer()
#     for i in range(5):
#         event = {"cam_id": 1, "frame_id": i, "status": "motion_detected"}
#         producer.send_json(TOPIC, event)
#         time.sleep(0.5)
#     producer.flush() # Đảm bảo tin đi hết

#     # 3. Consumer: Đọc lại tin vừa bắn
#     print("\n--- Bắt đầu đọc dữ liệu ---")
#     consumer = KafkaConsumer(TOPIC)
#     consumer.listen()
