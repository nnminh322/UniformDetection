import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ.setdefault("OPENCV_FFMPEG_WRITER_OPTIONS", "rtsp_transport;tcp")

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import torch
import datetime
import uuid
import time
import json
from utils import RTSPFFmpegWriter
from process import Process
import random
from kafka_controller import KafkaProducer

IMG_SAVE_PATH = "./deploy/image/"
LOG_PATH = "./deploy/log/log_test.jsonl"
IMG_BASE_URL = "http://103.155.161.67:8322/image/"
conf = {
    'bootstrap.servers': '103.155.161.67:9093',
    # 'bootstrap.servers': 'localhost:9094,localhost:9096',
    'linger.ms': 10,  
    'socket.timeout.ms': 60000,   # Chờ lâu hơn tí cho chắc
    'client.id': 'server_producer_app',
}
POLICY_CONFIG = {
    "hat": {
        "policyName": "Giám sát mũ bảo hộ",
        "actionContent": "Phát hiện không đội mũ bảo hộ",
        "policyType": "32",
        "msgSource": "MBF_CORE_HAT",
        "key_attr": "hat_check",
        "val_attr": "0",
    },
    "uniform": {
        "policyName": "Giám sát đồng phục",
        "actionContent": "Phát hiện không mặc đồng phục",
        "policyType": "32",
        "msgSource": "MBF_CORE_UNIFORM",
        "key_attr": "uniform_check",
        "val_attr": "0",
    },
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_hat", type=str, default="./models/hat_final.pt")
    p.add_argument("--model_uniform", type=str, default="./models/costume_final.pt")
    p.add_argument("--ip_cam", default="113.163.156.210")
    p.add_argument("--port", default="555")
    p.add_argument("--path", default="4/1")
    p.add_argument("--username", default="mobiphone")
    p.add_argument("--password", default="Hatien2025")
    p.add_argument("--w", type=int, default=1920)
    p.add_argument("--h", type=int, default=1080)
    p.add_argument("--output_rtsp", default="rtsp://127.0.0.1:8554/live")
    p.add_argument("--fps_out", type=float, default=25.0)
    p.add_argument("--fourcc", default="H264")
    p.add_argument("--conf_uniform", type=float, default=0.40)
    p.add_argument("--conf_hat", type=float, default=0.25)
    p.add_argument("--w_model", type=float, default=0.6)
    p.add_argument("--w_logic", type=float, default=0.4)
    p.add_argument("--final_thresh", type=float, default=0.5)
    p.add_argument("--roi_thresh", type=float, default=0.3)
    p.add_argument("--thresh_frames", type=int, default=2)  #
    p.add_argument("--cooldown_s", type=float, default=2)  #
    return p.parse_args()


@torch.inference_mode()
def _infer_one(model, img, conf):
    return model(img, conf=conf, verbose=False)[0]


def _infer_two(model_hat, model_uniform, img, conf_hat, conf_uniform):
    r_hat = _infer_one(model_hat, img, conf_hat)
    r_uni = _infer_one(model_uniform, img, conf_uniform)
    return r_hat, r_uni


def draw_detections(frame, hat_dets, uni_dets, model_hat, model_uni):
    for box, cls in (uni_dets or []):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model_uni.names[int(cls)]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for box, cls, final_score, logic_score in (hat_dets or []):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if float(logic_score) > 0.8 else (0, 255, 255)
        label = f"{model_hat.names[int(cls)]} {float(final_score):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def _xyxy_list(b):
    if hasattr(b, "tolist"):
        b = b.tolist()
    return list(map(float, b[:4]))


def process_frame_optimize_2(
    frame,
    model_hat,
    model_uniform,
    conf_hat=0.25,
    conf_uniform=0.40,
    w_model=0.6,
    w_logic=0.4,
    final_thresh=0.5,
    roi_thresh=0.3,
):
    h, w = frame.shape[:2]
    roi_y = int(h * roi_thresh)

    frame_infer = np.ascontiguousarray(frame, dtype=np.uint8)
    frame_draw = np.ascontiguousarray(frame.copy(), dtype=np.uint8)

    res_hat, res_uni = _infer_two(model_hat, model_uniform, frame_infer, conf_hat, conf_uniform)

    processor = Process(
        res_hat, res_uni,
        frame_height=h,
        w_model=w_model,
        w_logic=w_logic,
        final_thresh=final_thresh,
        roi_thresh=roi_thresh,
    )
    processor.apply_ensemble_logic()
    hat_dets, uni_dets = processor.get_final_detections()

    cv2.line(frame_draw, (0, roi_y), (w, roi_y), (0, 165, 255), 2)
    cv2.putText(frame_draw, f"ROI: y>{roi_y}", (10, max(20, roi_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    is_uni_violation = False
    uni_violation_box = None
    for b, cls in (uni_dets or []):
        if int(cls) == 0:
            uni_violation_box = _xyxy_list(b) + [1.0]
            is_uni_violation = True
            break

    is_hat_violation = False
    hat_violation_box = None
    for b, cls, final_score, logic_score in (hat_dets or []):
        if int(cls) == 1:
            hat_violation_box = _xyxy_list(b) + [float(final_score)]
            is_hat_violation = True
            break

    processed = draw_detections(frame_draw, hat_dets, uni_dets, model_hat, model_uniform)
    return processed, (is_uni_violation, uni_violation_box), (is_hat_violation, hat_violation_box)

def generate_safe_id():
    timestamp_ms = int(time.time() * 1000)
    rand_part = random.randint(100, 999)
    return str(timestamp_ms) + str(rand_part)


def send_violation_event(frame_bgr, violation_box, violation_type="uniform", device_id="CAM_VICEM_CONG_CHINH", producer=None):
    try:
        if frame_bgr is None:
            print("[ERROR] Empty frame provided")
            return

        config = POLICY_CONFIG.get(violation_type)
        if not config:
            print(f"[ERROR] Unknown violation type: {violation_type}")
            return

        event_id = generate_safe_id()
        curr_date = datetime.datetime.now().strftime("%Y%m%d")
        
        save_dir = os.path.join(IMG_SAVE_PATH, curr_date, violation_type)
        os.makedirs(save_dir, exist_ok=True)

        full_name = f"{event_id}_full.jpg"
        target_name = f"{event_id}_target.jpg"
        
        full_path = os.path.join(save_dir, full_name)
        target_path = os.path.join(save_dir, target_name)

        if not cv2.imwrite(full_path, frame_bgr):
            print("[ERROR] Failed to write full image to disk")
            return

        has_crop = False
        if violation_box is not None and len(violation_box) >= 4:
            x1, y1, x2, y2 = map(int, violation_box[:4])
            H, W = frame_bgr.shape[:2]
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(target_path, crop)
                    has_crop = True

        full_url = f"{IMG_BASE_URL}{curr_date}/{violation_type}/{full_name}"
        crop_url = f"{IMG_BASE_URL}{curr_date}/{violation_type}/{target_name if has_crop else full_name}"

        conf_str = "1.0"
        if violation_box is not None and len(violation_box) > 4:
            try:
                conf_val = float(violation_box[4])
                conf_str = f"{conf_val:.2f}"
            except (ValueError, TypeError):
                pass

        event = {
            "triggerEventId": event_id,
            "policyName": config["policyName"],
            "bkImageUrl": full_url,           
            "triggerTime": time.time(),
            "deviceId": device_id,
            "targetImgUrl": crop_url,          
            "finalQuality": "",
            "actionContent": config["actionContent"],
            "matchImageUrl": "",
            "similarity": "",
            "policyType": config["policyType"],
            "triggerImgUrl": full_url,       
            "attributes": [
                {
                    "conf": conf_str, 
                    "value": config["val_attr"], 
                    "key": config["key_attr"]
                }
            ],
            "integrateQuality": "",
            "msgSource": config["msgSource"],
            "desc": f"{config['actionContent']} tại Camera: {device_id}",
        }

    # if producer:
        producer.send_json("webhook.event", event)
        producer.send_json("uniform_natmin", event)
        # producer.flush() 
        print(f"[KAFKA] Sent event {event_id} - Type: {violation_type}")
    # else:
    #     os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
            f.write("\n")
    #     print(f"[LOG] Saved event {event_id} - Type: {violation_type}")

    except Exception as e:
        print(f"[CRITICAL] Error in send_violation_event: {str(e)}")

def process_rtsp_test_ac():
    args = parse_args()
    produce = KafkaProducer(conf)
    model_hat = YOLO(args.model_hat)
    model_uniform = YOLO(args.model_uniform)

    rtsp_in = f"rtsp://{args.username}:{args.password}@{args.ip_cam}:{args.port}/{args.path}"
    print(rtsp_in)
    cap = cv2.VideoCapture(rtsp_in)
    if not cap.isOpened():
        raise RuntimeError("cv2.VideoCapture open failed")

    writer = RTSPFFmpegWriter(
        args.output_rtsp,  
        args.w, args.h,
        use_gpu=True,
        gpu_id=0,
    )

    cnt_uniform = 0
    cnt_hat = 0
    last_time_uniform = 0.0
    last_time_hat = 0.0

    frame_count = 0
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if frame.shape[1] != args.w or frame.shape[0] != args.h:
                frame = cv2.resize(frame, (args.w, args.h), interpolation=cv2.INTER_LINEAR)

            processed, uni_info, hat_info = process_frame_optimize_2(
                frame,
                model_hat,
                model_uniform,
                conf_hat=args.conf_hat,
                conf_uniform=args.conf_uniform,
                w_model=args.w_model,
                w_logic=args.w_logic,
                final_thresh=args.final_thresh,
                roi_thresh=args.roi_thresh,
            )

            curr_t = time.time()

            is_uni_bad, uni_box = uni_info
            if is_uni_bad:
                cnt_uniform += 1
            else:
                cnt_uniform = 0
            if cnt_uniform >= args.thresh_frames and (curr_t - last_time_uniform) > args.cooldown_s:
                send_violation_event(processed.copy(), uni_box, "uniform", "CAM_VICEM_CONG_CHINH", producer=produce)
                last_time_uniform = curr_t
                cnt_uniform = 0

            is_hat_bad, hat_box = hat_info
            if is_hat_bad:
                cnt_hat += 1
            else:
                cnt_hat = 0
            if cnt_hat >= args.thresh_frames and (curr_t - last_time_hat) > args.cooldown_s:
                send_violation_event(processed.copy(), hat_box, "hat", "CAM_VICEM_CONG_CHINH", producer=produce)
                last_time_hat = curr_t
                cnt_hat = 0

            writer.write(processed)

            frame_count += 1
            if frame_count % 100 == 0:
                fps = frame_count / max(1e-6, (time.time() - start))
                print(f"Processed {frame_count} frames | Overall FPS: {fps:.1f}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        try:
            writer.close()
        except Exception:
            pass


if __name__ == "__main__":
    process_rtsp_test_ac()
