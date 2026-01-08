import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor
import datetime
import uuid
import time
import json

from process import Process
from deploy_uniform.utils import RTSPFFmpegReader, RTSPFFmpegWriter


IMG_SAVE_PATH = "./deploy/image/"
LOG_PATH = "./deploy/log/log_test.jsonl"
IMG_BASE_URL = "http://103.155.161.67:8101/static/"


POLICY_CONFIG = {
    "hat": {
        "policyName": "Giám sát mũ bảo hộ",
        "actionContent": "Phát hiện không đội mũ bảo hộ",
        "policyType": "101",        
        "msgSource": "MBF_CORE_HAT",
        "key_attr": "hat_check",    
        "val_attr": "0"
    },
    "uniform": {
        "policyName": "Giám sát đồng phục",
        "actionContent": "Phát hiện không mặc đồng phục",
        "policyType": "100",        
        "msgSource": "MBF_CORE_UNIFORM",
        "key_attr": "uniform_check",
        "val_attr": "0"
    }
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
    p.add_argument("--transport", default="auto")

    p.add_argument("--output_rtsp", default="rtsp://127.0.0.1:8554/live")

    p.add_argument("--conf_uniform", type=float, default=0.40)
    p.add_argument("--conf_hat", type=float, default=0.25)

    p.add_argument("--w_model", type=float, default=0.6)
    p.add_argument("--w_logic", type=float, default=0.4)
    p.add_argument("--final_thresh", type=float, default=0.5)
    p.add_argument("--roi_thresh", type=float, default=0.3)

    p.add_argument("--thresh_frames", type=int, default=10)
    p.add_argument("--cooldown_s", type=float, default=3.0)

    p.add_argument("--use_cv2_reader", default=True)
    return p.parse_args()


@torch.inference_mode()
def _infer_one(model, img, conf, imgsz=None):
    if imgsz is None:
        return model(img, conf=conf, verbose=False)[0]
    return model(img, conf=conf, imgsz=imgsz, verbose=False)[0]


def _infer_two(model_hat, model_uniform, img, conf_hat, conf_uniform, executor=None):
    if executor is None:
        r_hat = _infer_one(model_hat, img, conf_hat, imgsz=None)
        r_uni = _infer_one(model_uniform, img, conf_uniform, imgsz=None)
        return r_hat, r_uni
    f1 = executor.submit(_infer_one, model_hat, img, conf_hat, None)
    f2 = executor.submit(_infer_one, model_uniform, img, conf_uniform, None)
    return f1.result(), f2.result()


def draw_detections(frame, hat_dets, uni_dets, model_hat, model_uni):
    for box, cls in (uni_dets or []):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model_uni.names[int(cls)]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for box, cls, final_score, logic_score in (hat_dets or []):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if float(logic_score) > 0.8 else (0, 255, 255)
        label = f"{model_hat.names[int(cls)]} {float(final_score):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    infer_executor=None,
):
    h, w = frame.shape[:2]
    roi_y = int(h * roi_thresh)

    frame_draw = frame.copy()
    frame_draw = np.ascontiguousarray(frame_draw, dtype=np.uint8)

    res_hat, res_uni = _infer_two(model_hat, model_uniform, frame, conf_hat, conf_uniform, executor=infer_executor)

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
    cv2.putText(
        frame_draw, f"ROI: y>{roi_y}",
        (10, max(20, roi_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
    )

    # uniform names: {0:'non-uniform', 1:'uniform'} => violation cls=0
    is_uni_violation = False
    uni_violation_box = None
    for b, cls in (uni_dets or []):
        if int(cls) == 0:
            uni_violation_box = _xyxy_list(b) + [1.0]
            is_uni_violation = True
            break

    # hat names: {0:'Hardhat', 1:'NO-Hardhat'} => violation cls=1
    is_hat_violation = False
    hat_violation_box = None
    for b, cls, final_score, logic_score in (hat_dets or []):
        if int(cls) == 1:
            hat_violation_box = _xyxy_list(b) + [float(final_score)]
            is_hat_violation = True
            break

    processed = draw_detections(frame_draw, hat_dets, uni_dets, model_hat, model_uniform)
    return processed, (is_uni_violation, uni_violation_box), (is_hat_violation, hat_violation_box)


def send_violation_event(frame_bgr, violation_box, violation_type="uniform", device_id="CAM_01", producer=None):
    try:
        config = POLICY_CONFIG.get(violation_type)
        if not config:
            print(f"[ERROR] Unknown violation type: {violation_type}")
            return

        event_id = str(uuid.uuid4()) 
        
        curr_date = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(IMG_SAVE_PATH, curr_date)
        os.makedirs(save_dir, exist_ok=True)

        full_name = f"{event_id}_{violation_type}_full.jpg"
        target_name = f"{event_id}_{violation_type}_target.jpg"
        
        full_path = os.path.join(save_dir, full_name)
        target_path = os.path.join(save_dir, target_name)

        ok_full = cv2.imwrite(full_path, frame_bgr)
        if not ok_full:
            print("[ERROR] Failed to save full frame")
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

        final_target_name = target_name if has_crop else full_name

        full_url = f"{IMG_BASE_URL}{curr_date}/{full_name}"
        crop_url = f"{IMG_BASE_URL}{curr_date}/{final_target_name}"

        conf = "1.0"
        if violation_box is not None and len(violation_box) > 4:
            try:
                conf = f"{float(violation_box[4]):.2f}"
            except Exception:
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
                    "conf": conf, 
                    "value": config["val_attr"], 
                    "key": config["key_attr"]
                }
            ],
            "integrateQuality": "",
            "msgSource": config["msgSource"],
            "desc": f"{config['actionContent']} tại Camera: {device_id}",
        }

        if producer is not None:
            # producer.send("webhook.event", event)
            # producer.flush() # Nên flush để đảm bảo message được gửi
            print(f"[KAFKA] Sent event {event_id} ({violation_type})")
        else:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False)
                f.write("\n")
            print(f"[LOG] Saved event {event_id} ({violation_type}) to file")

    except Exception as e:
        print(f"[EXCEPTION] Error sending {violation_type} event: {e}")
        import traceback
        traceback.print_exc()

def process_rtsp_test_ac():
    args = parse_args()

    model_hat = YOLO(args.model_hat)
    model_uniform = YOLO(args.model_uniform)

    infer_executor = ThreadPoolExecutor(max_workers=2)
    event_executor = ThreadPoolExecutor(max_workers=2)

    cnt_uniform = 0
    cnt_hat = 0
    last_time_uniform = 0.0
    last_time_hat = 0.0

    rtsp = f"rtsp://{args.username}:{args.password}@{args.ip_cam}:{args.port}/{args.path}"

    if args.use_cv2_reader:
        reader = cv2.VideoCapture(rtsp)
        reader.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        read_fn = lambda: reader.read()
        close_reader = lambda: reader.release()
    else:
        reader = RTSPFFmpegReader(rtsp, args.w, args.h, use_gpu=True, transport=args.transport, buffer_size=3)
        read_fn = lambda: reader.read()
        close_reader = lambda: reader.close()

    writer = RTSPFFmpegWriter(
        args.output_rtsp,
        args.w, args.h,
        use_gpu=True,
        gpu_id=0,
    )

    frame_count = 0
    t_start = time.monotonic()
    t_last = t_start
    n_last = 0

    try:
        while True:
            ret, frame = read_fn()
            if not ret or frame is None:
                continue

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
                infer_executor=infer_executor,
            )

            now = time.monotonic()

            is_uni_bad, uni_box = uni_info
            if is_uni_bad:
                cnt_uniform += 1
            else:
                cnt_uniform = 0

            if cnt_uniform >= args.thresh_frames and (now - last_time_uniform) > args.cooldown_s:
                img = processed.copy()  # có ROI + box
                event_executor.submit(send_violation_event, img, uni_box, "uniform", "CAM_01", None)
                last_time_uniform = now
                cnt_uniform = 0

            is_hat_bad, hat_box = hat_info
            if is_hat_bad:
                cnt_hat += 1
            else:
                cnt_hat = 0

            if cnt_hat >= args.thresh_frames and (now - last_time_hat) > args.cooldown_s:
                img = processed.copy()
                event_executor.submit(send_violation_event, img, hat_box, "hat", "CAM_01", None)
                last_time_hat = now
                cnt_hat = 0

            writer.write(processed)
            frame_count += 1

            if now - t_last >= 2.0:
                inst_fps = (frame_count - n_last) / max(1e-6, (now - t_last))
                avg_fps = frame_count / max(1e-6, (now - t_start))
                print(f"FPS(inst): {inst_fps:.1f} | FPS(avg): {avg_fps:.1f} | frames: {frame_count}")
                t_last = now
                n_last = frame_count

    except KeyboardInterrupt:
        pass
    finally:
        try:
            close_reader()
        except Exception:
            pass
        try:
            infer_executor.shutdown(wait=True)
        except Exception:
            pass
        try:
            event_executor.shutdown(wait=True)
        except Exception:
            pass


if __name__ == "__main__":
    process_rtsp_test_ac()
