import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from process import Process
from deploy_uniform.utils import RTSPFFmpegReader, RTSPFFmpegWriter
import torch
from concurrent.futures import ThreadPoolExecutor
import datetime
import uuid
import time
import json



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="./models/hat_final.pt")
    parser.add_argument("--model2", type=str, default="./models/costume_final.pt")
    parser.add_argument("--source", type=str, default="input_frame")
    parser.add_argument("--output", type=str, default="output1")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--w_model", type=float, default=0.6)
    parser.add_argument("--w_logic", type=float, default=0.4)
    parser.add_argument("--final_thresh", type=float, default=0.5)
    parser.add_argument(
        "--roi_thresh",
        type=float,
        default=0.3,
    )
    parser.add_argument("--ip_cam", default="113.163.156.210")
    parser.add_argument("--port", default="555")
    parser.add_argument("--path", default="4/1")
    parser.add_argument("--username", default="mobiphone")
    parser.add_argument("--password", default="Hatien2025")
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--h", type=int, default=1080)
    parser.add_argument("--transport", default="tcp")
    parser.add_argument("--output_rtsp", default="rtsp://localhost:8554/live", help="Link RTSP server nội bộ")
    return parser.parse_args()


def _infer_one(model, img, conf, imgsz):
    with torch.inference_mode():
        if imgsz is None:
            return model(img, conf=conf, verbose=False)[0]
        else:
            return model(img, conf=conf, imgsz=imgsz, verbose=False)[0]

def _infer_two(model1, model2, img, conf1, conf2, imgsz, executor=None):
    if executor is None:
        r1 = _infer_one(model1, img, conf1, imgsz)
        r2 = _infer_one(model2, img, conf2, imgsz)
        return r1, r2

    f1 = executor.submit(_infer_one, model1, img, conf1, imgsz)
    f2 = executor.submit(_infer_one, model2, img, conf2, imgsz)
    return f1.result(), f2.result()


# INFER_W, INFER_H = 960, 544  
# _SMALL = np.empty((INFER_H, INFER_W, 3), dtype=np.uint8)
IMG_SAVE_PATH = "./deploy/image/"
LOG_PATH = "./deploy/log/log_test.jsonl"
IMG_BASE_URL = "http://103.155.161.67:8101/static/"



# def _scale_boxes_inplace(res, sx: float, sy: float):
#     boxes = getattr(res, "boxes", None)
#     if boxes is None or boxes.data is None or len(boxes) == 0:
#         return

#     d = boxes.data
#     d2 = d.clone()
#     d2[:, 0] *= sx
#     d2[:, 2] *= sx
#     d2[:, 1] *= sy
#     d2[:, 3] *= sy

#     try:
#         boxes.data = d2
#     except Exception:
#         BoxesCls = type(boxes)
#         res.boxes = BoxesCls(d2, boxes.orig_shape)
        

def draw_detections(frame, hat_detections, uniform_detections, model1, model2):
    for box, cls in uniform_detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model2.names[int(cls)]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

    for box, cls, final_score, logic_score in hat_detections:
        x1, y1, x2, y2 = map(int, box)

        color = (0, 255, 0) if logic_score > 0.8 else (0, 255, 255)

        label = f"{model1.names[int(cls)]} {final_score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return frame




def send_violation_event(frame, violation_box, violation_type="uniform", device_id="camera_01", producer=None):
    """
    violation_type: "uniform" hoặc "hat"
    """
    try:
        if violation_type == "hat":
            policy_name = "Giám sát mũ bảo hộ"
            action_content = "Phát hiện không đội mũ bảo hộ"
            policy_type = "101" 
            msg_source = "MBF_CORE_HAT"
            key_attr = "hat_check"
            val_attr = "0" 
        else: 
            policy_name = "Giám sát đồng phục"
            action_content = "Phát hiện không mặc đồng phục"
            policy_type = "100"
            msg_source = "MBF_CORE_UNIFORM"
            key_attr = "uniform_check"
            val_attr = "0" 

        curr_date = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(IMG_SAVE_PATH, curr_date)
        os.makedirs(save_dir, exist_ok=True)

        event_id = str(uuid.uuid4())
        
        full_name = f"{event_id}_{violation_type}_full.jpg"
        cv2.imwrite(os.path.join(save_dir, full_name), frame)
        
        target_name = f"{event_id}_{violation_type}_target.jpg"
        if violation_box is not None and len(violation_box) >= 4:
            x1, y1, x2, y2 = map(int, violation_box[:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size > 0:
                cv2.imwrite(os.path.join(save_dir, target_name), crop_img)
            else:
                target_name = full_name
        else:
             target_name = full_name

        full_url = f"{IMG_BASE_URL}{curr_date}/{full_name}"
        crop_url = f"{IMG_BASE_URL}{curr_date}/{target_name}"

        event = {
            "triggerEventId": event_id,
            "policyName": policy_name,
            "bkImageUrl": full_url,
            "triggerTime": time.time(),
            "deviceId": device_id,
            "targetImgUrl": crop_url,
            "finalQuality": "",
            "actionContent": action_content,
            "matchImageUrl": "",
            "similarity": "",
            "policyType": policy_type,
            "triggerImgUrl": crop_url,
            "attributes": [
                {
                    "conf": f"{float(violation_box[4]):.2f}" if violation_box is not None and len(violation_box)>4 else "1.0",
                    "value": val_attr,
                    "key": key_attr
                }
            ],
            "integrateQuality": "",
            "msgSource": msg_source,
            "desc": f"{action_content} tại Camera: {device_id}"
        }
        producer = None
        if producer:
            # producer.send("webhook.event", event)
            print(f"[EVENT-{violation_type.upper()}] Sent Kafka: {event_id}")
        else:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False)
                f.write("\n")
            print(f"[EVENT-{violation_type.upper()}] Logged to file: {event_id}")

    except Exception as e:
        print(f"Error sending {violation_type} event: {e}")
        
        

# def process_frame_optimize(frame, model1, model2, output, conf_thresh=0.40,
#                            w_model=0.6, w_logic=0.4, final_thresh=0.5, roi_thresh=0.3,
#                            executor=None):
#     h, w = frame.shape[:2]
#     cv2.resize(frame, (INFER_W, INFER_H), dst=_SMALL, interpolation=cv2.INTER_LINEAR)
#     res1, res2 = _infer_two(
#         model1, model2, _SMALL,
#         conf1=0.25, conf2=conf_thresh,
#         imgsz=(INFER_H, INFER_W),
#         executor=executor
#     )

#     sx = w / INFER_W
#     sy = h / INFER_H
#     _scale_boxes_inplace(res1, sx, sy)
#     _scale_boxes_inplace(res2, sx, sy)

#     processor = Process(
#         res1, res2,
#         frame_height=h,
#         w_model=w_model, w_logic=w_logic,
#         final_thresh=final_thresh, roi_thresh=roi_thresh
#     )
#     processor.apply_ensemble_logic()
#     hat_detections, uniform_detections = processor.get_final_detections()

#     roi_y = int(h * roi_thresh)
#     cv2.line(frame, (0, roi_y), (w, roi_y), (0, 165, 255), 2)

#     return draw_detections(frame, hat_detections, uniform_detections, model1, model2)


# def process_frame_optimize_2(frame, model1, model2, output, conf_thresh=0.40,
#                   w_model=0.6, w_logic=0.4, final_thresh=0.5, roi_thresh=0.3,
#                   executor=None):

#     height, width = frame.shape[:2]
#     roi_y = int(height * roi_thresh)

#     frame = np.ascontiguousarray(frame, dtype=np.uint8)

#     res1, res2 = _infer_two(
#         model1, model2, frame,
#         conf1=0.25, conf2=conf_thresh,
#         imgsz=None,          
#         executor=executor
#     )

#     processor = Process(
#         res1, res2,
#         frame_height=height,
#         w_model=w_model, w_logic=w_logic,
#         final_thresh=final_thresh, roi_thresh=roi_thresh
#     )
#     processor.apply_ensemble_logic()
#     hat_detections, uniform_detections = processor.get_final_detections()

#     cv2.line(frame, (0, roi_y), (width, roi_y), (0, 165, 255), 2)
#     cv2.putText(frame, f"ROI: y>{roi_y}", (10, roi_y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

#     return draw_detections(frame, hat_detections, uniform_detections, model1, model2)


def _xyxy_list(b):
    if hasattr(b, "tolist"):
        b = b.tolist()
    return list(map(float, b[:4]))

def process_frame_optimize_2(frame, model1, model2, output, conf_thresh=0.40,
                  w_model=0.6, w_logic=0.4, final_thresh=0.5, roi_thresh=0.3,
                  executor=None):

    height, width = frame.shape[:2]
    roi_y = int(height * roi_thresh)
    frame_draw = np.ascontiguousarray(frame, dtype=np.uint8)

    res1, res2 = _infer_two(model1, model2, frame, conf1=0.25, conf2=conf_thresh, imgsz=None, executor=executor)

    processor = Process(
        res1, res2,
        frame_height=height,
        w_model=w_model, w_logic=w_logic,
        final_thresh=final_thresh, roi_thresh=roi_thresh
    )
    processor.apply_ensemble_logic()
    hat_detections, uniform_detections = processor.get_final_detections()
    
    uni_violation_box = None
    is_uni_violation = False
    
    if uniform_detections is not None:
        for box in uniform_detections:
            if len(box) >= 6 and int(box[5]) == 0: 
                is_uni_violation = True
                uni_violation_box = box
                break 
    hat_detections, uniform_detections = processor.get_final_detections()
    
    cv2.line(frame, (0, roi_y), (width, roi_y), (0, 165, 255), 2)
    cv2.putText(
        frame,
        f"ROI: y>{roi_y}",
        (10, roi_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
    )
    
    is_uni_violation = False
    uni_violation_box = None
    for b, cls in (uniform_detections or []):
        if int(cls) == 0:  
            uni_violation_box = _xyxy_list(b) + [1.0]  
            is_uni_violation = True
            break

    is_hat_violation = False
    hat_violation_box = None
    for b, cls, final_score, logic_score in (hat_detections or []):
        if int(cls) == 1:  
            hat_violation_box = _xyxy_list(b) + [float(final_score)]  
            is_hat_violation = True
            break

    processed_img = draw_detections(frame_draw, hat_detections, uniform_detections, model1, model2)
    return processed_img, (is_uni_violation, uni_violation_box), (is_hat_violation, hat_violation_box)


# def process_frame(
#     frame,
#     model1,
#     model2,
#     output,
#     conf_thresh=0.40,
#     w_model=0.6,
#     w_logic=0.4,
#     final_thresh=0.5,
#     roi_thresh=0.3,
# ):

#     height, width = frame.shape[:2]
#     roi_y = int(height * roi_thresh)

#     frame = np.ascontiguousarray(frame, dtype=np.uint8)

#     results1 = model1(frame, conf=0.25, verbose=False)  # Hat model
#     results2 = model2(frame, conf=conf_thresh, verbose=False)  # Uniform model

#     res1 = results1[0]
#     res2 = results2[0]

#     processor = Process(
#         res1,
#         res2,
#         frame_height=height,
#         w_model=w_model,
#         w_logic=w_logic,
#         final_thresh=final_thresh,
#         roi_thresh=roi_thresh,
#     )
#     processor.apply_ensemble_logic()
#     hat_detections, uniform_detections = processor.get_final_detections()

#     cv2.line(frame, (0, roi_y), (width, roi_y), (0, 165, 255), 2)
#     cv2.putText(
#         frame,
#         f"ROI: y>{roi_y}",
#         (10, roi_y - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (0, 165, 255),
#         2,
#     )

#     frame = draw_detections(frame, hat_detections, uniform_detections, model1, model2)
#     return frame




# def process_rtsp():
#     args = parse_args()
#     model1 = YOLO(args.model1)
#     model2 = YOLO(args.model2)
    
#     rtsp = (
#         f"rtsp://{args.username}:{args.password}@{args.ip_cam}:{args.port}/{args.path}"
#     )
#     print(rtsp)
#     reader = RTSPFFmpegReader(rtsp, 1920, 1080, buffer_size=3)

#     while True:
#         ret, frame = reader.read()
#         if ret:
#             processed_frame = process_frame(
#                 frame,
#                 model1,
#                 model2,
#                 args.output,
#                 conf_thresh=args.conf,
#                 w_model=args.w_model,
#                 w_logic=args.w_logic,
#                 final_thresh=args.final_thresh,
#                 roi_thresh=args.roi_thresh,
#             )
#             cv2.imshow("frame", processed_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     reader.close()
#     cv2.destroyAllWindows()


# def process_rtsp_test():
#     args = parse_args()
#     model1 = YOLO(args.model1)
#     model2 = YOLO(args.model2)
#     rtsp = (
#         f"rtsp://{args.username}:{args.password}@{args.ip_cam}:{args.port}/{args.path}"
#     )
#     print(rtsp)
    # reader = RTSPFFmpegReader(
    #     rtsp, 
    #     1920, 1080, 
    #     use_gpu=True,
    #     buffer_size=3,
    #     transport="auto",  # <-- Quan trọng: để auto
    #     use_opencv_fallback=True,  # <-- Bật fallback
    # )
    # writer = RTSPFFmpegWriter("rtsp://localhost:8554/live", 1920, 1080, use_gpu=False, ffmpeg_bin="/usr/bin/ffmpeg")
    # writer = RTSPFFmpegWriter("rtsp://localhost:8554/live", 1920, 1080, use_gpu=True, ffmpeg_bin="/usr/bin/ffmpeg")
    # reader = RTSPFFmpegReader(
    #     rtsp, 
    #     1920, 1080, 
    #     use_gpu=True,  # <-- QUAN TRỌNG: CPU decode cho stable
    #     buffer_size=3,
    #     transport="auto",
    #     use_opencv_fallback=True,
    # )

    # writer = RTSPFFmpegWriter(
    #     "rtsp://localhost:8554/live", 
    #     1920, 1080, 
    #     use_gpu=True,  # <-- GPU encode OK
    #     gpu_id=0,
    # )
    # frame_count = 0
    # print("Processing frames... Press Ctrl+C to stop")
    # # cap = cv2.VideoCapture(rtsp)
    # try:
    #     while True:
    #         ret, frame = reader.read()
    #         if ret and frame is not None:
    #             processed_frame = process_frame(
    #                 frame,
    #                 model1,
    #                 model2,
    #                 args.output,
    #                 conf_thresh=args.conf,
    #                 w_model=args.w_model,
    #                 w_logic=args.w_logic,
    #                 final_thresh=args.final_thresh,
    #                 roi_thresh=args.roi_thresh,
    #             )
    #             writer.write(processed_frame)
    #             frame_count += 1
                
    #             if frame_count % 100 == 0:
    #                 print(f"Processed {frame_count} frames")
                    
    # except KeyboardInterrupt:
    #     print("\nStopping...")
    # finally:
    #     reader.close()
    #     print(f"Total processed: {frame_count} frames")


# def process_rtsp_test_ac():
#     args = parse_args()
#     model1 = YOLO(args.model1)
#     model2 = YOLO(args.model2)

#     rtsp = f"rtsp://{args.username}:{args.password}@{args.ip_cam}:{args.port}/{args.path}"
#     print(rtsp)

#     reader = RTSPFFmpegReader(rtsp, args.w, args.h, use_gpu=True, transport="auto")
#     writer = RTSPFFmpegWriter(
#         "rtsp://127.0.0.1:8554/live", 
#         1920, 1080, 
#         use_gpu=True,  
#         gpu_id=0,
#     )
#     executor = ThreadPoolExecutor(max_workers=2)  
#     frame_count = 0

#     import time
#     start = time.time()

#     try:
#         while True:
#             ret, frame = reader.read()
#             if not ret or frame is None:
#                 continue

#             processed = process_frame_optimize_2(
#                 frame, model1, model2, args.output,
#                 conf_thresh=args.conf,
#                 w_model=args.w_model,
#                 w_logic=args.w_logic,
#                 final_thresh=args.final_thresh,
#                 roi_thresh=args.roi_thresh,
#                 executor=executor
#             )
#             writer.write(processed)

#             frame_count += 1
#             if frame_count % 100 == 0:
#                 fps = frame_count / max(1e-6, (time.time() - start))
#                 print(f"Processed {frame_count} frames | Overall FPS: {fps:.1f}")

#     except KeyboardInterrupt:
#         pass
#     finally:
#         reader.close()
#         executor.shutdown(wait=True)
#         print(f"Total processed: {frame_count} frames")

def process_rtsp_test_ac():
    args = parse_args()
    model1 = YOLO(args.model1) 
    model2 = YOLO(args.model2) 
    producer = None 
    executor = ThreadPoolExecutor(max_workers=4) 

    cnt_uniform = 0
    last_time_uniform = 0
    cnt_hat = 0
    last_time_hat = 0
    THRESH_FRAME = 10
    COOLDOWN = 3
    rtsp = f"rtsp://{args.username}:{args.password}@{args.ip_cam}:{args.port}/{args.path}"
    # reader = RTSPFFmpegReader(rtsp, args.w, args.h, use_gpu=True, transport="auto")
    writer = RTSPFFmpegWriter(
        "rtsp://127.0.0.1:8554/live", 
        1920, 1080, 
        use_gpu=True,  
        gpu_id=0,
    )
    frame_count = 0
    start = time.time()
    reader = cv2.VideoCapture(rtsp)
    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                continue

            processed, uni_info, hat_info = process_frame_optimize_2(
                frame, model1, model2, args.output,
                conf_thresh=args.conf,
                executor=executor
            )
            
            curr_t = time.time()

            is_uni_bad, uni_box = uni_info
            if is_uni_bad:
                cnt_uniform += 1
            else:
                cnt_uniform = 0
            
            if cnt_uniform >= THRESH_FRAME:
                if (curr_t - last_time_uniform) > COOLDOWN:
                    frame_copy = frame.copy()
                    executor.submit(send_violation_event, frame_copy, uni_box, "uniform", "CAM_01", producer)
                    last_time_uniform = curr_t
                    cnt_uniform = 0 

            is_hat_bad, hat_box = hat_info
            if is_hat_bad:
                cnt_hat += 1
            else:
                cnt_hat = 0
            
            if cnt_hat >= THRESH_FRAME:
                if (curr_t - last_time_hat) > COOLDOWN:
                    frame_copy = frame.copy() 
                    executor.submit(send_violation_event, frame_copy, hat_box, "hat", "CAM_01", producer)
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
        reader.close()
        executor.shutdown(wait=True)

 
if __name__ == "__main__":
    process_rtsp_test_ac()
