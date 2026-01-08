# utils.py - HARDENED VERSION for problematic RTSP cameras
import os
import time
import threading
import queue
import subprocess
import socket
from collections import deque
from typing import Optional, Tuple

import numpy as np
import cv2


class _StderrTail:
    """Continuously drain stderr to avoid pipe deadlock; keep last N lines."""
    def __init__(self, max_lines: int = 200):
        self._buf = deque(maxlen=max_lines)
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self, pipe):
        self._buf.clear()
        self._stop.clear()

        def _run():
            try:
                while not self._stop.is_set():
                    line = pipe.readline()
                    if not line:
                        break
                    self._buf.append(line.decode("utf-8", errors="ignore").rstrip("\n"))
            except Exception:
                pass

        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.3)
        self._t = None

    def tail(self, max_chars: int = 2000) -> str:
        s = "\n".join(self._buf).strip()
        if len(s) > max_chars:
            s = s[-max_chars:]
        return s


def _probe_rtsp_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    """Quick TCP probe to see if camera responds to RTSP OPTIONS."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        
        # Send RTSP OPTIONS
        req = f"OPTIONS * RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: probe\r\n\r\n"
        s.sendall(req.encode())
        
        # Try to receive response
        data = s.recv(4096)
        s.close()
        
        # Check if we got ANY response
        return len(data) > 0
    except Exception as e:
        return False


class RTSPFFmpegReader:
    """
    HARDENED RTSP reader for problematic cameras.
    
    Key changes:
    - Tries UDP first (many cheap cameras only work with UDP)
    - More tolerant FFmpeg flags (fflags, flags)
    - Fallback to OpenCV if FFmpeg completely fails
    - Socket probe before attempting FFmpeg
    """
    def __init__(
        self,
        rtsp: str,
        w: int,
        h: int,
        use_gpu: bool = False,
        gpu_id: int = 0,
        transport: str = "auto",  # auto, tcp, udp, http
        first_frame_timeout_s: float = 20.0,  # Increased for slow cameras
        stall_timeout_s: float = 8.0,
        log: bool = True,
        buffer_size: int = 10,
        probesize: int = 10_000_000,  # Increased - some cameras need more
        analyzeduration: int = 10_000_000,
        max_delay: int = 500000,  # 500ms max delay
        ffmpeg_bin: Optional[str] = None,
        debug_cmd: bool = True,
        restart_backoff_base_s: float = 1.0,
        restart_backoff_max_s: float = 10.0,
        use_opencv_fallback: bool = True,  # NEW: fallback to OpenCV
    ):
        self.rtsp = rtsp
        self.w, self.h = int(w), int(h)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id)
        self.transport = transport

        self.probesize = int(probesize)
        self.analyzeduration = int(analyzeduration)
        self.max_delay = int(max_delay)

        self.first_frame_timeout_s = float(first_frame_timeout_s)
        self.stall_timeout_s = float(stall_timeout_s)
        self.log = bool(log)
        self.debug_cmd = bool(debug_cmd)
        self.use_opencv_fallback = bool(use_opencv_fallback)

        self.frame_bytes = self.w * self.h * 3
        self.q = queue.Queue(maxsize=int(buffer_size))

        self.proc: Optional[subprocess.Popen] = None
        self._stderr = _StderrTail(max_lines=200)

        self._stop = threading.Event()
        self._worker_t: Optional[threading.Thread] = None

        # state/metrics
        self.start_ts = 0.0
        self.last_frame_ts = 0.0
        self.frame_count = 0
        self.drop_count = 0
        self.fps_history = deque(maxlen=60)
        self.last_fps_log = time.time()

        # restart control
        self._fail_count = 0
        self._backoff_base = float(restart_backoff_base_s)
        self._backoff_max = float(restart_backoff_max_s)
        self._token = 0
        
        # OpenCV fallback
        self._opencv_mode = False
        self._cap: Optional[cv2.VideoCapture] = None

        # Transport selection
        self._current_transport = self._select_transport()

        if ffmpeg_bin:
            self.ffmpeg_bin = ffmpeg_bin
        else:
            self.ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"

        # Probe camera first
        if self.log:
            print("[Reader] Probing camera...")
            host, port = self._parse_host_port()
            if host and port:
                can_connect = _probe_rtsp_tcp(host, port, timeout=3.0)
                if can_connect:
                    print(f"[Reader] Camera at {host}:{port} responds to TCP")
                else:
                    print(f"[Reader] WARNING: Camera at {host}:{port} not responding to TCP probe")

        self._start(initial=True)

    def _parse_host_port(self) -> Tuple[Optional[str], Optional[int]]:
        """Extract host and port from RTSP URL."""
        try:
            # rtsp://user:pass@host:port/path
            import re
            match = re.search(r'rtsp://(?:[^@]+@)?([^:/]+):(\d+)', self.rtsp)
            if match:
                return match.group(1), int(match.group(2))
        except Exception:
            pass
        return None, None

    def _select_transport(self) -> str:
        """Select best transport based on user preference and testing."""
        if self.transport == "auto":
            # Try UDP first for problematic cameras (common in Vietnam)
            if self.log:
                print("[Reader] Auto-selecting transport: trying UDP first")
            return "udp"
        return self.transport

    def _cmd(self) -> list:
        """Build FFmpeg command with MAXIMUM tolerance flags."""
        cmd = [
            self.ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel", "info" if self.log else "error",  # Changed to info for debugging
        ]

        # CRITICAL: Flags for accepting broken/non-standard streams
        cmd += [
            "-fflags", "+genpts+igndts+discardcorrupt",  # Generate PTS, ignore bad timestamps
            "-flags", "low_delay",  # Low latency mode
            "-strict", "experimental",  # Allow experimental features
            "-err_detect", "ignore_err",  # Ignore errors in stream
        ]

        # Transport
        cmd += ["-rtsp_transport", self._current_transport]

        # Probe settings - CRITICAL for problem cameras
        cmd += [
            "-probesize", str(self.probesize),
            "-analyzeduration", str(self.analyzeduration),
            "-max_delay", str(self.max_delay),
        ]

        # NO timeout flags - they cause "Invalid data" errors
        # Camera will timeout naturally if dead

        # CRITICAL FIX: DO NOT USE GPU DECODE for this problematic camera
        # It causes memory corruption with non-standard streams
        # GPU decode benefit is minimal for inference workload anyway
        
        cmd += [
            "-i", self.rtsp,
            "-map", "0:v:0",
            "-an", "-sn",
            "-vf", "format=bgr24",
        ]

        cmd += [
            "-fps_mode", "passthrough",
            "-c:v", "rawvideo",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        return cmd

    def _drain_q(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Exception:
                break

    def _kill_proc(self):
        self._stderr.stop()
        p = self.proc
        self.proc = None
        if not p:
            return
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=0.7)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            if p.stdout:
                p.stdout.close()
            if p.stderr:
                p.stderr.close()
        except Exception:
            pass

    def _backoff_sleep(self):
        if self._fail_count <= 0:
            return
        delay = min(self._backoff_max, self._backoff_base * self._fail_count)
        if self.log:
            print(f"[Reader] Backoff {delay:.1f}s (fail_count={self._fail_count})")
        time.sleep(delay)

    def _try_opencv_fallback(self) -> bool:
        """Try OpenCV as last resort."""
        if not self.use_opencv_fallback or self._opencv_mode:
            return False

        if self.log:
            print("[Reader] FFmpeg failed repeatedly, trying OpenCV fallback...")

        try:
            # OpenCV with environment variable set
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|tcp_nodelay;1"
            self._cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            # Test read
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._opencv_mode = True
                if self.log:
                    print("[Reader] OpenCV fallback SUCCESS")
                return True
            else:
                self._cap.release()
                self._cap = None
                return False
        except Exception as e:
            if self.log:
                print(f"[Reader] OpenCV fallback failed: {e}")
            if self._cap:
                self._cap.release()
                self._cap = None
            return False

    def _start(self, initial: bool = False):
        if not initial:
            self._fail_count = min(self._fail_count + 1, 999)
            
            # After 3 failures with current transport, try switching
            if self._fail_count == 3 and self.transport == "auto":
                if self._current_transport == "udp":
                    self._current_transport = "tcp"
                    if self.log:
                        print("[Reader] Switching from UDP to TCP")
                elif self._current_transport == "tcp":
                    self._current_transport = "udp_multicast"
                    if self.log:
                        print("[Reader] Switching to UDP multicast")
            
            # After 6 failures, try OpenCV
            if self._fail_count >= 6:
                if self._try_opencv_fallback():
                    self._fail_count = 0
                    self.start_ts = time.time()
                    self.last_frame_ts = 0.0
                    self.frame_count = 0
                    
                    self._token += 1
                    token = self._token
                    self._worker_t = threading.Thread(target=self._worker_opencv, args=(token,), daemon=True)
                    self._worker_t.start()
                    return
            
            self._backoff_sleep()
        else:
            self._fail_count = 0

        self._opencv_mode = False
        self._kill_proc()
        self._drain_q()

        cmd = self._cmd()
        if self.log:
            mode = f"GPU {self.gpu_id}" if self.use_gpu else "CPU"
            print(f"[Reader] {mode} decode, transport={self._current_transport}")
            if self.debug_cmd:
                print("[Reader] ffmpeg cmd:", " ".join(cmd))

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
        )
        if self.proc.stderr:
            self._stderr.start(self.proc.stderr)

        self.start_ts = time.time()
        self.last_frame_ts = 0.0
        self.frame_count = 0
        self.drop_count = 0
        self.fps_history.clear()
        self.last_fps_log = time.time()

        self._token += 1
        token = self._token
        self._worker_t = threading.Thread(target=self._worker, args=(token,), daemon=True)
        self._worker_t.start()

        if self.log and not initial:
            mode = f"GPU{self.gpu_id}" if self.use_gpu else "CPU"
            print(f"[Reader {mode}] Restarted")

    def _worker(self, token: int):
        """FFmpeg worker thread."""
        p = self.proc
        if not p or not p.stdout:
            return

        buffer = bytearray()
        read_chunk = max(self.frame_bytes, 1 << 20)

        while not self._stop.is_set() and token == self._token:
            try:
                chunk = p.stdout.read(read_chunk)
                if not chunk:
                    break
                buffer.extend(chunk)

                while len(buffer) >= self.frame_bytes:
                    raw = bytes(buffer[:self.frame_bytes])
                    del buffer[:self.frame_bytes]

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3)).copy()
                    now = time.time()
                    self.last_frame_ts = now
                    self.frame_count += 1
                    self.fps_history.append(now)

                    dropped = 0
                    while self.q.full():
                        try:
                            self.q.get_nowait()
                            dropped += 1
                        except Exception:
                            break
                    if dropped:
                        self.drop_count += dropped

                    try:
                        self.q.put_nowait(frame)
                    except queue.Full:
                        self.drop_count += 1
            except Exception:
                break

    def _worker_opencv(self, token: int):
        """OpenCV fallback worker thread."""
        cap = self._cap
        if not cap:
            return

        while not self._stop.is_set() and token == self._token:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Resize if needed
                if frame.shape[1] != self.w or frame.shape[0] != self.h:
                    frame = cv2.resize(frame, (self.w, self.h))

                now = time.time()
                self.last_frame_ts = now
                self.frame_count += 1
                self.fps_history.append(now)

                dropped = 0
                while self.q.full():
                    try:
                        self.q.get_nowait()
                        dropped += 1
                    except Exception:
                        break
                if dropped:
                    self.drop_count += dropped

                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    self.drop_count += 1

            except Exception as e:
                if self.log:
                    print(f"[Reader OpenCV] Error: {e}")
                break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame = None
        try:
            frame = self.q.get(timeout=0.05)
        except queue.Empty:
            frame = None

        if frame is not None:
            while True:
                try:
                    frame = self.q.get_nowait()
                except queue.Empty:
                    break

            self._fail_count = 0

            now = time.time()
            if self.log and now - self.last_fps_log > 3.0 and len(self.fps_history) > 1:
                elapsed = self.fps_history[-1] - self.fps_history[0]
                fps = (len(self.fps_history) - 1) / elapsed if elapsed > 0 else 0.0
                drop_rate = self.drop_count / max(1, self.frame_count) * 100.0
                mode_str = "OpenCV" if self._opencv_mode else (f"GPU{self.gpu_id}" if self.use_gpu else "CPU")
                print(f"[Reader {mode_str}] FPS: {fps:.1f} | Drop: {drop_rate:.1f}%")
                self.last_fps_log = now

            return True, frame

        now = time.time()

        # OpenCV mode checks
        if self._opencv_mode:
            if self.last_frame_ts == 0.0:
                if (now - self.start_ts) > self.first_frame_timeout_s:
                    if self.log:
                        print("[Reader OpenCV] No first frame -> restart")
                    self._opencv_mode = False
                    if self._cap:
                        self._cap.release()
                        self._cap = None
                    self._start(initial=False)
                return False, None

            if (now - self.last_frame_ts) > self.stall_timeout_s:
                if self.log:
                    print("[Reader OpenCV] Stalled -> restart")
                self._opencv_mode = False
                if self._cap:
                    self._cap.release()
                    self._cap = None
                self._start(initial=False)
            return False, None

        # FFmpeg mode checks
        if self.proc and self.proc.poll() is not None:
            if self.log:
                tail = self._stderr.tail()
                if tail:
                    print(f"[Reader] ffmpeg stderr:\n{tail}")
            self._start(initial=False)
            return False, None

        if self.last_frame_ts == 0.0:
            if (now - self.start_ts) > self.first_frame_timeout_s:
                if self.log:
                    print("[Reader] No first frame -> restart")
                    tail = self._stderr.tail()
                    if tail:
                        print(f"[Reader] ffmpeg stderr:\n{tail}")
                self._start(initial=False)
            return False, None

        if (now - self.last_frame_ts) > self.stall_timeout_s:
            if self.log:
                print("[Reader] Stalled -> restart")
                tail = self._stderr.tail()
                if tail:
                    print(f"[Reader] ffmpeg stderr:\n{tail}")
            self._start(initial=False)

        return False, None

    def close(self):
        self._stop.set()
        if self._worker_t:
            self._worker_t.join(timeout=1.0)
        
        if self._opencv_mode and self._cap:
            self._cap.release()
            self._cap = None
        
        self._kill_proc()
        self._drain_q()


class RTSPFFmpegWriter:
    """
    RTSP writer via FFmpeg - same as before but with better error handling.
    """
    def __init__(
        self,
        rtsp_url: str,
        width: int,
        height: int,
        fps: int = 25,
        use_gpu: bool = False,
        gpu_id: int = 0,
        queue_size: int = 30,
        log: bool = True,
        ffmpeg_bin: Optional[str] = None,
        debug_cmd: bool = True,
        restart_backoff_base_s: float = 0.5,
        restart_backoff_max_s: float = 8.0,
    ):
        self.rtsp_url = rtsp_url
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id)
        self.log = bool(log)
        self.debug_cmd = bool(debug_cmd)

        if ffmpeg_bin:
            self.ffmpeg_bin = ffmpeg_bin
        else:
            self.ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"

        self.q = queue.Queue(maxsize=int(queue_size))
        self._stop = threading.Event()
        self.process: Optional[subprocess.Popen] = None
        self._stderr = _StderrTail(max_lines=200)

        self._fail_count = 0
        self._backoff_base = float(restart_backoff_base_s)
        self._backoff_max = float(restart_backoff_max_s)

        self.write_count = 0
        self.drop_count = 0
        self.fps_history = deque(maxlen=60)
        self.last_fps_log = time.time()

        self._start()
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _cmd(self) -> list:
        cmd = [
            self.ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel", "warning" if self.log else "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
        ]

        if self.use_gpu:
            # FIXED: h264_nvenc doesn't use -gpu flag, uses environment variable instead
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", "p1",  # p1 = fastest, p7 = slowest
                "-tune", "ll",  # low latency
                "-b:v", "2M",
                "-maxrate", "2M",
                "-bufsize", "4M",
            ]
        else:
            cmd += [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-crf", "23",
            ]

        cmd += [
            "-pix_fmt", "yuv420p",
            "-g", "50",  # GOP size
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]
        return cmd

    def _kill_proc(self):
        self._stderr.stop()
        p = self.process
        self.process = None
        if not p:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=0.7)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            if p.stderr:
                p.stderr.close()
        except Exception:
            pass

    def _backoff_sleep(self):
        if self._fail_count <= 0:
            return
        delay = min(self._backoff_max, self._backoff_base * self._fail_count)
        if self.log:
            print(f"[Writer] Backoff {delay:.1f}s (fail_count={self._fail_count})")
        time.sleep(delay)

    def _start(self):
        if self.process:
            self._fail_count = min(self._fail_count + 1, 999)
            self._backoff_sleep()
        else:
            self._fail_count = 0

        self._kill_proc()
        cmd = self._cmd()

        if self.log:
            mode = f"GPU {self.gpu_id}" if self.use_gpu else "CPU"
            print(f"[Writer] {mode} encode")
            if self.debug_cmd:
                print("[Writer] ffmpeg cmd:", " ".join(cmd))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
        )
        if self.process.stderr:
            self._stderr.start(self.process.stderr)

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            p = self.process
            if not p or not p.stdin or p.poll() is not None:
                if self.log:
                    tail = self._stderr.tail()
                    if tail:
                        print("[Writer] ffmpeg stderr:\n" + tail)
                    print("[Writer] Process died, restarting...")
                self._start()
                continue

            try:
                p.stdin.write(frame.tobytes())
            except BrokenPipeError:
                if self.log:
                    tail = self._stderr.tail()
                    if tail:
                        print("[Writer] ffmpeg stderr:\n" + tail)
                    print("[Writer] Broken pipe, restarting...")
                self._start()
                continue
            except Exception as e:
                if self.log:
                    print(f"[Writer] Write error: {e}")
                time.sleep(0.05)
                continue

            now = time.time()
            self.write_count += 1
            self.fps_history.append(now)

            if self.log and now - self.last_fps_log > 3.0 and len(self.fps_history) > 1:
                elapsed = self.fps_history[-1] - self.fps_history[0]
                fps = (len(self.fps_history) - 1) / elapsed if elapsed > 0 else 0.0
                drop_rate = self.drop_count / max(1, self.write_count) * 100.0
                mode = f"GPU{self.gpu_id}" if self.use_gpu else "CPU"
                print(f"[Writer {mode}] FPS: {fps:.1f} | Drop: {drop_rate:.1f}%")
                self.last_fps_log = now

    def write(self, frame: np.ndarray):
        try:
            if frame is None:
                return
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            if self.q.full():
                try:
                    self.q.get_nowait()
                    self.drop_count += 1
                except Exception:
                    pass

            self.q.put_nowait(frame)
        except Exception:
            self.drop_count += 1

    def close(self):
        self._stop.set()
        if hasattr(self, "t") and self.t:
            self.t.join(timeout=1.0)
        self._kill_proc()