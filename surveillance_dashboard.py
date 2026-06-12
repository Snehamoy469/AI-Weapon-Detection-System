import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import threading
import time
import os, sys, io, json, csv, queue, math
import urllib.request
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime, timedelta
import collections

# ── GPU / torch optional ────────────────────────────────────────────────────
try:
    import torch
    _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _TORCH_DEVICE = "cpu"

# ── NumPy Gaussian kernel for heatmap (replaces nested loops) ───────────────
def _make_gaussian_kernel(radius=6):
    size  = radius * 2 + 1
    ax    = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k     = np.exp(-(xx**2 + yy**2) / 8.0)
    return k

_GAUSS_KERNEL = _make_gaussian_kernel(6)   # pre-built once at import time

# ── optional voice TTS ───────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 165)
    _tts_engine.setProperty("volume", 1.0)
    voices = _tts_engine.getProperty("voices")
    for v in voices:
        if "english" in v.name.lower() or "en" in v.id.lower():
            _tts_engine.setProperty("voice", v.id)
            break
    HAS_TTS = True
except Exception:
    HAS_TTS = False

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ─────────────────────────────────────────────────────────────────────────────
# THEME  —  Cornflower Blue Industrial
# ─────────────────────────────────────────────────────────────────────────────
CORNFLOWER   = "#6495ED"
CORNFLOWER_D = "#4169C8"
CORNFLOWER_L = "#8AB4F8"
CORNFLOWER_G = "#2A3F6E"

BG_BASE   = "#0A0D14"
BG_PANEL  = "#0F1522"
BG_CARD   = "#141B2D"
BG_CARD2  = "#1A2238"
BG_BORDER = "#1E2D50"

TEXT_PRIMARY = "#E8EEFF"
TEXT_SEC     = "#8A9BC4"
TEXT_DIM     = "#4A5578"

ACCENT_RED   = "#FF4444"
ACCENT_AMBER = "#FFB347"
ACCENT_GREEN = "#44EE88"

F_TINY  = 10
F_SMALL = 11
F_MID   = 13
F_BODY  = 14
F_LG    = 16
F_XL    = 20
F_TITLE = 23


def threat_color(score):
    if score >= 75: return ACCENT_RED
    if score >= 40: return ACCENT_AMBER
    if score >= 10: return CORNFLOWER_L
    return ACCENT_GREEN


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def scan_models_folder():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".pt"))


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REDIRECT
# ─────────────────────────────────────────────────────────────────────────────
class TerminalRedirect(io.StringIO):
    def __init__(self, cb):
        super().__init__()
        self._cb = cb
    def write(self, text):
        if text.strip():
            self._cb(text)
    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GEOLOCATION
# ─────────────────────────────────────────────────────────────────────────────
def fetch_geolocation():
    try:
        url = ("http://ip-api.com/json/?fields=status,message,"
               "country,regionName,city,lat,lon,isp,query")
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
            if data.get("status") == "success":
                return data
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ALARM ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AlarmEngine:
    def __init__(self):
        self.enabled       = True
        self.triggered     = False
        self.trigger_count = 0
        self.last_trigger  = None
        self.cooldown_sec  = 6
        self._tts_lock     = threading.Lock()

    def trigger(self, callback_flash, classes=None):
        if not self.enabled:
            return
        now = time.time()
        if self.last_trigger and (now - self.last_trigger) < self.cooldown_sec:
            return
        self.last_trigger   = now
        self.trigger_count += 1
        self.triggered      = True
        callback_flash()
        cl = classes or []
        threading.Thread(target=self._speak, args=(cl,), daemon=True).start()

    def _speak(self, classes):
        obj_str = ", ".join(set(classes)) if classes else "object"
        msg     = f"Alert! {obj_str} detected. Threat level elevated."
        try:
            if HAS_TTS:
                with self._tts_lock:
                    _tts_engine.say(msg)
                    _tts_engine.runAndWait()
                return
        except Exception:
            pass
        try:
            if HAS_WINSOUND:
                for _ in range(4):
                    winsound.Beep(1100, 200)
                    time.sleep(0.08)
                    winsound.Beep(800, 200)
                    time.sleep(0.08)
            else:
                for _ in range(4):
                    os.system("printf '\\a'")
                    time.sleep(0.3)
        except Exception:
            pass

    def reset(self):
        self.triggered = False


# ─────────────────────────────────────────────────────────────────────────────
# LOG ANALYSIS ENGINE  — [FIX-2] bounded records list
# ─────────────────────────────────────────────────────────────────────────────
class LogAnalysisEngine:
    """
    records is now capped at 10 000 entries; oldest half is pruned when full.
    """
    MAX_RECORDS = 10_000
    PRUNE_TO    = 5_000

    def __init__(self):
        self.records = []
        self.hourly  = collections.defaultdict(int)

    def record(self, classes, confs):
        now = datetime.now()
        self.records.append({"ts": now, "classes": classes,
                              "confs": confs, "n": len(classes)})
        self.hourly[now.hour] += len(classes)
        # [FIX-2] Prune to avoid unbounded memory growth
        if len(self.records) > self.MAX_RECORDS:
            self.records = self.records[-self.PRUNE_TO:]

    def threat_score(self):
        if not self.records:
            return 0
        cutoff = datetime.now() - timedelta(seconds=30)
        recent = [r for r in self.records if r["ts"] >= cutoff]
        if not recent:
            return 0
        total    = sum(r["n"] for r in recent)
        avg_conf = (sum(c for r in recent for c in r["confs"]) /
                    max(sum(r["n"] for r in recent), 1))
        burst = min(total / 5.0, 1.0)
        return min(int(burst * 60 + avg_conf * 40), 100)

    def hourly_bars(self):
        return [(f"{h:02d}", self.hourly.get(h, 0)) for h in range(24)]

    def export_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "n_detections", "classes", "avg_conf"])
            for r in self.records:
                avg_c = sum(r["confs"]) / len(r["confs"]) if r["confs"] else 0
                w.writerow([r["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                             r["n"], "|".join(r["classes"]),
                             f"{avg_c:.3f}"])


# ─────────────────────────────────────────────────────────────────────────────
# MOTION HEATMAP  — [PERF-2] NumPy Gaussian kernel, ~10× faster
# ─────────────────────────────────────────────────────────────────────────────
class MotionHeatmap:
    def __init__(self, w=160, h=120):
        self.W   = w
        self.H   = h
        self.map = np.zeros((h, w), dtype=np.float32)

    def add(self, boxes_xyxyn):
        self.map *= 0.99
        r  = len(_GAUSS_KERNEL) // 2   # kernel radius (6)
        for b in boxes_xyxyn:
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            px = int(cx * self.W)
            py = int(cy * self.H)
            # Compute the valid overlap region between kernel and heatmap
            x0k = max(0, r - px);       x1k = r + min(self.W - px, r + 1)
            y0k = max(0, r - py);       y1k = r + min(self.H - py, r + 1)
            x0m = max(0, px - r);       x1m = min(self.W, px + r + 1)
            y0m = max(0, py - r);       y1m = min(self.H, py + r + 1)
            if x1m > x0m and y1m > y0m:
                self.map[y0m:y1m, x0m:x1m] += _GAUSS_KERNEL[y0k:y1k, x0k:x1k]

    def render(self, size=(300, 100)):
        mn, mx = self.map.min(), self.map.max()
        norm = ((self.map - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        rgb  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb).resize(size, Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        for x in range(0, size[0], 40):
            draw.line([(x, 0), (x, size[1])], fill=(255, 255, 255, 25), width=1)
        for y in range(0, size[1], 30):
            draw.line([(0, y), (size[0], y)], fill=(255, 255, 255, 25), width=1)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def cv2_to_pil(frame):
    """Convert CV2 frame to PIL - no mirroring (frame should be pre-mirrored if needed)."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def cv2_to_pil_noflip(frame):
    """Convert CV2 frame to PIL - video/image files (no mirroring)."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ═════════════════════════════════════════════════════════════════════════════
# LIGHTBOX
# ═════════════════════════════════════════════════════════════════════════════
class Lightbox(ctk.CTkToplevel):
    def __init__(self, master, pil_images, start_index=0):
        super().__init__(master)
        self.title("VIGI-SHIELD  ·  Detection Gallery")
        self.geometry("1100x700")
        self.configure(fg_color=BG_BASE)
        self.grab_set()

        self._images = pil_images
        self._idx    = start_index
        self._ctk_img = None

        top = ctk.CTkFrame(self, fg_color=BG_PANEL, height=44, corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)
        ctk.CTkLabel(top, text="◈  DETECTION  GALLERY  VIEWER",
                     font=("Courier New", F_MID, "bold"),
                     text_color=CORNFLOWER).pack(side="left", padx=16)
        ctk.CTkButton(top, text="✕  CLOSE", width=90, height=30,
                      fg_color=ACCENT_RED, hover_color="#CC0000",
                      font=("Courier New", F_SMALL, "bold"),
                      command=self.destroy).pack(side="right", padx=12)

        self.img_label = ctk.CTkLabel(self, text="", fg_color=BG_BASE)
        self.img_label.pack(fill="both", expand=True, padx=10, pady=6)

        self.meta_lbl = ctk.CTkLabel(self, text="",
                                     font=("Courier New", F_SMALL),
                                     text_color=TEXT_SEC, fg_color=BG_CARD)
        self.meta_lbl.pack(fill="x")

        nav = ctk.CTkFrame(self, fg_color=BG_PANEL, height=46, corner_radius=0)
        nav.pack(fill="x")
        nav.pack_propagate(False)
        ctk.CTkButton(nav, text="◀  PREV", width=110, height=32,
                      fg_color=CORNFLOWER_D, font=("Courier New", F_SMALL, "bold"),
                      command=self._prev).pack(side="left", padx=14, pady=6)
        self.counter_lbl = ctk.CTkLabel(nav, text="",
                                        font=("Courier New", F_MID, "bold"),
                                        text_color=CORNFLOWER_L)
        self.counter_lbl.pack(side="left", expand=True)
        ctk.CTkButton(nav, text="NEXT  ▶", width=110, height=32,
                      fg_color=CORNFLOWER_D, font=("Courier New", F_SMALL, "bold"),
                      command=self._next).pack(side="right", padx=14, pady=6)

        self.bind("<Left>",   lambda _: self._prev())
        self.bind("<Right>",  lambda _: self._next())
        self.bind("<Escape>", lambda _: self.destroy())
        self.bind("<Configure>", lambda _: self._show())
        self._show()

    def _show(self):
        if not self._images:
            return
        pil_img, meta = self._images[self._idx]
        W = max(self.img_label.winfo_width(),  900)
        H = max(self.img_label.winfo_height(), 560)
        iw, ih = pil_img.size
        scale  = min(W / iw, H / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        bg = Image.new("RGB", (W, H), BG_BASE)
        bg.paste(resized, ((W - nw) // 2, (H - nh) // 2))
        self._ctk_img = ctk.CTkImage(light_image=bg, dark_image=bg, size=(W, H))
        self.img_label.configure(image=self._ctk_img, text="")
        self.meta_lbl.configure(text=f"  {meta}")
        self.counter_lbl.configure(text=f"  {self._idx+1}  /  {len(self._images)}")

    def _prev(self):
        self._idx = (self._idx - 1) % len(self._images)
        self._show()

    def _next(self):
        self._idx = (self._idx + 1) % len(self._images)
        self._show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class VigiShieldApp(ctk.CTk):

    # ── Configurable performance knob ──────────────────────────────────────
    INFERENCE_EVERY_N_FRAMES = 1    # [PERF-3] set to 2 to skip every other frame
    THUMB_EVERY_N_DETECTIONS = 10   # [PERF-4] rate-limit thumbnail creation

    def __init__(self):
        super().__init__()
        self.title("VIGILANT EYE ·  Industrial Surveillance Platform")
        self.geometry("1820x1040")
        self.minsize(1400, 860)
        self.configure(fg_color=BG_BASE)
        ctk.set_appearance_mode("dark")

        # ── Core state ────────────────────────────────────────────────────
        self.model           = None
        self.cap             = None
        self.running         = False
        self.paused          = False
        self.is_video_file   = False
        self.frame_count     = 0
        self.fps_val         = 0.0
        self.confidence_threshold = 0.25  # Minimum confidence for detections
        self.fps_history     = collections.deque(maxlen=80)
        self.det_history     = collections.deque(maxlen=80)
        self.class_counts    = {}
        self.confidence_vals = collections.deque(maxlen=300)
        self.session_start   = None
        self.log_queue       = queue.Queue()
        # [FIX-2] bounded event list
        self._all_events     = collections.deque(maxlen=5_000)
        self.det_thumb_row   = 0
        self.det_thumb_col   = 0
        self._det_frame_ctr  = 0   # [PERF-4] counts detection frames for thumb throttle
        self._gallery        = collections.deque(maxlen=100)
        self._frame_queue    = queue.Queue(maxsize=2)

        # [FIX-3] threading.Event for zero-CPU pause
        self._pause_event    = threading.Event()
        self._pause_event.set()   # set = running, clear = paused

        # ── Sub-systems ───────────────────────────────────────────────────
        self.alarm   = AlarmEngine()
        self.log_eng = LogAnalysisEngine()
        self.heatmap = MotionHeatmap()

        # ── Geo ───────────────────────────────────────────────────────────
        self.geo = {"city": "Fetching…", "regionName": "", "country": "",
                    "lat": 0.0, "lon": 0.0, "isp": "—", "query": "—"}
        threading.Thread(target=self._fetch_geo, daemon=True).start()

        # ── Alarm flash ───────────────────────────────────────────────────
        self._alarm_flash_count = 0
        self._alarm_flashing    = False

        self._build_ui()
        self._redirect_terminal()
        self._schedule_ui_updates()
        self._schedule_feed_render()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        device_info = f"GPU ({_TORCH_DEVICE.upper()})" if _TORCH_DEVICE == "cuda" else "CPU"
        self._log(f"VIGILANT EYE ready.  Device: {device_info}", "good")
        self._log("Select model → LOAD → Camera / Video / Image.", "info")
        if HAS_TTS:
            self._log("Voice alarm engine: ACTIVE (pyttsx3)", "good")
        else:
            self._log("pyttsx3 not found — falling back to beep alarm.", "warn")

    # ═════════════════════════════════════════════════════════════════════
    # GEOLOCATION
    # ═════════════════════════════════════════════════════════════════════
    def _fetch_geo(self):
        data = fetch_geolocation()
        if data:
            self.geo = data
            self._log(f"Geo resolved: {data.get('city')}, "
                      f"{data.get('regionName')}, {data.get('country')} "
                      f"[{data.get('lat'):.4f}, {data.get('lon'):.4f}]", "good")
        else:
            self.geo["city"] = "Unavailable"
            self._log("Geolocation unavailable (offline?).", "warn")
        self.after(0, self._refresh_geo_panel)

    def _refresh_geo_panel(self):
        g = self.geo
        self.lbl_geo_city.configure(
            text=f"{g.get('city','—')}, {g.get('regionName','—')}")
        self.lbl_geo_country.configure(text=g.get("country", "—"))
        self.lbl_geo_coords.configure(
            text=f"{g.get('lat',0):.4f}°N  {g.get('lon',0):.4f}°E")
        self.lbl_geo_ip.configure(text=g.get("query", "—"))
        self.lbl_geo_isp.configure(text=g.get("isp", "—"))
        self._draw_mini_map()

    def _draw_mini_map(self):
        raw_w = max(self.map_canvas.winfo_width(),  120)
        raw_h = max(self.map_canvas.winfo_height(), 120)
        side  = min(raw_w, raw_h)
        W, H  = side, side
        img  = Image.new("RGB", (W, H), "#0D1825")
        draw = ImageDraw.Draw(img)
        for x in range(0, W, 20):
            draw.line([(x, 0), (x, H)], fill="#1A2840", width=1)
        for y in range(0, H, 15):
            draw.line([(0, y), (W, y)], fill="#1A2840", width=1)
        lat = self.geo.get("lat", 0)
        lon = self.geo.get("lon", 0)
        px  = int((lon + 180) / 360 * W)
        py  = int((90 - lat)  / 180 * H)
        for radius, alpha in [(18, 40), (12, 90), (6, 160)]:
            col = (min(100 + alpha, 255), min(149 + alpha // 4, 255), 237)
            col_hex = "#{:02x}{:02x}{:02x}".format(*col)
            draw.ellipse([px-radius, py-radius, px+radius, py+radius],
                         outline=col_hex, width=1)
        draw.ellipse([px-4, py-4, px+4, py+4], fill=CORNFLOWER, outline=TEXT_PRIMARY)
        draw.text((4, 2), f"{lat:.2f}°N  {lon:.2f}°E", fill=TEXT_SEC)
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=(W, H))
        self.map_canvas.configure(image=ci)
        self.map_canvas._img = ci

    # ═════════════════════════════════════════════════════════════════════
    # UI BUILD
    # ═════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self._build_header()
        self._build_alarm_banner()
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        content.columnconfigure(0, weight=50)
        content.columnconfigure(1, weight=22)
        content.columnconfigure(2, weight=28)
        content.rowconfigure(0, weight=1)
        self._build_live_panel(content)
        self._build_center_panel(content)
        self._build_right_panel(content)

    def _build_header(self):
        # ── Outer header — auto-height, no pack_propagate(False) ─────────
        # Two rows: top = brand + status chips, bottom = model selector + buttons
        hdr = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        hdr.pack(fill="x")

        # ── Row 1: brand left, stat chips right ───────────────────────────
        row1 = ctk.CTkFrame(hdr, fg_color="transparent", height=46)
        row1.pack(fill="x", padx=8, pady=(6, 0))
        row1.pack_propagate(False)

        brand = ctk.CTkFrame(row1, fg_color="transparent")
        brand.pack(side="left")
        self.status_canvas = tk.Canvas(brand, width=14, height=14,
                                       bg=BG_PANEL, highlightthickness=0)
        self.status_canvas.pack(side="left", padx=(0, 8))
        self._draw_status_dot(ACCENT_RED)
        ctk.CTkLabel(brand, text="VIGILANT EYE",
                     font=("Courier New", F_XL, "bold"),
                     text_color=CORNFLOWER).pack(side="left")

        # Stat chips — compact, no outer padding eating space
        chips = ctk.CTkFrame(row1, fg_color="transparent")
        chips.pack(side="right")
        self.lbl_session    = self._chip(chips, "SESSION",    "—",   CORNFLOWER)
        self.lbl_fps_hdr    = self._chip(chips, "FPS",        "0",   ACCENT_GREEN)
        self.lbl_frames_hdr = self._chip(chips, "FRAMES",     "0",   CORNFLOWER_L)
        self.lbl_dets_hdr   = self._chip(chips, "DETECTIONS", "0",   ACCENT_AMBER)
        self.lbl_threat_hdr = self._chip(chips, "THREAT",     "LOW", ACCENT_GREEN)

        # Thin separator line between rows
        ctk.CTkFrame(hdr, fg_color=BG_BORDER, height=1,
                     corner_radius=0).pack(fill="x", padx=0)

        # ── Row 2: model picker left, all buttons filling remaining width ─
        row2 = ctk.CTkFrame(hdr, fg_color="transparent", height=42)
        row2.pack(fill="x", padx=8, pady=(4, 6))
        row2.pack_propagate(False)

        # Model selector + refresh button
        _model_list = scan_models_folder() or ["(no models found)"]
        self.model_var = ctk.StringVar(value=_model_list[0])
        self.model_selector = ctk.CTkOptionMenu(
            row2,
            values=_model_list,
            variable=self.model_var,
            width=148, height=32,
            fg_color=BG_CARD2, button_color=CORNFLOWER_D,
            dropdown_fg_color=BG_CARD, dropdown_hover_color=CORNFLOWER_G,
            font=("Courier New", F_SMALL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self.model_selector.pack(side="left", padx=(0, 2))
        ctk.CTkButton(row2, text="⟳", width=32, height=32,
                      fg_color=BG_CARD2, hover_color=CORNFLOWER_G,
                      font=("Courier New", F_MID, "bold"),
                      text_color=CORNFLOWER_L,
                      command=self._refresh_model_list).pack(side="left", padx=(0, 8))

        # All action buttons — smaller width/height so they all fit on row 2
        self.btn_load   = self._hdr_btn(row2, "⬆ LOAD",      self._load_model,      CORNFLOWER_G)
        self.btn_cam    = self._hdr_btn(row2, "📷 CAMERA",    self._start_camera,    CORNFLOWER_D)
        self.btn_video  = self._hdr_btn(row2, "🎬 VIDEO",     self._browse_video,    "#2D4A7A")
        self.btn_image  = self._hdr_btn(row2, "🖼 IMAGE",     self._browse_image,    "#4A3B7A")
        self.btn_pause  = self._hdr_btn(row2, "⏸ PAUSE",      self._pause_detection, BG_CARD2)
        self.btn_stop   = self._hdr_btn(row2, "■ STOP",       self._stop_detection,  ACCENT_RED)
        self.btn_export = self._hdr_btn(row2, "↓ EXPORT",     self._export_log,      CORNFLOWER_G)
        self.btn_alarm  = self._hdr_btn(row2, "🔔 ALARM ON",  self._toggle_alarm,    "#2A5030")

        self.btn_cam.configure(state="disabled")
        self.btn_video.configure(state="disabled")
        self.btn_image.configure(state="disabled")
        self.btn_pause.configure(state="disabled")
        self.btn_stop.configure(state="disabled")

    def _build_alarm_banner(self):
        self.alarm_banner = ctk.CTkFrame(self, fg_color=ACCENT_RED,
                                          height=40, corner_radius=0)
        ctk.CTkLabel(self.alarm_banner,
                     text="⚠   THREAT DETECTED  —  VIGILANT EYE SECURITY ALERT  —  WEAPON IDENTIFIED  ⚠",
                     font=("Courier New", F_MID, "bold"),
                     text_color="white").pack(side="left", padx=20)
        ctk.CTkButton(self.alarm_banner, text="DISMISS", width=90, height=28,
                      fg_color="#CC0000", hover_color="#990000",
                      font=("Courier New", F_SMALL, "bold"),
                      command=self._dismiss_alarm).pack(side="right", padx=14)

    # ── [FIX-1] Live feed panel — pack_propagate(False) on feed_container ─
    def _build_live_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=10)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._panel_title(panel, "◈  LIVE SURVEILLANCE FEED")

        self.feed_container = ctk.CTkFrame(panel, fg_color=BG_BASE, corner_radius=8)
        self.feed_container.pack(fill="both", expand=True, padx=8, pady=(6, 8))
        # [FIX-1] Prevent the container expanding to fit the image label
        self.feed_container.pack_propagate(False)

        self.feed_label = ctk.CTkLabel(self.feed_container, text="",
                                       fg_color=BG_BASE, corner_radius=0)
        self.feed_label.pack(fill="both", expand=True)
        self._show_feed_placeholder()

        info = ctk.CTkFrame(self.feed_container, fg_color=BG_CARD2,
                            height=30, corner_radius=0)
        info.pack(fill="x")
        info.pack_propagate(False)
        self.lbl_cam_status = ctk.CTkLabel(info, text="● CAMERA OFFLINE",
                                           font=("Courier New", F_SMALL),
                                           text_color=ACCENT_RED)
        self.lbl_cam_status.pack(side="left", padx=10)
        self.lbl_ts = ctk.CTkLabel(info, text="",
                                   font=("Courier New", F_SMALL),
                                   text_color=TEXT_SEC)
        self.lbl_ts.pack(side="right", padx=10)
        self.lbl_det_live = ctk.CTkLabel(info, text="OBJECTS: 0",
                                         font=("Courier New", F_SMALL, "bold"),
                                         text_color=CORNFLOWER_L)
        self.lbl_det_live.pack(side="right", padx=10)

        tr = ctk.CTkFrame(self.feed_container, fg_color=BG_CARD,
                          height=26, corner_radius=0)
        tr.pack(fill="x")
        tr.pack_propagate(False)
        ctk.CTkLabel(tr, text="THREAT",
                     font=("Courier New", F_SMALL),
                     text_color=TEXT_DIM).pack(side="left", padx=8)
        self.threat_bar = ctk.CTkProgressBar(tr, height=10, corner_radius=3,
                                              fg_color=BG_CARD2,
                                              progress_color=ACCENT_GREEN)
        self.threat_bar.set(0)
        self.threat_bar.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_threat_score = ctk.CTkLabel(tr, text="0  LOW",
                                             font=("Courier New", F_SMALL, "bold"),
                                             text_color=ACCENT_GREEN)
        self.lbl_threat_score.pack(side="left", padx=(0, 8))

    def _show_feed_placeholder(self):
        W, H = 640, 600
        img  = Image.new("RGB", (W, H), BG_BASE)
        draw = ImageDraw.Draw(img)
        for x in range(0, W, 40):
            draw.line([(x, 0), (x, H)], fill="#111520", width=1)
        for y in range(0, H, 40):
            draw.line([(0, y), (W, y)], fill="#111520", width=1)
        cx, cy = W // 2, H // 2
        draw.rectangle([cx-80, cy-50, cx+80, cy+50], outline=CORNFLOWER_G, width=1)
        for bx, by, dx, dy in [(20,20,1,1),(W-20,20,-1,1),(20,H-20,1,-1),(W-20,H-20,-1,-1)]:
            draw.line([(bx, by), (bx+dx*28, by)], fill=CORNFLOWER, width=2)
            draw.line([(bx, by), (bx, by+dy*28)], fill=CORNFLOWER, width=2)
        draw.text((cx, cy+80), "AWAITING FEED  —  LOAD MODEL & START",
                  fill=CORNFLOWER_G, anchor="mm")
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=(W, H))
        self.feed_label.configure(image=ci, text="")
        self.feed_label._image = ci

    def _build_center_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=10)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 4))
        self._panel_title(panel, "◈  ANALYSIS  &  INTELLIGENCE")

        # ── Scrollable body — all analysis widgets live here ──────────────
        # Scrolls vertically if the window is shorter than the total content
        body = ctk.CTkScrollableFrame(
            panel, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BG_CARD2,
            scrollbar_button_hover_color=CORNFLOWER_D)
        body.pack(fill="both", expand=True)
        self._bind_mousewheel(body)

        # 4 stat boxes
        sr = ctk.CTkFrame(body, fg_color="transparent")
        sr.pack(fill="x", padx=8, pady=(6, 4))
        for i in range(4):
            sr.columnconfigure(i, weight=1)
        self.stat_fps   = self._stat_box(sr, "FPS",       "0", ACCENT_GREEN, 0)
        self.stat_conf  = self._stat_box(sr, "CONF",      "—", CORNFLOWER_L, 1)
        self.stat_alert = self._stat_box(sr, "TOTAL DET", "0", ACCENT_AMBER, 2)
        self.stat_alarm = self._stat_box(sr, "ALARMS",    "0", ACCENT_RED,   3)

        self._sec_lbl(body, "FPS  TREND")
        self.fps_canvas = tk.Canvas(body, height=46, bg=BG_CARD2, highlightthickness=0)
        self.fps_canvas.pack(fill="x", padx=8, pady=(1, 3))

        self._sec_lbl(body, "DETECTION  VOLUME")
        self.det_canvas = tk.Canvas(body, height=46, bg=BG_CARD2, highlightthickness=0)
        self.det_canvas.pack(fill="x", padx=8, pady=(1, 3))

        self._sec_lbl(body, "24-HOUR  ACTIVITY")
        self.hourly_canvas = tk.Canvas(body, height=50, bg=BG_CARD2, highlightthickness=0)
        self.hourly_canvas.pack(fill="x", padx=8, pady=(1, 3))

        # Class breakdown — height 90 so multiple rows are comfortably visible
        self._sec_lbl(body, "CLASS  BREAKDOWN")
        self.class_frame = ctk.CTkScrollableFrame(body, fg_color=BG_CARD2,
                                                   height=90, corner_radius=6)
        self.class_frame.pack(fill="x", padx=8, pady=(1, 6))
        self._bind_mousewheel(self.class_frame)

        # ── Heatmap stacked above Geolocation ────────────────────────────
        # Heatmap card
        hm_card = ctk.CTkFrame(body, fg_color=BG_CARD2, corner_radius=8)
        hm_card.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(hm_card, text="  MOTION  HEATMAP",
                     font=("Courier New", F_SMALL, "bold"),
                     text_color=CORNFLOWER, anchor="w").pack(fill="x", padx=6, pady=(5, 2))
        self.heatmap_lbl = ctk.CTkLabel(hm_card, text="", fg_color=BG_BASE,
                                        corner_radius=4, width=160, height=140)
        self.heatmap_lbl.pack(fill="x", padx=6, pady=(0, 6))
        self._render_heatmap()

        # Geo card — sits below heatmap
        geo_card = ctk.CTkFrame(body, fg_color=BG_CARD2, corner_radius=8)
        geo_card.pack(fill="x", padx=8, pady=(0, 10))
        ctk.CTkLabel(geo_card, text="  SENSOR  GEOLOCATION",
                     font=("Courier New", F_SMALL, "bold"),
                     text_color=CORNFLOWER, anchor="w").pack(fill="x", padx=6, pady=(5, 2))
        self.map_canvas = ctk.CTkLabel(geo_card, text="", fg_color=BG_BASE,
                                       corner_radius=4, width=160, height=100)
        self.map_canvas.pack(fill="x", padx=6, pady=(0, 4))
        self._draw_mini_map()

        geo_rows = ctk.CTkFrame(geo_card, fg_color="transparent")
        geo_rows.pack(fill="x", padx=8, pady=(0, 6))

        def geo_row(label, val):
            r = ctk.CTkFrame(geo_rows, fg_color="transparent")
            r.pack(fill="x", pady=1)
            ctk.CTkLabel(r, text=f"{label}:", font=("Courier New", F_TINY),
                         text_color=TEXT_DIM, width=58, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(r, text=val, font=("Courier New", F_TINY, "bold"),
                               text_color=CORNFLOWER_L, anchor="w")
            lbl.pack(side="left")
            return lbl

        self.lbl_geo_city    = geo_row("CITY",    "…")
        self.lbl_geo_country = geo_row("COUNTRY", "…")
        self.lbl_geo_coords  = geo_row("COORDS",  "…")
        self.lbl_geo_ip      = geo_row("IP",       "…")
        self.lbl_geo_isp     = geo_row("ISP",      "…")

    def _build_right_panel(self, parent):
        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=2, sticky="nsew")
        # minsize prevents any section from collapsing to zero on short windows
        right.rowconfigure(0, weight=4, minsize=200)
        right.rowconfigure(1, weight=3, minsize=170)
        right.rowconfigure(2, weight=3, minsize=150)
        right.columnconfigure(0, weight=1)

        # ── Captured detections ───────────────────────────────────────────
        det_panel = ctk.CTkFrame(right, fg_color=BG_PANEL, corner_radius=10)
        det_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self._panel_title(det_panel, "◈  CAPTURED DETECTIONS  (click to enlarge)")
        self.det_scroll = ctk.CTkScrollableFrame(det_panel, fg_color=BG_CARD2,
                                                  corner_radius=6)
        self.det_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.det_scroll.columnconfigure((0, 1), weight=1)
        self._bind_mousewheel(self.det_scroll)

        # ── Event log ─────────────────────────────────────────────────────
        ev_panel = ctk.CTkFrame(right, fg_color=BG_PANEL, corner_radius=10)
        ev_panel.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self._panel_title(ev_panel, "◈  EVENT LOG ANALYSIS")
        filter_row = ctk.CTkFrame(ev_panel, fg_color="transparent")
        filter_row.pack(fill="x", padx=8, pady=(4, 2))
        ctk.CTkLabel(filter_row, text="FILTER:", font=("Courier New", F_SMALL),
                     text_color=TEXT_DIM).pack(side="left")
        self.event_filter = ctk.CTkEntry(filter_row, height=24, width=120,
                                         font=("Courier New", F_SMALL),
                                         fg_color=BG_CARD, border_color=BG_BORDER,
                                         placeholder_text="class / keyword…")
        self.event_filter.pack(side="left", padx=4)
        ctk.CTkButton(filter_row, text="APPLY", width=52, height=24,
                      font=("Courier New", F_SMALL), fg_color=CORNFLOWER_D,
                      command=self._apply_event_filter).pack(side="left")
        self.event_list = ctk.CTkScrollableFrame(ev_panel, fg_color=BG_CARD2,
                                                  corner_radius=6)
        self.event_list.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self._bind_mousewheel(self.event_list)

        # ── Terminal ──────────────────────────────────────────────────────
        term_panel = ctk.CTkFrame(right, fg_color=BG_PANEL, corner_radius=10)
        term_panel.grid(row=2, column=0, sticky="nsew")
        self._panel_title(term_panel, "◈  SYSTEM  TERMINAL")
        ctk.CTkButton(term_panel, text="CLR", width=38, height=20,
                      font=("Courier New", F_SMALL), fg_color=BG_CARD2,
                      command=self._clear_terminal).place(relx=1.0, rely=0,
                                                          anchor="ne", x=-8, y=5)
        self.terminal = ctk.CTkTextbox(term_panel,
                                       font=("Courier New", F_BODY),
                                       fg_color=BG_BASE, text_color=CORNFLOWER_L,
                                       corner_radius=6, wrap="word",
                                       state="disabled")
        self.terminal.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.terminal._textbox.tag_configure("info",  foreground=CORNFLOWER_L)
        self.terminal._textbox.tag_configure("warn",  foreground=ACCENT_AMBER)
        self.terminal._textbox.tag_configure("error", foreground=ACCENT_RED)
        self.terminal._textbox.tag_configure("good",  foreground=ACCENT_GREEN)
        self.terminal._textbox.tag_configure("ts",    foreground=TEXT_DIM)
        self.terminal._textbox.tag_configure(
            "alarm", foreground=ACCENT_RED,
            font=("Courier New", F_BODY, "bold"))

    # ═════════════════════════════════════════════════════════════════════
    # MOUSEWHEEL SCROLL HELPER
    # ═════════════════════════════════════════════════════════════════════
    def _bind_mousewheel(self, widget):
        """
        Bind mousewheel events to a CTkScrollableFrame so hovering over it
        and scrolling works on Windows, macOS and Linux without clicking first.
        Bindings propagate to all child widgets inside the frame.
        """
        def _on_mousewheel(event):
            # Windows / macOS give delta; Linux uses Button-4/5
            if event.num == 4:
                widget._parent_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                widget._parent_canvas.yview_scroll(1, "units")
            else:
                widget._parent_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)), "units")

        def _bind_all(w):
            w.bind("<MouseWheel>", _on_mousewheel, add="+")
            w.bind("<Button-4>",   _on_mousewheel, add="+")
            w.bind("<Button-5>",   _on_mousewheel, add="+")
            for child in w.winfo_children():
                _bind_all(child)

        # Bind to the scrollable canvas and its interior frame
        try:
            _bind_all(widget._parent_canvas)
            _bind_all(widget)
        except AttributeError:
            pass

    # ═════════════════════════════════════════════════════════════════════
    # WIDGET HELPERS
    # ═════════════════════════════════════════════════════════════════════
    def _panel_title(self, parent, text):
        bar = ctk.CTkFrame(parent, fg_color=BG_CARD2, height=36, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkFrame(bar, fg_color=CORNFLOWER, width=4,
                     corner_radius=0).pack(side="left", fill="y")
        ctk.CTkLabel(bar, text=f"  {text}",
                     font=("Courier New", F_MID, "bold"),
                     text_color=CORNFLOWER, anchor="w").pack(side="left", padx=8)

    def _chip(self, parent, label, value, color):
        f = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=6)
        f.pack(side="right", padx=3)
        ctk.CTkLabel(f, text=label, font=("Courier New", F_TINY),
                     text_color=TEXT_DIM).pack(padx=10, pady=(4, 0))
        lbl = ctk.CTkLabel(f, text=value, font=("Courier New", F_LG, "bold"),
                           text_color=color)
        lbl.pack(padx=10, pady=(0, 4))
        return lbl

    def _hdr_btn(self, parent, text, cmd, color):
        b = ctk.CTkButton(parent, text=text, command=cmd,
                          font=("Courier New", F_SMALL, "bold"),
                          fg_color=color, hover_color=CORNFLOWER_D,
                          text_color=TEXT_PRIMARY,
                          width=110, height=32, corner_radius=5)
        b.pack(side="left", padx=2)
        return b

    def _stat_box(self, parent, label, value, color, col):
        f = ctk.CTkFrame(parent, fg_color=BG_CARD2, corner_radius=7)
        f.grid(row=0, column=col, padx=2, pady=2, sticky="ew")
        ctk.CTkLabel(f, text=label, font=("Courier New", F_TINY),
                     text_color=TEXT_DIM).pack(pady=(4, 0))
        lbl = ctk.CTkLabel(f, text=value,
                           font=("Courier New", F_LG, "bold"), text_color=color)
        lbl.pack(pady=(0, 4))
        return lbl

    def _sec_lbl(self, parent, text):
        ctk.CTkLabel(parent, text=f"  {text}",
                     font=("Courier New", F_SMALL, "bold"),
                     text_color=CORNFLOWER, anchor="w").pack(fill="x", padx=8, pady=(3, 0))

    # ═════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═════════════════════════════════════════════════════════════════════
    def _redirect_terminal(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TerminalRedirect(lambda t: self.log_queue.put(("info", t)))
        sys.stderr = TerminalRedirect(lambda t: self.log_queue.put(("warn", t)))

    def _log(self, msg, level="info"):
        self.log_queue.put((level, msg))

    def _flush_log_queue(self):
        try:
            while True:
                level, msg = self.log_queue.get_nowait()
                ts = datetime.now().strftime("%H:%M:%S")
                self.terminal.configure(state="normal")
                self.terminal._textbox.insert("end", f"[{ts}] ", "ts")
                self.terminal._textbox.insert("end", msg.rstrip() + "\n", level)
                self.terminal._textbox.see("end")
                self.terminal.configure(state="disabled")
        except queue.Empty:
            pass

    def _clear_terminal(self):
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    # ═════════════════════════════════════════════════════════════════════
    # ALARM
    # ═════════════════════════════════════════════════════════════════════
    def _toggle_alarm(self):
        self.alarm.enabled = not self.alarm.enabled
        if self.alarm.enabled:
            self.btn_alarm.configure(text="🔔 ALARM ON",  fg_color="#2A5030")
            self._log("Alarm system ENABLED.", "good")
        else:
            self.btn_alarm.configure(text="🔕 ALARM OFF", fg_color=BG_CARD2)
            self._log("Alarm system DISABLED.", "warn")

    def _fire_alarm(self):
        self._alarm_flashing    = True
        self._alarm_flash_count = 0
        self.alarm_banner.pack(fill="x")
        self._do_flash()
        self._log("⚠  ALARM TRIGGERED — THREAT DETECTED", "alarm")

    def _do_flash(self):
        if not self._alarm_flashing:
            return
        c = [ACCENT_RED, ACCENT_AMBER][self._alarm_flash_count % 2]
        self.alarm_banner.configure(fg_color=c)
        self._alarm_flash_count += 1
        if self._alarm_flash_count < 20:
            self.after(120, self._do_flash)
        else:
            self.alarm_banner.configure(fg_color=ACCENT_RED)
            self._alarm_flashing = False

    def _dismiss_alarm(self):
        self.alarm_banner.pack_forget()
        self.alarm.reset()

    # ═════════════════════════════════════════════════════════════════════
    # CONTROLS
    # ═════════════════════════════════════════════════════════════════════
    def _load_model(self):
        selected = self.model_var.get()
        if selected == "(no models found)":
            self._log("No .pt files found in models/ folder.", "error")
            return
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        candidate  = os.path.join(models_dir, selected)
        if not os.path.exists(candidate):
            candidate = selected
        if not os.path.exists(candidate):
            self._log(f"Model file not found: {selected}", "error")
            return
        self._log(f"Loading model: {selected} …", "info")
        threading.Thread(target=self._do_load_model, args=(candidate, selected),
                         daemon=True).start()

    def _refresh_model_list(self):
        found = scan_models_folder()
        if not found:
            self._log("models/ folder is empty or missing.", "warn")
            found = ["(no models found)"]
        self.model_selector.configure(values=found)
        self.model_var.set(found[0])
        self._log(f"Model list refreshed — {len(found)} file(s) found.", "info")

    def _do_load_model(self, path, display_name):
        try:
            self.model = YOLO(path)
            # [PERF-1] Auto-move to GPU if available
            self.model.to(_TORCH_DEVICE)
            if _TORCH_DEVICE == "cuda":
                try:
                    self.model.model.half()   # FP16 — ~30-40% faster on GPU
                    self._log("FP16 half-precision enabled on GPU.", "good")
                except Exception:
                    pass
            self._log(f"Model loaded ✓  ({display_name})  →  {_TORCH_DEVICE.upper()}", "good")
            self.btn_cam.configure(state="normal")
            self.btn_video.configure(state="normal")
            self.btn_image.configure(state="normal")
        except Exception as e:
            self._log(f"Model load failed: {e}", "error")

    def _start_camera(self):
        if self.running:
            self._stop_detection()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log("Cannot access webcam.", "error")
            return
        self.cap           = cap
        self.is_video_file = False
        self._start_common("CAMERA")

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                       ("All files", "*.*")])
        if not path:
            return
        if self.running:
            self._stop_detection()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._log(f"Cannot open video: {path}", "error")
            return
        self.cap           = cap
        self.is_video_file = True
        fname = os.path.basename(path)
        self._log(f"Video loaded: {fname}", "good")
        self._start_common(f"VIDEO: {fname[:28]}")

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Static Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        if self.running:
            self._stop_detection()
        fname = os.path.basename(path)
        self._log(f"Analyzing static image: {fname}", "info")
        self.is_video_file = True
        self._draw_status_dot(ACCENT_AMBER)
        self.lbl_cam_status.configure(text=f"● IMAGE: {fname[:28]}", text_color=ACCENT_AMBER)
        threading.Thread(target=self._process_static_image, args=(path,), daemon=True).start()

    def _process_static_image(self, path):
        frame = cv2.imread(path)
        if frame is None:
            self._log("Failed to load image file.", "error")
            self.after(0, self._draw_status_dot, ACCENT_RED)
            return

        # [FIX-4] Inference wrapped in try/except
        try:
            results   = self.model(frame, conf=self.confidence_threshold, verbose=False)
            annotated = results[0].plot()
        except Exception as e:
            self._log(f"Inference error on image: {e}", "error")
            self.after(0, self._draw_status_dot, ACCENT_RED)
            return

        boxes = results[0].boxes
        n_det = len(boxes) if boxes is not None else 0

        classes_this, confs_this = [], []
        if n_det and boxes.cls is not None:
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                name = self.model.names[int(cls_id)]
                classes_this.append(name)
                confs_this.append(conf)
                self.class_counts[name] = self.class_counts.get(name, 0) + 1
                self.confidence_vals.append(conf)

        if n_det:
            self.log_eng.record(classes_this, confs_this)
            score = self.log_eng.threat_score()
            if score >= 40:
                self.alarm.trigger(self._fire_alarm, classes_this)

        full_pil = cv2_to_pil_noflip(annotated)
        if confs_this:
            meta = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                    f"IMAGE  |  Objects: {n_det}  |  "
                    f"Classes: {', '.join(set(classes_this)) or '—'}  |  "
                    f"Avg conf: {sum(confs_this)/len(confs_this)*100:.0f}%")
        else:
            meta = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  IMAGE  |  No detections"
        self._gallery.append((full_pil, meta))
        gallery_ref = (full_pil, meta)
        thumb       = full_pil.resize((176, 116), Image.LANCZOS)

        self.after(0, self._set_feed_image, full_pil, 0.0, n_det)
        self.after(0, self._insert_thumb_ref, thumb, n_det, classes_this, gallery_ref)
        if n_det:
            self.after(0, self._insert_event, n_det, classes_this, confs_this)
        self.after(200, self._draw_status_dot, ACCENT_GREEN if n_det else TEXT_DIM)
        self._log(f"Image analysis complete — {n_det} object(s) detected.",
                  "good" if n_det else "info")

    def _start_common(self, source_label):
        self.running       = True
        self.paused        = False
        self.session_start = time.time()
        self.frame_count   = 0
        self._det_frame_ctr = 0
        self._pause_event.set()   # [FIX-3] ensure event is set (running)
        self._draw_status_dot(ACCENT_GREEN)
        self.lbl_cam_status.configure(
            text=f"● {source_label}", text_color=ACCENT_GREEN)
        self.btn_cam.configure(state="disabled")
        self.btn_video.configure(state="disabled")
        self.btn_image.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self._log(f"Detection started — {source_label}.", "good")
        threading.Thread(target=self._detection_loop, daemon=True).start()

    # [FIX-3] threading.Event pause — zero CPU when paused
    def _pause_detection(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self._pause_event.clear()   # block the inference thread
            self.btn_pause.configure(text="▶ RESUME")
            self._draw_status_dot(ACCENT_AMBER)
            self._log("Detection paused.", "warn")
        else:
            self._pause_event.set()     # unblock the inference thread
            self.btn_pause.configure(text="⏸ PAUSE")
            self._draw_status_dot(ACCENT_GREEN)
            self._log("Detection resumed.", "good")

    def _stop_detection(self):
        self.running = False
        self.paused  = False
        self._pause_event.set()   # always unblock so thread can exit
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.configure(state="normal" if self.model else "disabled")
        self.btn_video.configure(state="normal" if self.model else "disabled")
        self.btn_image.configure(state="normal" if self.model else "disabled")
        self.btn_pause.configure(state="disabled", text="⏸ PAUSE")
        self.btn_stop.configure(state="disabled")
        self._draw_status_dot(ACCENT_RED)
        self.lbl_cam_status.configure(text="● OFFLINE", text_color=ACCENT_RED)
        self._show_feed_placeholder()
        self._log("Detection stopped.", "warn")

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile=f"vigi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if path:
            self.log_eng.export_csv(path)
            self._log(f"Log exported → {path}", "good")

    def _apply_event_filter(self):
        kw = self.event_filter.get().strip().lower()
        self._redraw_event_list(kw)

    # ═════════════════════════════════════════════════════════════════════
    # DETECTION LOOP  — [FIX-3] Event-based pause, [FIX-4] try/except,
    #                   [FIX-5] camera auto-reconnect, [PERF-3] frame skip
    # ═════════════════════════════════════════════════════════════════════
    def _detection_loop(self):
        prev_time       = time.time()
        reconnect_tries = 0

        while self.running:
            # [FIX-3] Zero-CPU wait when paused
            self._pause_event.wait()
            if not self.running:
                break

            ret, frame = self.cap.read()

            # [FIX-5] Auto-reconnect for live camera (not video files)
            if not ret:
                if self.is_video_file:
                    self._log("Video playback complete.", "info")
                    break
                reconnect_tries += 1
                self._log(f"Frame read failed — reconnect attempt {reconnect_tries}…", "warn")
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(0)
                if reconnect_tries >= 5:
                    self._log("Camera reconnect failed after 5 attempts.", "error")
                    break
                continue

            reconnect_tries = 0   # reset on successful read
            self.frame_count += 1

            # Mirror camera feed ONCE before all processing
            if not self.is_video_file:
                frame = cv2.flip(frame, 1)

            # [PERF-3] Frame skipping — skip inference on non-Nth frames
            if self.frame_count % self.INFERENCE_EVERY_N_FRAMES != 0:
                # Still push the frame so feed stays live
                feed_pil = cv2_to_pil_noflip(frame)
                try:
                    self._frame_queue.put_nowait((feed_pil, self.fps_val, 0))
                except queue.Full:
                    pass
                continue

            # [FIX-4] Guard against corrupted frames / inference errors
            try:
                results   = self.model(frame, conf=self.confidence_threshold, verbose=False)
                annotated = results[0].plot()
            except Exception as e:
                self._log(f"Inference error (frame dropped): {e}", "warn")
                continue

            cur       = time.time()
            fps       = 1.0 / (cur - prev_time) if prev_time else 0
            prev_time = cur
            self.fps_val = fps
            self.fps_history.append(fps)

            boxes = results[0].boxes
            n_det = len(boxes) if boxes is not None else 0
            self.det_history.append(n_det)

            classes_this = []
            confs_this   = []
            if n_det and boxes.cls is not None:
                for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    name = self.model.names[int(cls_id)]
                    self.class_counts[name] = self.class_counts.get(name, 0) + 1
                    self.confidence_vals.append(conf)
                    classes_this.append(name)
                    confs_this.append(conf)
                if boxes.xyxyn is not None:
                    self.heatmap.add(boxes.xyxyn.tolist())
                self.log_eng.record(classes_this, confs_this)
                score = self.log_eng.threat_score()
                if score >= 40:
                    self.alarm.trigger(self._fire_alarm, classes_this)

                # [PERF-4] Throttle thumbnail creation
                self._det_frame_ctr += 1
                if self._det_frame_ctr % self.THUMB_EVERY_N_DETECTIONS == 0:
                    full_pil = cv2_to_pil_noflip(annotated)
                    meta = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                            f"Objects: {n_det}  |  "
                            f"Classes: {', '.join(set(classes_this))}  |  "
                            f"Avg conf: {sum(confs_this)/len(confs_this)*100:.0f}%")
                    gallery_ref = (full_pil, meta)
                    self._gallery.append(gallery_ref)
                    thumb = full_pil.resize((176, 116), Image.LANCZOS)
                    self.after(0, self._insert_thumb_ref, thumb, n_det, classes_this, gallery_ref)
                    self.after(0, self._insert_event, n_det, classes_this, confs_this)

            feed_pil = cv2_to_pil_noflip(annotated)
            try:
                self._frame_queue.put_nowait((feed_pil, fps, n_det))
            except queue.Full:
                pass

        self.after(0, self._stop_detection)
        self._log("Detection loop ended.", "info")

    # ─── Decoupled 30 fps feed renderer ──────────────────────────────────
    def _schedule_feed_render(self):
        try:
            pil_img, fps, n_det = self._frame_queue.get_nowait()
            self._set_feed_image(pil_img, fps, n_det)
        except queue.Empty:
            pass
        self.after(33, self._schedule_feed_render)

    # ═════════════════════════════════════════════════════════════════════
    # FEED IMAGE  — [FIX-1] measure container not label, cap scale at 1.0
    # ═════════════════════════════════════════════════════════════════════
    def _set_feed_image(self, pil_img, fps, n_det):
        # [FIX-1] Measure the locked container — not the expanding label
        W = max(self.feed_container.winfo_width(),  480)
        H = max(self.feed_container.winfo_height(), 400)
        iw, ih = pil_img.size
        # [FIX-1] Cap scale at 1.0 — never upscale beyond native resolution
        scale  = min(W / iw, H / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
        canvas  = Image.new("RGB", (W, H), BG_BASE)
        ox, oy  = (W - nw) // 2, (H - nh) // 2
        canvas.paste(pil_img, (ox, oy))
        draw = ImageDraw.Draw(canvas)
        L = 24
        c = ACCENT_RED if n_det > 0 else CORNFLOWER
        for bx, by, dx, dy in [(2,2,1,1),(W-2,2,-1,1),(2,H-2,1,-1),(W-2,H-2,-1,-1)]:
            draw.line([(bx, by), (bx+dx*L, by)], fill=c, width=2)
            draw.line([(bx, by), (bx, by+dy*L)], fill=c, width=2)
        draw.rectangle([6,  6,  130, 26], fill="#00000099")
        draw.text((9, 8),  f"FPS {fps:.1f}", fill=ACCENT_GREEN)
        draw.rectangle([6, 28, 165, 48], fill="#00000099")
        draw.text((9, 30), f"OBJECTS  {n_det}",
                  fill=ACCENT_RED if n_det else CORNFLOWER_L)
        ts_txt = datetime.now().strftime("%H:%M:%S")
        draw.rectangle([W-90, 6,  W-2, 26], fill="#00000099")
        draw.text((W-88, 8),  ts_txt, fill=TEXT_SEC)
        src = "VIDEO" if self.is_video_file else "LIVE"
        draw.rectangle([W-80, 28, W-2, 48], fill="#00000099")
        draw.text((W-78, 30), src, fill=CORNFLOWER_L)
        ci = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(W, H))
        self.feed_label.configure(image=ci, text="")
        self.feed_label._image = ci
        self.lbl_det_live.configure(
            text=f"OBJECTS: {n_det}",
            text_color=ACCENT_RED if n_det else CORNFLOWER_L)
        self.lbl_ts.configure(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    # ═════════════════════════════════════════════════════════════════════
    # THUMBNAILS  +  GALLERY LIGHTBOX
    # ═════════════════════════════════════════════════════════════════════
    def _insert_thumb_ref(self, thumb_pil, n_det, classes, gallery_entry):
        bc = ACCENT_RED if n_det else CORNFLOWER
        bordered = Image.new("RGB", (180, 120), bc)
        bordered.paste(thumb_pil, (2, 2))
        draw = ImageDraw.Draw(bordered)
        draw.rectangle([2, 103, 178, 118], fill="#00000099")
        draw.text((5, 104),
                  datetime.now().strftime("%H:%M:%S") +
                  f"  {','.join(set(classes))[:22]}",
                  fill=TEXT_PRIMARY)
        ci = ctk.CTkImage(light_image=bordered, dark_image=bordered, size=(180, 120))
        lbl = ctk.CTkLabel(self.det_scroll, image=ci, text="",
                           fg_color="transparent", cursor="hand2")
        lbl._img = ci
        lbl.bind("<Button-1>",
                 lambda e, ref=gallery_entry: self._open_lightbox_ref(ref))
        lbl.grid(row=self.det_thumb_row, column=self.det_thumb_col, padx=2, pady=2)
        self.det_thumb_col += 1
        if self.det_thumb_col >= 2:
            self.det_thumb_col = 0
            self.det_thumb_row += 1
        children = self.det_scroll.winfo_children()
        if len(children) > 24:
            children[0].destroy()

    def _open_lightbox_ref(self, entry):
        images = list(self._gallery)
        if not images:
            return
        try:
            idx = images.index(entry)
        except ValueError:
            idx = len(images) - 1
        Lightbox(self, images, start_index=idx).focus()

    # ═════════════════════════════════════════════════════════════════════
    # EVENT LOG
    # ═════════════════════════════════════════════════════════════════════
    def _insert_event(self, n_det, classes, confs):
        ts    = datetime.now().strftime("%H:%M:%S")
        score = self.log_eng.threat_score()
        self._all_events.append({"ts": ts, "n": n_det,
                                  "classes": classes,
                                  "confs": confs, "score": score})
        kw = self.event_filter.get().strip().lower()
        self._redraw_event_list(kw)

    def _redraw_event_list(self, kw=""):
        for w in self.event_list.winfo_children():
            w.destroy()
        filtered = [e for e in self._all_events
                    if not kw or kw in " ".join(e["classes"]).lower()]
        for ev in reversed(list(filtered)[-60:]):
            tc  = threat_color(ev["score"])
            row = ctk.CTkFrame(self.event_list, fg_color=BG_CARD, corner_radius=4)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text="●", font=("Courier New", F_BODY),
                         text_color=tc, width=16).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=ev["ts"],
                         font=("Courier New", F_SMALL),
                         text_color=TEXT_DIM).pack(side="left")
            ctk.CTkLabel(row,
                         text=f"  {ev['n']} obj  "
                              f"[{','.join(set(ev['classes']))[:20]}]",
                         font=("Courier New", F_SMALL),
                         text_color=TEXT_PRIMARY).pack(side="left")
            ctk.CTkLabel(row, text=f"T:{ev['score']}",
                         font=("Courier New", F_SMALL, "bold"),
                         text_color=tc).pack(side="right", padx=8)

    # ═════════════════════════════════════════════════════════════════════
    # PERIODIC REFRESH  (500 ms)
    # ═════════════════════════════════════════════════════════════════════
    def _schedule_ui_updates(self):
        self._flush_log_queue()
        self._update_stats()
        self._draw_spark(self.fps_canvas, self.fps_history, ACCENT_GREEN, 60)
        self._draw_spark(self.det_canvas, self.det_history, CORNFLOWER, 10)
        self._draw_hourly_histogram()
        self._update_class_table()
        self._render_heatmap()
        self._draw_mini_map()
        self._update_threat_bar()
        self._tick_clock()
        self.after(500, self._schedule_ui_updates)

    def _tick_clock(self):
        if self.session_start and self.running:
            e = int(time.time() - self.session_start)
            ts = f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}"
            self.lbl_session.configure(text=ts)

    def _update_stats(self):
        avg_conf = (f"{sum(self.confidence_vals)/len(self.confidence_vals)*100:.0f}%"
                    if self.confidence_vals else "—")
        total_det = sum(r["n"] for r in self.log_eng.records)
        self.stat_fps.configure(text=f"{self.fps_val:.1f}")
        self.stat_conf.configure(text=avg_conf)
        self.stat_alert.configure(text=str(total_det))
        self.stat_alarm.configure(text=str(self.alarm.trigger_count))
        self.lbl_fps_hdr.configure(text=f"{self.fps_val:.1f}")
        self.lbl_frames_hdr.configure(text=str(self.frame_count))
        self.lbl_dets_hdr.configure(text=str(total_det))

    def _update_threat_bar(self):
        score = self.log_eng.threat_score()
        self.threat_bar.set(score / 100)
        tc = threat_color(score)
        self.threat_bar.configure(progress_color=tc)
        label = ("CRITICAL" if score >= 75 else
                 "HIGH"     if score >= 50 else
                 "MODERATE" if score >= 25 else "LOW")
        self.lbl_threat_score.configure(text=f"{score}  {label}", text_color=tc)
        self.lbl_threat_hdr.configure(text=label, text_color=tc)

    def _draw_spark(self, canvas, data, color, max_val):
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10:
            return
        pts = list(data)
        if len(pts) < 2:
            return
        step   = w / (len(pts) - 1)
        coords = []
        for i, v in enumerate(pts):
            x = i * step
            y = h - 4 - (v / max(max_val, 1)) * (h - 8)
            coords.extend([x, y])
        r, g, b  = hex_to_rgb(color)
        fill_hex = "#{:02x}{:02x}{:02x}".format(r//3, g//3, b//3)
        canvas.create_polygon([0, h] + coords + [w, h], fill=fill_hex, outline="")
        if len(coords) >= 4:
            canvas.create_line(coords, fill=color, width=2, smooth=True)
        canvas.create_text(w-4, 4, text=f"{pts[-1]:.1f}",
                           fill=color, anchor="ne", font=("Courier New", F_SMALL))

    def _draw_hourly_histogram(self):
        canvas = self.hourly_canvas
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10:
            return
        bars  = self.log_eng.hourly_bars()
        mx    = max(v for _, v in bars) or 1
        bw    = w / 24
        cur_h = datetime.now().hour
        for i, (lbl, val) in enumerate(bars):
            x0  = i * bw + 1
            x1  = x0 + bw - 2
            bh  = int((val / mx) * (h - 14))
            col = ACCENT_AMBER if i == cur_h else CORNFLOWER
            canvas.create_rectangle(x0, h-2-bh, x1, h-2, fill=col, outline="")
            if i % 6 == 0:
                canvas.create_text(x0+bw/2, h-1, text=lbl,
                                   fill=TEXT_DIM, anchor="s",
                                   font=("Courier New", F_TINY))

    def _update_class_table(self):
        for w in self.class_frame.winfo_children():
            w.destroy()
        if not self.class_counts:
            ctk.CTkLabel(self.class_frame, text="No detections yet.",
                         font=("Courier New", F_BODY),
                         text_color=TEXT_DIM).pack(anchor="w", padx=8, pady=4)
            return
        total = sum(self.class_counts.values())
        for cls, cnt in sorted(self.class_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total
            row = ctk.CTkFrame(self.class_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=cls.upper(),
                         font=("Courier New", F_BODY, "bold"),
                         text_color=TEXT_PRIMARY, width=100,
                         anchor="w").pack(side="left", padx=4)
            bar = ctk.CTkProgressBar(row, height=10, corner_radius=3,
                                     fg_color=BG_BASE, progress_color=CORNFLOWER)
            bar.set(pct)
            bar.pack(side="left", fill="x", expand=True, padx=4)
            ctk.CTkLabel(row, text=str(cnt),
                         font=("Courier New", F_BODY),
                         text_color=CORNFLOWER_L, width=40).pack(side="left")

    def _render_heatmap(self):
        w = max(self.heatmap_lbl.winfo_width(), 80)
        h = max(self.heatmap_lbl.winfo_height(), 80)
        side = min(w, h)
        img = self.heatmap.render((side, side))
        ci  = ctk.CTkImage(light_image=img, dark_image=img, size=(side, side))
        self.heatmap_lbl.configure(image=ci, text="")
        self.heatmap_lbl._img = ci

    # ═════════════════════════════════════════════════════════════════════
    # MISC
    # ═════════════════════════════════════════════════════════════════════
    def _draw_status_dot(self, color):
        c = self.status_canvas
        c.delete("all")
        c.create_oval(2, 2, 14, 14, fill=color, outline=color)

    def _on_close(self):
        self.running = False
        self._pause_event.set()   # unblock any waiting thread
        if self.cap:
            self.cap.release()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = VigiShieldApp()
    app.mainloop()