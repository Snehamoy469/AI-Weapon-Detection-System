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

# ── GPU / torch optional ──────────────────────────────────────────────────────
try:
    import torch
    _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _TORCH_DEVICE = "cpu"


try:
    from deepsort_tracker import (
        DeepSortWrapper,
        TrajectoryStore,
        ThreatPersistenceTimer,
        AlarmDeduplicator,
        WEAPON_CLASSES,         # frozenset of weapon class name strings
        HIGH_INTEREST_CLASSES,  # frozenset for future logic hooks
    )
    _DEEPSORT_MODULE_OK = True
except ImportError as _ds_err:
    _DEEPSORT_MODULE_OK = False
    _DEEPSORT_IMPORT_ERR = str(_ds_err)
    # Provide minimal stubs so the rest of the file parses without errors
    WEAPON_CLASSES        = frozenset({"gun", "pistol", "rifle", "weapon", "knife",
                                       "handgun", "firearm", "grenade", "bomb"})
    HIGH_INTEREST_CLASSES = frozenset({"person", "car", "truck"})

# ── NumPy Gaussian kernel for heatmap (unchanged from v1) ─────────────────────
def _make_gaussian_kernel(radius=6):
    size  = radius * 2 + 1
    ax    = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k     = np.exp(-(xx**2 + yy**2) / 8.0)
    return k

_GAUSS_KERNEL = _make_gaussian_kernel(6)

# ── Optional voice TTS ────────────────────────────────────────────────────────
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
# THEME  —  Cornflower Blue Industrial  (unchanged from v1)
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
ACCENT_TRACK = "#00DCAA"

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


def is_weapon_class(name):
    return str(name).strip().lower() in WEAPON_CLASSES


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
# GEOLOCATION  (unchanged)
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
# ALARM ENGINE  (unchanged from v1 — per-track dedup is in AlarmDeduplicator)
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
# LOG ANALYSIS ENGINE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class LogAnalysisEngine:
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
# MOTION HEATMAP  (unchanged — trajectory paths are separate in TrajectoryStore)
# ─────────────────────────────────────────────────────────────────────────────
class MotionHeatmap:
    def __init__(self, w=160, h=120):
        self.W   = w
        self.H   = h
        self.map = np.zeros((h, w), dtype=np.float32)

    def add(self, boxes_xyxyn):
        self.map *= 0.99
        r  = len(_GAUSS_KERNEL) // 2
        for b in boxes_xyxyn:
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            px = int(cx * self.W)
            py = int(cy * self.H)
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
        img  = Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)
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
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



# ═════════════════════════════════════════════════════════════════════════════
# LIGHTBOX  (unchanged)
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
        resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
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

    INFERENCE_EVERY_N_FRAMES = 5
    THUMB_EVERY_N_DETECTIONS = 10

    def __init__(self):
        super().__init__()
        self.title("VIGILANT EYE - Industrial Surveillance Platform v2")
        self.geometry("1820x1040")
        self.minsize(1400, 860)
        self.configure(fg_color=BG_BASE)
        ctk.set_appearance_mode("dark")

        # ── Core state (all unchanged from v1) ────────────────────────────
        self.model           = None
        self.model_is_weapon_only = False
        self.cap             = None
        self.running         = False
        self.paused          = False
        self.is_video_file   = False
        self.frame_count     = 0
        self.fps_val         = 0.0
        self.confidence_threshold = 0.15
        self.fps_history     = collections.deque(maxlen=80)
        self.det_history     = collections.deque(maxlen=80)
        self.class_counts    = {}
        self.confidence_vals = collections.deque(maxlen=300)
        self.session_start   = None
        self.log_queue       = queue.Queue()
        self._all_events     = collections.deque(maxlen=5_000)
        self.det_thumb_row   = 0
        self.det_thumb_col   = 0
        self._det_frame_ctr  = 0
        self._gallery        = collections.deque(maxlen=100)
        self._frame_queue    = queue.Queue(maxsize=2)
        self._pause_event    = threading.Event()
        self._pause_event.set()

        # ── Sub-systems (unchanged) ───────────────────────────────────────
        self.alarm   = AlarmEngine()
        self.log_eng = LogAnalysisEngine()
        self.heatmap = MotionHeatmap()

        # Tracker state variables
        # ds_tracker       : DeepSortWrapper — the main tracker (reset per session)
        # trajectory_store : TrajectoryStore — per-track path history
        # threat_timer     : ThreatPersistenceTimer — guards weapon alarm debounce
        # alarm_dedup      : AlarmDeduplicator — per-track cooldown
        # _active_track_count : int — passed to feed renderer and stat display
        #
        # Initialised as None/zero here; reset() called inside _start_common.
        # This avoids loading the embedder network before any session starts.
        self.ds_tracker          = None
        self.trajectory_store    = TrajectoryStore(max_len=60)
        self.threat_timer        = ThreatPersistenceTimer(persist_secs=2.5)
        self.alarm_dedup         = AlarmDeduplicator(cooldown_secs=30.0)
        self._active_track_count = 0

        # ── Geo ───────────────────────────────────────────────────────────
        self.geo = {"city": "Fetching…", "regionName": "", "country": "",
                    "lat": 0.0, "lon": 0.0, "isp": "—", "query": "—"}
        self._geo_dirty = True
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
        self._log(f"VIGILANT EYE v2 ready. Device: {device_info}", "good")
        self._log("Select model → LOAD → Camera / Video / Image.", "info")

        # [DEEPSORT-2] Warn if module failed to import
        if not _DEEPSORT_MODULE_OK:
            self._log(
                "⚠  deepsort_tracker.py not found — place it next to this file.",
                "error")
        else:
            self._log("Tracking module: LOADED", "good")
            if _TORCH_DEVICE == "cuda":
                self._log("Appearance embedder: GPU (mobilenet FP16) ✓", "good")
            else:
                self._log("CPU tracking mode active. GPU recommended for best speed.", "warn")

        if HAS_TTS:
            self._log("Voice alarm: ACTIVE (pyttsx3)", "good")
        else:
            self._log("pyttsx3 not found — falling back to beep alarm.", "warn")

    # ═════════════════════════════════════════════════════════════════════
    # GEOLOCATION  (unchanged)
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
        self._geo_dirty = True

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
    # UI BUILD  (only header chip modified)
    # ═════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self._build_header()
        self._build_alarm_banner()
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        content.columnconfigure(0, weight=62)
        content.columnconfigure(1, weight=18)
        content.columnconfigure(2, weight=20)
        content.rowconfigure(0, weight=1)
        self._build_live_panel(content)
        self._build_center_panel(content)
        self._build_right_panel(content)

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        hdr.pack(fill="x")

        row1 = ctk.CTkFrame(hdr, fg_color="transparent", height=72)
        row1.pack(fill="x", padx=12, pady=(10, 4))
        row1.pack_propagate(False)

        brand = ctk.CTkFrame(row1, fg_color="transparent")
        brand.pack(side="left")
        self.status_canvas = tk.Canvas(brand, width=14, height=14,
                                       bg=BG_PANEL, highlightthickness=0)
        self.status_canvas.pack(side="left", padx=(0, 8))
        self._draw_status_dot(ACCENT_RED)
        ctk.CTkLabel(brand, text="VIGILANT EYE  v2",
                     font=("Courier New", F_TITLE, "bold"),
                     text_color=CORNFLOWER).pack(side="left")

        chips = ctk.CTkFrame(row1, fg_color="transparent")
        chips.pack(side="right")
        self.lbl_session    = self._chip(chips, "SESSION",    "—",   CORNFLOWER)
        self.lbl_fps_hdr    = self._chip(chips, "FPS",        "0",   ACCENT_GREEN)
        self.lbl_frames_hdr = self._chip(chips, "FRAMES",     "0",   CORNFLOWER_L)
        self.lbl_dets_hdr   = self._chip(chips, "DETECT",     "0",   ACCENT_AMBER)
        self.lbl_threat_hdr = self._chip(chips, "THREAT",     "LOW", ACCENT_GREEN)
        self.lbl_tracks_hdr = self._chip(chips, "OBJECTS",    "0",   ACCENT_TRACK)

        ctk.CTkFrame(hdr, fg_color=BG_BORDER, height=1,
                     corner_radius=0).pack(fill="x", padx=0)

        row2 = ctk.CTkFrame(hdr, fg_color="transparent", height=44)
        row2.pack(fill="x", padx=10, pady=(4, 8))
        row2.pack_propagate(False)

        _model_list = scan_models_folder() or ["(no models found)"]
        self.model_var = ctk.StringVar(value=_model_list[0])
        self.model_selector = ctk.CTkOptionMenu(
            row2, values=_model_list, variable=self.model_var,
            width=148, height=32,
            fg_color=BG_CARD2, button_color=CORNFLOWER_D,
            dropdown_fg_color=BG_CARD, dropdown_hover_color=CORNFLOWER_G,
            font=("Courier New", F_SMALL, "bold"), text_color=TEXT_PRIMARY,
        )
        self.model_selector.pack(side="left", padx=(0, 2))
        ctk.CTkButton(row2, text="⟳", width=32, height=32,
                      fg_color=BG_CARD2, hover_color=CORNFLOWER_G,
                      font=("Courier New", F_MID, "bold"),
                      text_color=CORNFLOWER_L,
                      command=self._refresh_model_list).pack(side="left", padx=(0, 8))

        self.btn_load   = self._hdr_btn(row2, "⬆ LOAD",      self._load_model,      CORNFLOWER_G)
        self.btn_cam    = self._hdr_btn(row2, "📷 CAMERA",    self._start_camera,    CORNFLOWER_D)
        self.btn_video  = self._hdr_btn(row2, "🎬 VIDEO",     self._browse_video,    "#2D4A7A")
        self.btn_image  = self._hdr_btn(row2, "🖼 IMAGE",     self._browse_image,    "#4A3B7A")
        self.btn_pause  = self._hdr_btn(row2, "⏸ PAUSE",      self._pause_detection, BG_CARD2)
        self.btn_stop   = self._hdr_btn(row2, "■ STOP",       self._stop_detection,  ACCENT_RED)
        self.btn_export = self._hdr_btn(row2, "↓ EXPORT",     self._export_log,      CORNFLOWER_G)
        self.btn_alarm  = self._hdr_btn(row2, "🔔 ALARM ON",  self._toggle_alarm,    "#2A5030")
        self.btn_test   = self._hdr_btn(row2, "TEST CAM",     self._test_camera,     "#3A5A3A")

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

    def _build_live_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=10)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._panel_title(panel, "LIVE SURVEILLANCE FEED")

        self.feed_container = ctk.CTkFrame(panel, fg_color=BG_BASE, corner_radius=8)
        self.feed_container.pack(fill="both", expand=True, padx=6, pady=(4, 6))
        self.feed_container.pack_propagate(False)

        self.feed_label = ctk.CTkLabel(self.feed_container, text="",
                                       fg_color=BG_BASE, corner_radius=0)
        self.feed_label.pack(fill="both", expand=True)
        self._show_feed_placeholder()

        self.lbl_cam_status = None
        self.lbl_ts = None
        self.lbl_det_live = None
        self.threat_bar = None
        self.lbl_threat_score = None

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
        draw.text((cx, cy+80), "AWAITING FEED - LOAD MODEL & START",
                  fill=CORNFLOWER_G, anchor="mm")
        draw.text((cx, cy+100), "Object detection ready",
                  fill=TEXT_DIM, anchor="mm")
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=(W, H))
        self.feed_label.configure(image=ci, text="")
        self.feed_label._image = ci

    def _build_center_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=10)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 4))
        self._panel_title(panel, "ANALYSIS & INTELLIGENCE")

        body = ctk.CTkScrollableFrame(
            panel, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BG_CARD2,
            scrollbar_button_hover_color=CORNFLOWER_D)
        body.pack(fill="both", expand=True)
        self._bind_mousewheel(body)

        sr = ctk.CTkFrame(body, fg_color="transparent")
        sr.pack(fill="x", padx=8, pady=(6, 4))
        for i in range(5):
            sr.columnconfigure(i, weight=1)
        self.stat_fps    = self._stat_box(sr, "FPS",       "0",   ACCENT_GREEN,  0)
        self.stat_conf   = self._stat_box(sr, "CONF",      "—",   CORNFLOWER_L,  1)
        self.stat_alert  = self._stat_box(sr, "TOTAL DET", "0",   ACCENT_AMBER,  2)
        self.stat_alarm  = self._stat_box(sr, "ALARMS",    "0",   ACCENT_RED,    3)
        self.stat_tracks = self._stat_box(sr, "OBJECTS",   "0",   ACCENT_TRACK,  4)

        self._sec_lbl(body, "FPS  TREND")
        self.fps_canvas = tk.Canvas(body, height=46, bg=BG_CARD2, highlightthickness=0)
        self.fps_canvas.pack(fill="x", padx=8, pady=(1, 3))

        self._sec_lbl(body, "DETECTION  VOLUME")
        self.det_canvas = tk.Canvas(body, height=46, bg=BG_CARD2, highlightthickness=0)
        self.det_canvas.pack(fill="x", padx=8, pady=(1, 3))

        self._sec_lbl(body, "24-HOUR  ACTIVITY")
        self.hourly_canvas = tk.Canvas(body, height=50, bg=BG_CARD2, highlightthickness=0)
        self.hourly_canvas.pack(fill="x", padx=8, pady=(1, 3))

        self._sec_lbl(body, "CLASS  BREAKDOWN")
        self.class_frame = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=6, height=90)
        self.class_frame.pack(fill="x", padx=8, pady=(1, 4))
        self._bind_mousewheel(self.class_frame)

        self._sec_lbl(body, "THREAT  LEVEL")
        trow = ctk.CTkFrame(body, fg_color="transparent")
        trow.pack(fill="x", padx=8, pady=(1, 2))
        ctk.CTkLabel(trow, text="THREAT",
                     font=("Courier New", F_SMALL),
                     text_color=TEXT_DIM).pack(side="left", padx=8)
        self.threat_bar2 = ctk.CTkProgressBar(trow, height=14, corner_radius=4,
                                               fg_color=BG_CARD,
                                               progress_color=ACCENT_GREEN)
        self.threat_bar2.set(0)
        self.threat_bar2.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_threat_score2 = ctk.CTkLabel(trow, text="0  LOW",
                                              font=("Courier New", F_SMALL, "bold"),
                                              text_color=ACCENT_GREEN)
        self.lbl_threat_score2.pack(side="left", padx=(0, 8))

        self._sec_lbl(body, "EVENT  LOG")
        filt_row = ctk.CTkFrame(body, fg_color="transparent")
        filt_row.pack(fill="x", padx=8, pady=(1, 2))
        self.event_filter = ctk.CTkEntry(filt_row, placeholder_text="Filter class...",
                                          width=160, height=30,
                                          fg_color=BG_CARD2,
                                          font=("Courier New", F_SMALL))
        self.event_filter.pack(side="left")
        ctk.CTkButton(filt_row, text="FILTER", width=72, height=30,
                      fg_color=CORNFLOWER_D,
                      font=("Courier New", F_SMALL, "bold"),
                      command=self._apply_event_filter).pack(side="left", padx=4)
        self.event_list = ctk.CTkScrollableFrame(body, fg_color=BG_CARD,
                                                  corner_radius=6, height=180)
        self.event_list.pack(fill="x", padx=8, pady=(1, 6))

        self._sec_lbl(body, "MOTION  HEATMAP")
        self.heatmap_lbl = ctk.CTkLabel(body, text="", fg_color=BG_CARD2,
                                        corner_radius=6, height=100)
        self.heatmap_lbl.pack(fill="x", padx=8, pady=(1, 4))

    def _build_right_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=10)
        panel.grid(row=0, column=2, sticky="nsew")
        self._panel_title(panel, "DETECTIONS & INTELLIGENCE")

        body = ctk.CTkScrollableFrame(
            panel, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BG_CARD2,
            scrollbar_button_hover_color=CORNFLOWER_D)
        body.pack(fill="both", expand=True)
        self._bind_mousewheel(body)

        self._sec_lbl(body, "DETECTION  GALLERY")
        self.det_scroll = ctk.CTkScrollableFrame(body, fg_color=BG_CARD, corner_radius=6, height=190)
        self.det_scroll.pack(fill="both", expand=True, padx=8, pady=(1, 8))
        self._bind_mousewheel(self.det_scroll)

        self._sec_lbl(body, "SYSTEM  TERMINAL")
        self.terminal = ctk.CTkTextbox(body, height=220, fg_color=BG_BASE,
                                        font=("Courier New", F_SMALL),
                                        text_color=TEXT_PRIMARY,
                                        state="disabled")
        self.terminal.pack(fill="x", padx=8, pady=(1, 6))
        self.terminal._textbox.tag_configure("ts",    foreground=TEXT_DIM)
        self.terminal._textbox.tag_configure("good",  foreground=ACCENT_GREEN)
        self.terminal._textbox.tag_configure("warn",  foreground=ACCENT_AMBER)
        self.terminal._textbox.tag_configure("error", foreground=ACCENT_RED)
        self.terminal._textbox.tag_configure("alarm", foreground=ACCENT_RED)
        self.terminal._textbox.tag_configure("info",  foreground=TEXT_PRIMARY)
        self.terminal._textbox.tag_configure("track", foreground=ACCENT_TRACK)
        ctk.CTkButton(body, text="CLEAR TERMINAL", height=28,
                      fg_color=BG_CARD2, hover_color=BG_BORDER,
                      font=("Courier New", F_SMALL),
                      text_color=TEXT_SEC,
                      command=self._clear_terminal).pack(fill="x", padx=8, pady=(0, 8))

        self._sec_lbl(body, "GEOLOCATION  &  NETWORK")
        geo_card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=6)
        geo_card.pack(fill="x", padx=8, pady=(1, 8))
        for lbl_text, attr in [("LOCATION", "lbl_geo_city"),
                                ("COUNTRY",  "lbl_geo_country"),
                                ("COORDS",   "lbl_geo_coords"),
                                ("IP",       "lbl_geo_ip"),
                                ("ISP",      "lbl_geo_isp")]:
            r = ctk.CTkFrame(geo_card, fg_color="transparent")
            r.pack(fill="x", padx=8, pady=1)
            ctk.CTkLabel(r, text=lbl_text, font=("Courier New", F_SMALL),
                         text_color=TEXT_DIM, width=70, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(r, text="-", font=("Courier New", F_SMALL),
                                text_color=TEXT_PRIMARY, anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            setattr(self, attr, lbl)

        self._sec_lbl(body, "POSITION  MAP")
        self.map_canvas = ctk.CTkLabel(body, text="", fg_color=BG_CARD2,
                                        corner_radius=6, height=130)
        self.map_canvas.pack(fill="x", padx=8, pady=(1, 8))

    # Widget factory helpers
    def _chip(self, parent, label, value, color):
        f = ctk.CTkFrame(parent, fg_color=BG_CARD2, border_color=BG_BORDER,
                         border_width=1, corner_radius=6, width=118, height=56)
        f.pack(side="left", padx=5)
        f.pack_propagate(False)
        ctk.CTkLabel(f, text=label, font=("Courier New", F_BODY, "bold"),
                     text_color=TEXT_SEC).pack(pady=(7, 0))
        lbl = ctk.CTkLabel(f, text=value, font=("Courier New", F_LG, "bold"),
                            text_color=color)
        lbl.pack(pady=(0, 7))
        return lbl

    def _hdr_btn(self, parent, text, cmd, color):
        btn = ctk.CTkButton(parent, text=text, width=90, height=32,
                            fg_color=color, hover_color=CORNFLOWER_G,
                            font=("Courier New", F_SMALL, "bold"),
                            text_color=TEXT_PRIMARY, command=cmd)
        btn.pack(side="left", padx=3)
        return btn

    def _panel_title(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=("Courier New", F_MID, "bold"),
                     text_color=CORNFLOWER).pack(anchor="w", padx=12, pady=(8, 4))
        ctk.CTkFrame(parent, fg_color=BG_BORDER, height=1,
                     corner_radius=0).pack(fill="x", padx=8)

    def _stat_box(self, parent, label, value, color, col):
        f = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        f.grid(row=0, column=col, sticky="ew", padx=2, pady=2)
        ctk.CTkLabel(f, text=label, font=("Courier New", F_TINY),
                     text_color=TEXT_DIM).pack(pady=(5, 0))
        v = ctk.CTkLabel(f, text=value, font=("Courier New", F_LG, "bold"),
                         text_color=color)
        v.pack(pady=(0, 5))
        return v

    def _sec_lbl(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=("Courier New", F_SMALL, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=12, pady=(6, 1))

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>",
                    lambda e: widget._parent_canvas.yview_scroll(
                        int(-1 * e.delta / 120), "units")
                    if hasattr(widget, "_parent_canvas") else None)

    # ── Terminal redirect ─────────────────────────────────────────────────
    def _redirect_terminal(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TerminalRedirect(lambda t: self._log(t, "info"))
        sys.stderr = TerminalRedirect(lambda t: self._log(t, "warn"))

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
    # ALARM  (unchanged)
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
            self.model.to(_TORCH_DEVICE)
            model_names = list(getattr(self.model, "names", {}).values())
            self.model_is_weapon_only = bool(model_names) and all(
                is_weapon_class(name) for name in model_names
            )
            self.confidence_threshold = 0.10 if self.model_is_weapon_only else 0.25
            if _TORCH_DEVICE == "cuda":
                try:
                    self.model.model.half()
                    self._log("FP16 half-precision enabled on GPU.", "good")
                except Exception:
                    pass
            self._log(f"Model loaded ({display_name}) -> {_TORCH_DEVICE.upper()}", "good")
            if self.model_is_weapon_only:
                self._log("Weapon model detected: YOLO boxes shown directly at conf=0.10.", "good")
            self.after(0, lambda: self.btn_cam.configure(state="normal"))
            self.after(0, lambda: self.btn_video.configure(state="normal"))
            self.after(0, lambda: self.btn_image.configure(state="normal"))
        except Exception as e:
            self._log(f"Model load failed: {e}", "error")

    def _test_camera(self):
        """Quick camera test button for diagnostics."""
        self._log("Testing camera access...", "info")
        try:
            for idx in [0, 1, 2]:
                for be in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    try:
                        c = cv2.VideoCapture(idx, be)
                        ok = c.isOpened()
                        self._log(f"  idx={idx} backend={be} -> isOpened={ok}", "info")
                        if ok:
                            ret, frame = c.read()
                            self._log(f"  frame read: ret={ret} frame shape={frame.shape if ret else 'N/A'}", "info")
                            if ret:
                                self._log(f"Camera test SUCCESS: idx={idx} backend={be}", "good")
                                # Show a single frame in the feed
                                pil = cv2_to_pil(frame)
                                self.after(0, self._set_feed_image, pil, 0, 0, 0)
                        c.release()
                    except Exception as e:
                        self._log(f"  idx={idx} backend={be} EXCEPTION: {e}", "warn")
        except Exception as e:
            self._log(f"Camera test failed: {e}", "error")

    def _start_camera(self):
        if self.running:
            self._stop_detection()
        self._log("Opening camera...", "info")
        # Try multiple backends and indices for Windows compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        indices = [0, 1, 2]
        cap = None
        for idx in indices:
            for be in backends:
                try:
                    c = cv2.VideoCapture(idx, be)
                    if c.isOpened():
                        cap = c
                        self._log(f"Camera opened: index={idx}, backend={be}", "good")
                        break
                except Exception as e:
                    self._log(f"Camera try index={idx} backend={be} failed: {e}", "warn")
            if cap is not None:
                break
        if cap is None or not cap.isOpened():
            self._log("Cannot access webcam. Check privacy settings and ensure no other app is using the camera.", "error")
            return
        # Set reasonable defaults
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
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
        ok, preview = cap.read()
        if ok and preview is not None:
            self._set_feed_image(cv2_to_pil(preview), 0.0, 0, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self._log(f"Video opened but first frame could not be read: {path}", "error")
            cap.release()
            return
        self.cap           = cap
        self.is_video_file = True
        fname = os.path.basename(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self._log(f"Video loaded: {fname} ({frames} frames @ {fps:.1f} FPS)", "good")
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
        if self.lbl_cam_status is not None:
            self.lbl_cam_status.configure(text=f"● IMAGE: {fname[:28]}", text_color=ACCENT_AMBER)
        threading.Thread(target=self._process_static_image, args=(path,), daemon=True).start()

    def _process_static_image(self, path):
        """
        Static image processing — unchanged from v1.
        Video tracking is meaningful only across frames (video/camera),
        so image mode still uses YOLO's native annotation for simplicity.
        """
        frame = cv2.imread(path)
        if frame is None:
            self._log("Failed to load image file.", "error")
            self.after(0, self._draw_status_dot, ACCENT_RED)
            return
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
            weapon_classes = [c for c in classes_this if is_weapon_class(c)]
            if weapon_classes:
                self.alarm.trigger(self._fire_alarm, weapon_classes)
            elif score >= 40:
                self.alarm.trigger(self._fire_alarm, classes_this)

        full_pil = cv2_to_pil(annotated)
        if confs_this:
            meta = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                    f"IMAGE  |  Objects: {n_det}  |  "
                    f"Classes: {', '.join(set(classes_this)) or '—'}  |  "
                    f"Avg conf: {sum(confs_this)/len(confs_this)*100:.0f}%")
        else:
            meta = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  IMAGE  |  No detections"
        self._gallery.append((full_pil, meta))
        gallery_ref = (full_pil, meta)
        thumb       = full_pil.resize((176, 116), Image.Resampling.LANCZOS)

        self.after(0, self._set_feed_image, full_pil, 0.0, n_det, 0)
        self.after(0, self._insert_thumb_ref, thumb, n_det, classes_this, gallery_ref)
        if n_det:
            self.after(0, self._insert_event, n_det, classes_this, confs_this, [])
        self.after(200, self._draw_status_dot, ACCENT_GREEN if n_det else TEXT_DIM)
        self._log(f"Image analysis complete — {n_det} object(s) detected.",
                  "good" if n_det else "info")

    def _start_common(self, source_label):
        self.running       = True
        self.paused        = False
        self.session_start = time.time()
        self.frame_count   = 0
        self._det_frame_ctr = 0
        self._pause_event.set()
        self._draw_status_dot(ACCENT_GREEN)
        if self.lbl_cam_status is not None:
            self.lbl_cam_status.configure(
                text=f"● {source_label}", text_color=ACCENT_GREEN)
        self.btn_cam.configure(state="disabled")
        self.btn_video.configure(state="disabled")
        self.btn_image.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self._log(f"Detection started — {source_label}.", "good")

        # Reset all tracker state for a clean session.
        # DeepSortWrapper.reset() creates a fresh DeepSort instance,
        # clearing all Kalman filters, track IDs, and appearance galleries.
        if self.model_is_weapon_only:
            self.ds_tracker = None
            self.trajectory_store.reset()
            self.threat_timer.reset()
            self.alarm_dedup.reset()
            self._log("Fast weapon detection mode enabled.", "track")
        elif _DEEPSORT_MODULE_OK:
            try:
                if self.ds_tracker is None:
                    # First session — instantiate the tracker now so the heavy
                    # embedder model is loaded on the inference thread, not GUI thread.
                    self.ds_tracker = DeepSortWrapper(use_gpu=(_TORCH_DEVICE == "cuda"))
                    self._log(
                        f"Tracker initialised "
                        f"[embedder=mobilenet  gpu={_TORCH_DEVICE=='cuda'}]",
                        "track")
                else:
                    # Subsequent session — reset state but reuse loaded embedder.
                    self.ds_tracker.reset()
                    self._log("Tracker reset for new session.", "track")
                self.trajectory_store.reset()
                self.threat_timer.reset()
                self.alarm_dedup.reset()
            except Exception as e:
                self.ds_tracker = None
                self._log(f"Tracking unavailable; using YOLO-only detection: {e}", "warn")
        self._active_track_count = 0

        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _pause_detection(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self._pause_event.clear()
            self.btn_pause.configure(text="▶ RESUME")
            self._draw_status_dot(ACCENT_AMBER)
            self._log("Detection paused.", "warn")
        else:
            self._pause_event.set()
            self.btn_pause.configure(text="⏸ PAUSE")
            self._draw_status_dot(ACCENT_GREEN)
            self._log("Detection resumed.", "good")

    def _stop_detection(self):
        self.running = False
        self.paused  = False
        self._pause_event.set()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.configure(state="normal" if self.model else "disabled")
        self.btn_video.configure(state="normal" if self.model else "disabled")
        self.btn_image.configure(state="normal" if self.model else "disabled")
        self.btn_pause.configure(state="disabled", text="⏸ PAUSE")
        self.btn_stop.configure(state="disabled")
        self._draw_status_dot(ACCENT_RED)
        if self.lbl_cam_status is not None:
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
    # DETECTION LOOP  ── [DEEPSORT-5]  Core tracking integration
    # ═════════════════════════════════════════════════════════════════════
    def _detection_loop(self):
        """
        Main inference + tracking loop.
        """
        try:
            prev_time = time.time()
            reconnect_tries = 0

            while self.running:
                self._pause_event.wait()
                if not self.running:
                    break

                if self.cap is None:
                    self._log("Capture source is not available.", "error")
                    break

                ret, frame = self.cap.read()

                if not ret:
                    if self.is_video_file:
                        self._log("Video playback complete.", "info")
                        break
                    reconnect_tries += 1
                    self._log(f"Frame read failed - reconnect attempt {reconnect_tries}...", "warn")
                    if self.cap:
                        self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(0)
                    if reconnect_tries >= 5:
                        self._log("Camera reconnect failed after 5 attempts.", "error")
                        break
                    continue

                reconnect_tries = 0
                self.frame_count += 1

                if not self.is_video_file:
                    frame = cv2.flip(frame, 1)

                try:
                    self._frame_queue.put_nowait(
                        (cv2_to_pil(frame), self.fps_val, 0, self._active_track_count))
                except queue.Full:
                    pass

                if self.frame_count % self.INFERENCE_EVERY_N_FRAMES != 0:
                    continue

                try:
                    results = self.model(frame, conf=self.confidence_threshold,
                                         verbose=False)
                except Exception as e:
                    self._log(f"Inference error (frame dropped): {e}", "warn")
                    continue

                boxes = results[0].boxes
                raw_classes = []
                raw_confs = []
                if boxes is not None and len(boxes) > 0 and boxes.cls is not None:
                    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                        raw_classes.append(self.model.names[int(cls_id)])
                        raw_confs.append(float(conf))

                raw_dets = (DeepSortWrapper.yolo_to_deepsort(boxes, self.model.names)
                            if (_DEEPSORT_MODULE_OK and self.ds_tracker)
                            else [])

                if _DEEPSORT_MODULE_OK and self.ds_tracker:
                    tracks = self.ds_tracker.update(raw_dets, frame)
                    meta = self.ds_tracker.extract_meta(tracks)
                else:
                    tracks = []
                    meta = {"ids": [], "classes": [], "confs": [],
                            "boxes_ltrb": [], "n": len(boxes) if boxes is not None else 0}
                    meta["classes"] = list(raw_classes)
                    meta["confs"] = list(raw_confs)
                    meta["n"] = len(raw_classes)

                n_tracks = meta["n"]
                classes_this = meta["classes"]
                confs_this = meta["confs"]
                active_ids = meta["ids"]
                display_count = n_tracks

                # Confirmed tracks can lag or be empty for brief weapon
                # detections. For weapon models, show/log raw YOLO boxes immediately.
                if raw_classes and (self.model_is_weapon_only or not classes_this):
                    classes_this = raw_classes
                    confs_this = raw_confs
                    active_ids = []
                    display_count = len(raw_classes)

                if _DEEPSORT_MODULE_OK and self.ds_tracker and n_tracks > 0:
                    annotated = self.ds_tracker.draw(
                        frame.copy(), tracks,
                        self.trajectory_store,
                        self.threat_timer,
                        show_confidence=True,
                        show_trajectory=True,
                    )
                else:
                    annotated = results[0].plot()

                cur = time.time()
                fps = 1.0 / (cur - prev_time) if prev_time else 0
                prev_time = cur
                self.fps_val = fps
                self.fps_history.append(fps)

                n_det = len(boxes) if boxes is not None else 0
                self.det_history.append(display_count)

                for cls_name, conf in zip(classes_this, confs_this):
                    self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + 1
                    self.confidence_vals.append(float(conf))

                if boxes is not None and boxes.xyxyn is not None and len(boxes.xyxyn) > 0:
                    self.heatmap.add(boxes.xyxyn.tolist())

                weapon_alarm_classes = []
                for tid, cls_name in zip(active_ids, classes_this):
                    if is_weapon_class(cls_name):
                        self.threat_timer.seen(tid, cls_name)
                        if (self.threat_timer.is_persistent(tid, cls_name)
                                and self.alarm_dedup.should_alarm(tid, cls_name)):
                            weapon_alarm_classes.append(cls_name)
                            self.alarm_dedup.record(tid, cls_name)
                            elapsed = self.threat_timer.elapsed(tid, cls_name)
                            self._log(
                                f"WEAPON TRACK #{tid} class={cls_name} "
                                f"persisted {elapsed:.1f}s -> ALARM",
                                "alarm")

                if self.model_is_weapon_only and raw_classes:
                    weapon_alarm_classes.extend(
                        cls for cls in raw_classes if is_weapon_class(cls)
                    )

                self.threat_timer.clean_stale(active_ids)
                self.alarm_dedup.clean_stale(active_ids)

                if classes_this:
                    self.log_eng.record(classes_this, confs_this)
                    score = self.log_eng.threat_score()
                    if weapon_alarm_classes:
                        self.alarm.trigger(self._fire_alarm, weapon_alarm_classes)
                    elif score >= 40:
                        self.alarm.trigger(self._fire_alarm, classes_this)

                self._active_track_count = display_count if self.model_is_weapon_only else n_tracks

                if classes_this:
                    self._det_frame_ctr += 1
                    if self._det_frame_ctr % self.THUMB_EVERY_N_DETECTIONS == 0:
                        full_pil = cv2_to_pil(annotated)
                        avg_conf = (sum(confs_this) / len(confs_this) * 100
                                    if confs_this else 0)
                        meta_str = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                                    f"Detections: {display_count}  |  "
                                    f"Classes: {', '.join(set(classes_this))}  |  "
                                    f"Avg conf: {avg_conf:.0f}%")
                        gallery_ref = (full_pil, meta_str)
                        self._gallery.append(gallery_ref)
                        thumb = full_pil.resize((176, 116), Image.Resampling.LANCZOS)
                        self.after(0, self._insert_thumb_ref, thumb, display_count,
                                   classes_this, gallery_ref)
                        self.after(0, self._insert_event, display_count, classes_this,
                                   confs_this, active_ids)

                feed_pil = cv2_to_pil(annotated)
                try:
                    while True:
                        self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait((feed_pil, fps, n_det, display_count))
                except queue.Full:
                    pass

            self._log("Detection loop ended.", "info")
            self.after(0, self._stop_detection)
        except Exception as loop_err:
            self._log(f"Detection loop error: {loop_err}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
            self.after(0, self._stop_detection)

    # Decoupled 30 fps feed renderer
    def _schedule_feed_render(self):
        try:
            item = self._frame_queue.get_nowait()
            # [DEEPSORT-8] Queue now carries 4 values: pil, fps, n_det, n_tracks
            if len(item) == 4:
                pil_img, fps, n_det, n_tracks = item
            else:
                pil_img, fps, n_det = item
                n_tracks = n_det
            self._set_feed_image(pil_img, fps, n_det, n_tracks)
        except queue.Empty:
            pass
        self.after(33, self._schedule_feed_render)

    # ═════════════════════════════════════════════════════════════════════
    # FEED IMAGE RENDERER  — [DEEPSORT-8] tracks count displayed
    # ═════════════════════════════════════════════════════════════════════
    def _set_feed_image(self, pil_img, fps, n_det, n_tracks=0):
        W = max(self.feed_container.winfo_width(),  480)
        H = max(self.feed_container.winfo_height(), 400)
        iw, ih = pil_img.size
        scale  = min(W / iw, H / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
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
        draw.text((9, 30), f"DET: {n_det}",
                  fill=ACCENT_RED if n_det else CORNFLOWER_L)
        # [DEEPSORT-8] Additional overlay line: confirmed track count
        draw.rectangle([6, 50, 185, 70], fill="#00000099")
        draw.text((9, 52), f"OBJECTS: {n_tracks}",
                  fill="#00DCAA" if n_tracks else TEXT_DIM)
        ts_txt = datetime.now().strftime("%H:%M:%S")
        draw.rectangle([W-90, 6,  W-2, 26], fill="#00000099")
        draw.text((W-88, 8),  ts_txt, fill=TEXT_SEC)
        src = "VIDEO" if self.is_video_file else "LIVE"
        draw.rectangle([W-80, 28, W-2, 48], fill="#00000099")
        draw.text((W-78, 30), src, fill=CORNFLOWER_L)
        draw.rectangle([W-112, 50, W-2, 70], fill="#00000088")
        draw.text((W-110, 52), "DETECTION", fill=ACCENT_TRACK)
        ci = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(W, H))
        self.feed_label.configure(image=ci, text="")
        self.feed_label._image = ci
        if self.lbl_det_live is not None:
            self.lbl_det_live.configure(
                text=f"OBJECTS: {n_tracks}",
                text_color=ACCENT_TRACK if n_tracks else TEXT_DIM)
        if self.lbl_ts is not None:
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
        # Use scrollable frame's inner frame for grid layout (fallback for older CTk)
        try:
            inner = self.det_scroll._scrollable_frame
        except AttributeError:
            try:
                inner = self.det_scroll._frame
            except AttributeError:
                inner = self.det_scroll
        lbl = ctk.CTkLabel(inner, image=ci, text="",
                           fg_color="transparent", cursor="hand2")
        lbl._img = ci
        lbl.bind("<Button-1>",
                 lambda e, ref=gallery_entry: self._open_lightbox_ref(ref))
        lbl.grid(row=self.det_thumb_row, column=self.det_thumb_col, padx=2, pady=2)
        self.det_thumb_col += 1
        if self.det_thumb_col >= 2:
            self.det_thumb_col = 0
            self.det_thumb_row += 1
        children = inner.winfo_children()
        if len(children) > 24:
            children[0].destroy()
        # Configure grid columns for the inner frame
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

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
    # EVENT LOG  — [DEEPSORT-10] shows track IDs in entries
    # ═════════════════════════════════════════════════════════════════════
    def _insert_event(self, n_det, classes, confs, track_ids=None):
        # [DEEPSORT-10] track_ids is the list of confirmed track IDs
        ts    = datetime.now().strftime("%H:%M:%S")
        score = self.log_eng.threat_score()
        self._all_events.append({
            "ts":        ts,
            "n":         n_det,
            "classes":   classes,
            "confs":     confs,
            "score":     score,
            "track_ids": track_ids or [],   # [DEEPSORT-10]
        })
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
            # [DEEPSORT-10] Show track IDs in the event row
            if ev.get("track_ids"):
                id_str = "  IDs:" + ",".join(f"#{i}" for i in ev["track_ids"][:4])
                ctk.CTkLabel(row, text=id_str,
                             font=("Courier New", F_TINY),
                             text_color=ACCENT_TRACK).pack(side="left")
            ctk.CTkLabel(row, text=f"T:{ev['score']}",
                         font=("Courier New", F_SMALL, "bold"),
                         text_color=tc).pack(side="right", padx=8)

    # ═════════════════════════════════════════════════════════════════════
    # PERIODIC REFRESH  (500 ms)
    # ═════════════════════════════════════════════════════════════════════
    def _schedule_ui_updates(self):
        self._flush_log_queue()
        if self._geo_dirty:
            self._refresh_geo_panel()
            self._geo_dirty = False
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
        # [DEEPSORT-9] Update active tracks stat box + header chip
        self.stat_tracks.configure(text=str(self._active_track_count))
        self.lbl_tracks_hdr.configure(text=str(self._active_track_count))
        self.lbl_fps_hdr.configure(text=f"{self.fps_val:.1f}")
        self.lbl_frames_hdr.configure(text=str(self.frame_count))
        self.lbl_dets_hdr.configure(text=str(total_det))

    def _update_threat_bar(self):
        score = self.log_eng.threat_score()
        if self.threat_bar is not None:
            self.threat_bar.set(score / 100)
        self.threat_bar2.set(score / 100)
        tc = threat_color(score)
        if self.threat_bar is not None:
            self.threat_bar.configure(progress_color=tc)
        self.threat_bar2.configure(progress_color=tc)
        label = ("CRITICAL" if score >= 75 else
                 "HIGH"     if score >= 50 else
                 "MODERATE" if score >= 25 else "LOW")
        if self.lbl_threat_score is not None:
            self.lbl_threat_score.configure(text=f"{score}  {label}", text_color=tc)
        self.lbl_threat_score2.configure(text=f"{score}  {label}", text_color=tc)
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
        r_new = min(255, r + (255 - r) // 3)
        g_new = min(255, g + (255 - g) // 3)
        b_new = min(255, b + (255 - b) // 3)
        fill_hex = "#{:02x}{:02x}{:02x}".format(r_new, g_new, b_new)
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
            # Highlight weapon classes in red
            lbl_color = ACCENT_RED if is_weapon_class(cls) else TEXT_PRIMARY
            ctk.CTkLabel(row, text=cls.upper(),
                         font=("Courier New", F_BODY, "bold"),
                         text_color=lbl_color, width=100,
                         anchor="w").pack(side="left", padx=4)
            bar = ctk.CTkProgressBar(row, height=10, corner_radius=3,
                                     fg_color=BG_BASE,
                                     progress_color=ACCENT_RED if is_weapon_class(cls)
                                         else CORNFLOWER)
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
        self._pause_event.set()
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
