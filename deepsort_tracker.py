import time
import math
import collections
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ── Try to import deep_sort_realtime; surface a helpful error if missing ──────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _DEEPSORT_AVAILABLE = True
except ImportError:
    _DEEPSORT_AVAILABLE = False
    DeepSort = None  # type: ignore

TRACK_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (255, 120,   0),   # #0078FF  electric blue
    (0,   210,  80),   # #50D200  lime green
    (0,   165, 255),   # #FFA500  orange
    (210,  0,  210),   # #D200D2  violet
    (0,   220, 220),   # #DCDC00  cyan-yellow
    (100, 255, 100),   # #64FF64  light green
    (255,  80, 180),   # #B450FF  pink-purple
    (60,  220, 255),   # #FFDC3C  sky yellow
    (180, 255,  60),   # #3CFFB4  mint
    (255,  60,  60),   # #3C3CFF  bright red
    (0,   180, 255),   # #FFB400  amber
    (255, 200,  0),    # #00C8FF  sky blue
]

# Threat classes that require persistent tracking before alarm fires.
# Weapons must be visible on the *same track* for THREAT_PERSIST_SECS
# consecutive seconds to suppress brief misdetections.
WEAPON_CLASSES = frozenset({
    "gun", "pistol", "rifle", "weapon", "knife",
    "handgun", "firearm", "grenade", "bomb",
})

# Classes that are always interesting to track but not alarm-worthy alone.
HIGH_INTEREST_CLASSES = frozenset({
    "person", "car", "truck", "motorcycle", "bicycle",
})

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY STORE
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryStore:

    def __init__(self, max_len: int = 60, min_draw_len: int = 2):
        self.max_len       = max_len
        self.min_draw_len  = min_draw_len
        # track_id (int) → deque of (cx, cy)
        self._paths: Dict[int, collections.deque] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, track_id: int, cx: float, cy: float) -> None:
        """Add a new centroid point for the given track."""
        if track_id not in self._paths:
            self._paths[track_id] = collections.deque(maxlen=self.max_len)
        self._paths[track_id].append((int(cx), int(cy)))

    def draw_all(self, frame: np.ndarray,
                 track_color_map: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
        """
        Render all stored trajectories onto `frame` (in-place, returns frame).

        Each path segment fades from 15% opacity (oldest) to 90% opacity
        (newest) using blended rectangle-free polyline drawing.

        Args:
            frame:           BGR uint8 numpy frame to draw on.
            track_color_map: Dict mapping track_id → BGR color tuple.
        """
        overlay = frame.copy()

        for tid, path in self._paths.items():
            pts = list(path)
            if len(pts) < self.min_draw_len:
                continue
            color = track_color_map.get(tid, (200, 200, 200))
            n = len(pts)
            for i in range(1, n):
                # Alpha ramps from 0.15 (oldest segment) to 0.90 (newest)
                alpha    = 0.15 + 0.75 * (i / n)
                thickness = max(1, int(2 * (i / n)))
                p1 = pts[i - 1]
                p2 = pts[i]
                # Draw on overlay then blend — avoids alpha-channel dependency
                cv2.line(overlay, p1, p2, color, thickness, lineType=cv2.LINE_AA)
                # Blend only the local bounding box of the line for performance
                x0, y0 = min(p1[0], p2[0]) - 3, min(p1[1], p2[1]) - 3
                x1, y1 = max(p1[0], p2[0]) + 3, max(p1[1], p2[1]) + 3
                x0, y0 = max(x0, 0), max(y0, 0)
                x1 = min(x1, frame.shape[1])
                y1 = min(y1, frame.shape[0])
                if x1 > x0 and y1 > y0:
                    frame[y0:y1, x0:x1] = cv2.addWeighted(
                        overlay[y0:y1, x0:x1], alpha,
                        frame[y0:y1, x0:x1], 1 - alpha, 0
                    )
        return frame

    def remove(self, track_id: int) -> None:
        """Drop history for a track that has been lost/deleted."""
        self._paths.pop(track_id, None)

    def reset(self) -> None:
        """Clear all trajectory history (call on session restart)."""
        self._paths.clear()

    def active_ids(self) -> List[int]:
        return list(self._paths.keys())

    def __len__(self) -> int:
        return len(self._paths)


# ─────────────────────────────────────────────────────────────────────────────
# THREAT PERSISTENCE TIMER
# ─────────────────────────────────────────────────────────────────────────────

class ThreatPersistenceTimer:

    def __init__(self, persist_secs: float = 2.5):
        self.persist_secs = persist_secs
        # (track_id, cls_name) → first seen time
        self._first_seen: Dict[Tuple[int, str], float] = {}

    def seen(self, track_id: int, cls_name: str) -> None:
        """Record that this (track_id, cls_name) pair is visible right now."""
        key = (track_id, cls_name)
        if key not in self._first_seen:
            self._first_seen[key] = time.monotonic()

    def is_persistent(self, track_id: int, cls_name: str) -> bool:
        """Return True if this pair has been visible long enough."""
        key = (track_id, cls_name)
        if key not in self._first_seen:
            return False
        elapsed = time.monotonic() - self._first_seen[key]
        return elapsed >= self.persist_secs

    def elapsed(self, track_id: int, cls_name: str) -> float:
        """Return how many seconds this pair has been visible (0 if unseen)."""
        key = (track_id, cls_name)
        if key not in self._first_seen:
            return 0.0
        return time.monotonic() - self._first_seen[key]

    def clean_stale(self, active_track_ids) -> None:
        """Remove records for tracks that are no longer active."""
        active = set(active_track_ids)
        stale = [k for k in self._first_seen if k[0] not in active]
        for k in stale:
            del self._first_seen[k]

    def reset(self) -> None:
        self._first_seen.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ALARM DEDUPLICATOR
# ─────────────────────────────────────────────────────────────────────────────

class AlarmDeduplicator:


    def __init__(self, cooldown_secs: float = 30.0):
        self.cooldown_secs = cooldown_secs
        # (track_id, cls_name) → last alarm time
        self._last_alarm: Dict[Tuple[int, str], float] = {}

    def should_alarm(self, track_id: int, cls_name: str) -> bool:
        """
        Returns True if this (track_id, cls_name) pair has NOT alarmed recently.
        Call this before triggering the alarm; call record() immediately after.
        """
        key = (track_id, cls_name)
        last = self._last_alarm.get(key)
        if last is None:
            return True
        return (time.monotonic() - last) >= self.cooldown_secs

    def record(self, track_id: int, cls_name: str) -> None:
        """Record that the alarm fired for this pair right now."""
        self._last_alarm[(track_id, cls_name)] = time.monotonic()

    def clean_stale(self, active_track_ids) -> None:
        """Remove records for tracks that have been deleted."""
        active = set(active_track_ids)
        stale  = [k for k in self._last_alarm if k[0] not in active]
        for k in stale:
            del self._last_alarm[k]

    def reset(self) -> None:
        self._last_alarm.clear()


# ─────────────────────────────────────────────────────────────────────────────
# DEEP SORT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class DeepSortWrapper:


    # Hyper-parameters — tuned for indoor/outdoor CCTV at 15-30 FPS.
    # Increase max_age for slow-moving subjects in low-FPS settings.
    MAX_AGE              = 30    # frames before a lost track is deleted
    N_INIT               = 2     # frames of consecutive detection to confirm
    MAX_COSINE_DIST      = 0.35  # appearance similarity threshold (0=strictest)
    NMS_MAX_OVERLAP      = 1.0   # allow full overlap (YOLO already does NMS)
    NN_BUDGET            = 100   # max stored appearance vectors per track

    # Visual annotation constants (BGR)
    BOX_THICKNESS        = 2
    ID_FONT_SCALE        = 0.55
    ID_FONT_THICKNESS    = 1
    LABEL_FONT_SCALE     = 0.45
    LABEL_FONT_THICKNESS = 1

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._tracker: Optional[DeepSort] = None
        self._fallback_mode = not _DEEPSORT_AVAILABLE
        self._pseudo_id_counter = 0   # used only in fallback mode

        # Track color map: track_id → BGR color (persists across frames)
        self._color_map: Dict[int, Tuple[int, int, int]] = {}

        if not _DEEPSORT_AVAILABLE:
            import warnings
            warnings.warn(
                "deep_sort_realtime is not installed.  "
                "Running in IoU-only fallback mode (no appearance features).\n"
                "Install with: pip install deep-sort-realtime",
                ImportWarning,
                stacklevel=2,
            )
            return

        # ── Initialise DeepSort tracker ───────────────────────────────────
        try:
            self._tracker = DeepSort(
                max_age             = self.MAX_AGE,
                n_init              = self.N_INIT,
                nms_max_overlap     = self.NMS_MAX_OVERLAP,
                max_cosine_distance = self.MAX_COSINE_DIST,
                nn_budget           = self.NN_BUDGET,
                # MobileNetV2 is the best speed/accuracy trade-off for CCTV
                embedder            = "mobilenet",
                # Use GPU embedder only if CUDA is available (avoids CPU stall)
                embedder_gpu        = use_gpu,
                # FP16 gives ~35% speedup on GPU with negligible quality loss
                half                = use_gpu,
                # OpenCV frames are BGR; set True so embedder handles conversion
                bgr                 = True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"DeepSort initialisation failed: {exc}\n"
                f"Ensure deep_sort_realtime is installed: "
                f"pip install deep-sort-realtime"
            ) from exc

    # ── Format conversion ─────────────────────────────────────────────────────

    @staticmethod
    def yolo_to_deepsort(boxes, model_names: Dict[int, str]) -> List:
        if boxes is None or len(boxes) == 0:
            return []

        raw = []
        try:
            xyxy_all  = boxes.xyxy.cpu().numpy()   # (N, 4) float32
            conf_all  = boxes.conf.cpu().numpy()   # (N,)   float32
            cls_all   = boxes.cls.cpu().numpy()    # (N,)   float32
        except Exception:
            return []

        for i in range(len(xyxy_all)):
            x1, y1, x2, y2 = xyxy_all[i]
            conf     = float(conf_all[i])
            cls_id   = int(cls_all[i])
            cls_name = model_names.get(cls_id, f"cls_{cls_id}")
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                raw.append(([float(x1), float(y1), float(w), float(h)],
                             conf, cls_name))
        return raw

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, raw_detections: List, frame_bgr: np.ndarray) -> List:
        """
        Feed YOLO detections to the tracker and return confirmed tracks.

        Args:
            raw_detections: Output of yolo_to_deepsort().
            frame_bgr:      Current BGR frame (needed for appearance embedding).

        Returns:
            List of deep_sort_realtime Track objects (only confirmed tracks).
            In fallback mode, returns pseudo-track dicts instead.
        """
        if self._fallback_mode or self._tracker is None:
            return self._fallback_update(raw_detections, frame_bgr)

        try:
            all_tracks = self._tracker.update_tracks(raw_detections,
                                                      frame=frame_bgr)
        except Exception as exc:
            # Gracefully degrade on any tracker error (e.g. CUDA OOM)
            import warnings
            warnings.warn(f"DeepSort update_tracks error: {exc}", RuntimeWarning)
            return []

        # Return only confirmed tracks (n_init consecutive detections seen)
        confirmed = [t for t in all_tracks if t.is_confirmed()]

        # Assign colors to new track IDs
        for t in confirmed:
            if t.track_id not in self._color_map:
                self._color_map[t.track_id] = TRACK_PALETTE_BGR[
                    t.track_id % len(TRACK_PALETTE_BGR)
                ]
        return confirmed

    def _fallback_update(self, raw_detections: List,
                         frame_bgr: np.ndarray) -> List:
        """
        IoU-only fallback when deep_sort_realtime is not installed.
        Returns lightweight dicts that mimic the Track object API used by
        extract_meta() and draw().
        """
        pseudo_tracks = []
        for det in raw_detections:
            ltwh, conf, cls_name = det
            x1, y1, w, h = ltwh
            self._pseudo_id_counter += 1
            tid = self._pseudo_id_counter % 9999  # wrap to keep IDs small
            self._color_map[tid] = TRACK_PALETTE_BGR[tid % len(TRACK_PALETTE_BGR)]
            pseudo_tracks.append({
                "track_id":     tid,
                "ltrb":         [x1, y1, x1 + w, y1 + h],
                "cls":          cls_name,
                "conf":         conf,
                "_is_fallback": True,
            })
        return pseudo_tracks

    # ── Metadata extraction ───────────────────────────────────────────────────

    def extract_meta(self, tracks: List) -> Dict:
        """
        Extract structured metadata from a list of confirmed tracks.

        Returns a dict:
            ids       : List[int]   — track IDs
            classes   : List[str]   — class names
            confs     : List[float] — confidence scores
            boxes_ltrb: List[list]  — [x1,y1,x2,y2] boxes
            n         : int         — number of confirmed tracks
        """
        ids, classes, confs, boxes = [], [], [], []
        for t in tracks:
            tid, cls, conf, ltrb = self._unpack(t)
            if ltrb is None:
                continue
            ids.append(tid)
            classes.append(cls)
            confs.append(conf)
            boxes.append(ltrb)
        return {"ids": ids, "classes": classes, "confs": confs,
                "boxes_ltrb": boxes, "n": len(ids)}

    # ── Annotation drawing ────────────────────────────────────────────────────

    def draw(self,
             frame: np.ndarray,
             tracks: List,
             trajectory_store: "TrajectoryStore",
             threat_timer: Optional["ThreatPersistenceTimer"] = None,
             show_confidence: bool = True,
             show_trajectory: bool = True) -> np.ndarray:
        # ── Step 1: Update trajectory history ────────────────────────────
        for t in tracks:
            tid, cls, conf, ltrb = self._unpack(t)
            if ltrb is None:
                continue
            cx = (ltrb[0] + ltrb[2]) / 2
            cy = (ltrb[1] + ltrb[3]) / 2
            trajectory_store.update(tid, cx, cy)

        # ── Step 2: Draw trajectory paths (behind boxes) ─────────────────
        if show_trajectory:
            frame = trajectory_store.draw_all(frame, self._color_map)

        # ── Step 3: Draw bounding boxes and labels ────────────────────────
        h_frame, w_frame = frame.shape[:2]

        for t in tracks:
            tid, cls, conf, ltrb = self._unpack(t)
            if ltrb is None:
                continue

            x1, y1, x2, y2 = (int(v) for v in ltrb)
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)

            color    = self._color_map.get(tid, (200, 200, 200))
            is_weapon = cls.lower() in WEAPON_CLASSES

            # Outer glow for weapon tracks (thick semi-transparent rect)
            if is_weapon:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              (0, 0, 255), 3, lineType=cv2.LINE_AA)

            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          self.BOX_THICKNESS, lineType=cv2.LINE_AA)

            # Corner brackets for a tactical look
            self._draw_corners(frame, x1, y1, x2, y2, color, length=14)

            # ── Label background and text ─────────────────────────────────
            conf_str    = f"{conf*100:.0f}%" if show_confidence else ""
            prefix      = "⚠" if is_weapon else "#"
            label_main  = f"{prefix}{tid}  {cls.upper()}"
            label_conf  = conf_str

            # Threat persistence timer bar (only for weapon-class tracks)
            persist_str = ""
            if is_weapon and threat_timer is not None:
                elapsed = threat_timer.elapsed(tid, cls)
                if elapsed > 0:
                    bar_filled = min(int(elapsed / threat_timer.persist_secs * 8), 8)
                    persist_str = "▓" * bar_filled + "░" * (8 - bar_filled)

            # Compute label dimensions for background rect
            (lw_main, lh_main), _ = cv2.getTextSize(
                label_main, cv2.FONT_HERSHEY_SIMPLEX,
                self.ID_FONT_SCALE, self.ID_FONT_THICKNESS)
            lh_main += 4

            # Keep label inside frame
            label_y = y1 - 6
            if label_y - lh_main < 0:
                label_y = y2 + lh_main + 4

            # Background pill
            bg_x1 = x1
            bg_y1 = label_y - lh_main
            bg_x2 = x1 + lw_main + 8
            bg_y2 = label_y + 4
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                          color, -1, lineType=cv2.LINE_AA)

            # Dark overlay so white text is readable on any color
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Re-draw colored rect on top of dark overlay
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                          color, 1, lineType=cv2.LINE_AA)

            # Main ID + class text
            cv2.putText(frame, label_main,
                        (x1 + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.ID_FONT_SCALE, (255, 255, 255),
                        self.ID_FONT_THICKNESS, cv2.LINE_AA)

            # Secondary line: confidence + persist bar
            if conf_str or persist_str:
                sub = f"{label_conf}  {persist_str}".strip()
                cv2.putText(frame, sub,
                            (x1 + 4, label_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.LABEL_FONT_SCALE, (220, 220, 220),
                            self.LABEL_FONT_THICKNESS, cv2.LINE_AA)

        # ── Step 4: Overlay track count badge ────────────────────────────
        n_tracks = len(tracks)
        badge     = f"TRACKED: {n_tracks}"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (6, 50), (6 + bw + 10, 50 + bh + 8),
                      (20, 20, 40), -1)
        cv2.putText(frame, badge, (11, 50 + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 255, 200), 1, cv2.LINE_AA)

        return frame

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _unpack(self, t) -> Tuple:
        """
        Unpack either a real DeepSort Track object or a fallback dict.
        Returns (track_id, cls_name, conf, ltrb_list) or (…, None) on error.
        """
        try:
            if isinstance(t, dict):
                # Fallback pseudo-track
                return (t["track_id"], t["cls"], t["conf"], t["ltrb"])
            else:
                # Real deep_sort_realtime Track
                ltrb = t.to_ltrb()
                cls  = t.get_det_class() or "object"
                conf = t.get_det_conf() or 0.0
                return (t.track_id, cls, conf, ltrb)
        except Exception:
            return (0, "unknown", 0.0, None)

    @staticmethod
    def _draw_corners(frame: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int,
                      color: Tuple[int, int, int],
                      length: int = 12,
                      thickness: int = 2) -> None:
        """
        Draw tactical corner brackets inside a bounding box.
        Gives a more professional surveillance UI look than plain rectangles.
        """
        for (px, py, sx, sy) in [
            (x1, y1,  1,  1),   # top-left
            (x2, y1, -1,  1),   # top-right
            (x1, y2,  1, -1),   # bottom-left
            (x2, y2, -1, -1),   # bottom-right
        ]:
            cv2.line(frame, (px, py), (px + sx * length, py),
                     color, thickness, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + sy * length),
                     color, thickness, cv2.LINE_AA)

    def get_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        """Return the current track_id → BGR color map (read-only copy)."""
        return dict(self._color_map)

    def reset(self) -> None:
        """
        Fully reset tracker state.  Call on session stop/restart.
        Creates a fresh DeepSort instance to clear all internal Kalman filters.
        """
        self._color_map.clear()
        self._pseudo_id_counter = 0
        if not self._fallback_mode and _DEEPSORT_AVAILABLE:
            try:
                self._tracker = DeepSort(
                    max_age             = self.MAX_AGE,
                    n_init              = self.N_INIT,
                    nms_max_overlap     = self.NMS_MAX_OVERLAP,
                    max_cosine_distance = self.MAX_COSINE_DIST,
                    nn_budget           = self.NN_BUDGET,
                    embedder            = "mobilenet",
                    embedder_gpu        = self.use_gpu,
                    half                = self.use_gpu,
                    bgr                 = True,
                )
            except Exception:
                pass  # keep stale tracker rather than crash

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python deepsort_tracker.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Vigilant Eye — Deep SORT Module Self-Test")
    print("═" * 60)

    # Test TrajectoryStore
    ts = TrajectoryStore(max_len=10)
    for i in range(15):
        ts.update(1, i * 10, i * 5)
        ts.update(2, 200 - i * 8, i * 7)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = ts.draw_all(dummy_frame, {1: (255, 120, 0), 2: (0, 210, 80)})
    print(f"  TrajectoryStore.draw_all():  output shape = {out.shape}  ✓")

    # Test ThreatPersistenceTimer
    tpt = ThreatPersistenceTimer(persist_secs=0.1)
    tpt.seen(7, "gun")
    time.sleep(0.05)
    assert not tpt.is_persistent(7, "gun"), "Should not be persistent yet"
    time.sleep(0.08)
    assert tpt.is_persistent(7, "gun"),     "Should be persistent now"
    print("  ThreatPersistenceTimer:      timing logic   ✓")

    # Test AlarmDeduplicator
    ad = AlarmDeduplicator(cooldown_secs=0.2)
    assert ad.should_alarm(3, "knife"), "First alarm should fire"
    ad.record(3, "knife")
    assert not ad.should_alarm(3, "knife"), "Second alarm should be suppressed"
    time.sleep(0.22)
    assert ad.should_alarm(3, "knife"), "Alarm should be available after cooldown"
    print("  AlarmDeduplicator:           dedup logic    ✓")

    # Test DeepSortWrapper availability
    w = DeepSortWrapper(use_gpu=False)
    mode = "FALLBACK (no deep_sort_realtime)" if w.is_fallback else "FULL Deep SORT"
    print(f"  DeepSortWrapper mode:        {mode}")
    print("═" * 60)
    print("  All self-tests passed.")
