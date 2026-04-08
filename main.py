"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          COW TRACKER — Detection · Tracking · Count                         ║
║          YOLOv8 + ByteTrack + Supervision                                    ║
║          Author : Muhammad Qadeer | AI/ML Engineer · GIFT University         ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO USE
──────────
1.  Install dependencies (once):
        pip install ultralytics supervision lapx opencv-python

2.  Edit CONFIG section below — set your weights path, video path,
    and confidence threshold.

3.  Run:
        python main.py                          # video file (ROI selector opens)
        python main.py --source 0               # webcam
        python main.py --source video.mp4 --show
        python main.py --source video.mp4 --no-save

ROI SELECTION
─────────────
•  A popup window shows the first frame.
•  LEFT-CLICK  to add polygon points (shown as green dots).
•  RIGHT-CLICK to remove the last point.
•  Press ENTER / SPACE to confirm the polygon.
•  Press R to reset all points.
•  Press ESC to skip ROI and process the full frame.
•  Need ≥ 3 points to form a valid polygon.

OUTPUTS
───────
•  Annotated video  (cow_tracked_<stem>.mp4)
•  Per-frame CSV    (cow_log_<stem>.csv)
•  Terminal summary
"""

from __future__ import annotations

# ══ stdlib ════════════════════════════════════════════════════════════════════
import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ══ third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np

try:
    import supervision as sv
    from supervision import (
        ByteTrack, Color, ColorPalette, Detections, LabelAnnotator,
        PolygonZone, PolygonZoneAnnotator, RoundBoxAnnotator, TraceAnnotator,
    )
except ImportError:
    sys.exit("❌  supervision not found.  Run:  pip install supervision lapx")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌  ultralytics not found.  Run:  pip install ultralytics")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIGURATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "weights"    : "weights/best.pt",
    "conf"       : 0.25,          # ← lowered for better recall
    "iou"        : 0.40,
    "imgsz"      : 640,
    "device"     : "",            # "" = auto, "cpu", "0", "cuda:0"

    # ── Source ───────────────────────────────────────────────────────────────
    "source"     : "input/i4.mp4",

    # ── ByteTracker ──────────────────────────────────────────────────────────
    "track_thresh"      : 0.25,
    "track_buffer"      : 60,     # ← longer buffer → fewer ID switches
    "match_thresh"      : 0.85,
    "min_consec_frames" : 2,

    # ── ROI Polygon Zone ──────────────────────────────────────────────────────
    # Leave "points" as None to trigger the interactive selector at startup.
    # Or supply a hard-coded array to skip the selector.
    "polygon_zone": {
        "enabled" : True,
        "points"  : None,         # None → interactive selector opens
        # "points": np.array([(311,182),(511,182),(525,311),(315,314)], dtype=np.int64),
        "label"   : "Farm Area",
    },

    # ── Output ───────────────────────────────────────────────────────────────
    "show"       : False,
    "save"       : True,
    "save_csv"   : True,
    "output_dir" : ".",

    # ── Visual ───────────────────────────────────────────────────────────────
    "trace_length" : 40,
    "hud_alpha"    : 0.60,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        COLOUR PALETTE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PAL = ColorPalette.from_hex([
    "#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
    "#0891B2", "#BE185D", "#15803D", "#B45309", "#1D4ED8",
])

C_ZONE   = Color(r=0,   g=220, b=120)
C_HUD_BG = (10,  10,  10)
C_WHITE  = (255, 255, 255)
C_CYAN   = (0,   230, 230)
C_GREEN  = (50,  230, 100)
C_AMBER  = (0,   180, 255)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   INTERACTIVE ROI SELECTOR                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def select_roi_interactive(frame: np.ndarray, window_title: str = "ROI Selector") -> np.ndarray | None:
    """
    Opens a popup window on `frame`.
    User clicks to place polygon vertices.
    Returns np.array of shape (N,2) or None if skipped.
    """
    MAX_DIM   = 900          # max side length for the selector window
    h, w      = frame.shape[:2]
    scale     = min(MAX_DIM / w, MAX_DIM / h, 1.0)
    disp_w    = int(w * scale)
    disp_h    = int(h * scale)
    disp_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    points: list[tuple[int, int]] = []
    skipped = [False]
    mouse_events = [0]

    HELP = [
        "LEFT-CLICK : add point",
        "RIGHT-CLICK: remove last",
        "R          : reset",
        "ENTER/SPACE/C: confirm (need >= 3 pts)",
        "ESC/S      : skip ROI (full frame)",
        "D          : draw rectangle fallback",
        "M          : manual points input",
    ]

    def _redraw():
        vis = disp_frame.copy()

        # ── semi-transparent polygon fill ──────────────────────────────────
        if len(points) >= 3:
            pts_arr = np.array(points, dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts_arr], (0, 180, 80))
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            cv2.polylines(vis, [pts_arr], isClosed=True,
                          color=(0, 230, 100), thickness=2)

        # ── edges (connecting lines between points) ────────────────────────
        for i in range(len(points) - 1):
            cv2.line(vis, points[i], points[i + 1], (0, 200, 80), 1)

        # ── dots + index labels ────────────────────────────────────────────
        for idx, (px, py) in enumerate(points):
            cv2.circle(vis, (px, py), 6, (0, 230, 100), -1)
            cv2.circle(vis, (px, py), 6, (255, 255, 255), 1)
            cv2.putText(vis, str(idx + 1), (px + 8, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # ── help panel ────────────────────────────────────────────────────
        panel_x, panel_y = 10, 10
        line_h = 18
        panel_h = len(HELP) * line_h + 20
        panel_w = 290
        overlay2 = vis.copy()
        cv2.rectangle(overlay2, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.65, vis, 0.35, 0, vis)
        for i, txt in enumerate(HELP):
            cv2.putText(vis, txt,
                        (panel_x + 8, panel_y + 16 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # ── point count badge ─────────────────────────────────────────────
        badge = f"Points: {len(points)}"
        color = (0, 230, 100) if len(points) >= 3 else (0, 140, 255)
        cv2.putText(vis, badge,
                    (panel_x + 8, panel_y + panel_h + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)

        cv2.imshow(window_title, vis)

    def _mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_events[0] += 1
            points.append((x, y))
            _redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse_events[0] += 1
            if points:
                points.pop()
                _redraw()

    try:
        cv2.startWindowThread()
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, disp_w, disp_h)
        try:
            cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            # Not all backends support TOPMOST; keep going if unavailable.
            pass
        cv2.setMouseCallback(window_title, _mouse_cb)
        _redraw()
    except cv2.error as e:
        print("  ⚠  OpenCV GUI window could not be opened for ROI selection.")
        print(f"     Reason: {e}")
        print("  Falling back to full-frame processing.\n")
        return None

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │        INTERACTIVE ROI SELECTOR          │")
    print("  │  Left-click to place polygon points.     │")
    print("  │  Press ENTER/SPACE/C to confirm (≥3 pts).│")
    print("  │  Press D for rectangle or M for manual.  │")
    print("  │  Press ESC/S to skip ROI (full frame).   │")
    print("  └─────────────────────────────────────────┘\n")

    while True:
        # If the ROI window is closed manually, skip ROI cleanly.
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                skipped[0] = True
                break
        except cv2.error:
            skipped[0] = True
            break

        key = cv2.waitKeyEx(20) & 0xFF
        if key in (13, 10, 32, ord('c'), ord('C')):   # ENTER/LF/SPACE/C
            if len(points) >= 3:
                break
            else:
                print("  ⚠  Need at least 3 points before confirming.")
        elif key in (27, ord('s'), ord('S')):         # ESC/S → skip
            skipped[0] = True
            break
        elif key in (ord('r'), ord('R')):
            points.clear()
            _redraw()
        elif key in (ord('d'), ord('D')):
            print("  Rectangle mode: drag a box, then press ENTER/SPACE.")
            x, y, rw, rh = cv2.selectROI("ROI Rectangle Fallback", disp_frame, False, False)
            cv2.destroyWindow("ROI Rectangle Fallback")
            if rw > 1 and rh > 1:
                points = [
                    (int(x), int(y)),
                    (int(x + rw), int(y)),
                    (int(x + rw), int(y + rh)),
                    (int(x), int(y + rh)),
                ]
                _redraw()
                print("  Rectangle ROI created. Press C to confirm.")
            else:
                print("  ⚠  Rectangle selection cancelled.")
        elif key in (ord('m'), ord('M')):
            cv2.destroyWindow(window_title)
            while True:
                raw = input("  Enter ROI points as x1,y1;x2,y2;x3,y3 (blank to skip): ").strip()
                if not raw:
                    print("  ROI skipped — running on full frame.\n")
                    return None
                try:
                    arr = _parse_roi_points(raw)
                    print(f"  ROI confirmed with {len(arr)} points: {arr.tolist()}\n")
                    return arr
                except ValueError as e:
                    print(f"  ⚠  {e}")

    cv2.destroyWindow(window_title)

    if skipped[0]:
        print("  ROI skipped — running on full frame.\n")
        return None

    # Scale points back to original resolution
    real_points = [(int(px / scale), int(py / scale)) for (px, py) in points]
    arr = np.array(real_points, dtype=np.int64)
    print(f"  ROI confirmed with {len(arr)} points: {arr.tolist()}\n")
    return arr


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   CROP-THEN-DETECT HELPERS                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _polygon_bbox(polygon: np.ndarray):
    """Returns (x_min, y_min, x_max, y_max) for a polygon."""
    x_min = int(polygon[:, 0].min())
    y_min = int(polygon[:, 1].min())
    x_max = int(polygon[:, 0].max()) + 1
    y_max = int(polygon[:, 1].max()) + 1
    return x_min, y_min, x_max, y_max


def _crop_and_mask(frame: np.ndarray, polygon: np.ndarray):
    """
    Returns:
      crop     — the cropped sub-image (only pixels inside polygon non-black)
      x0, y0  — top-left corner of the crop in the original frame
      mask_2d  — H×W uint8 mask (255 inside polygon, 0 outside)
    """
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = _polygon_bbox(polygon)

    # Clamp to frame boundaries
    x_min = max(x_min, 0); y_min = max(y_min, 0)
    x_max = min(x_max, w); y_max = min(y_max, h)

    # Mask in full-frame space
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_full, [polygon.astype(np.int32)], 255)

    # Zero out pixels outside polygon, then crop
    masked_frame = frame.copy()
    masked_frame[mask_full == 0] = 0
    crop = masked_frame[y_min:y_max, x_min:x_max]

    return crop, x_min, y_min


def _offset_detections(det: Detections, x0: int, y0: int) -> Detections:
    """Shifts bounding box coordinates back to full-frame space."""
    if len(det) == 0:
        return det
    det.xyxy[:, 0] += x0
    det.xyxy[:, 1] += y0
    det.xyxy[:, 2] += x0
    det.xyxy[:, 3] += y0
    return det


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        HUD                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    n_current: int,
    n_total_unique: int,
    zone_label: str,
    alpha: float = 0.60,
) -> np.ndarray:
    panel_w = 260
    panel_h = 120
    x0, y0 = 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (50, 50, 50), 1)

    # Header bar
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 24), (37, 99, 235), -1)
    cv2.putText(frame, "COW TRACKER",
                (x0 + 8, y0 + 17),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, C_WHITE, 1, cv2.LINE_AA)

    def row(label, value, y, color=C_WHITE):
        cv2.putText(frame, label,      (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, str(value), (x0 + 155, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    row("Frame",               f"{frame_idx:>6}",     y0 + 44)
    row("Live FPS",            f"{fps:>5.1f}",         y0 + 60,  C_CYAN)
    row("Cows in ROI",         f"{n_current:>6}",      y0 + 78,  C_GREEN)
    row("Total unique cows",   f"{n_total_unique:>6}", y0 + 96,  C_AMBER)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (x0 + panel_w - 58, y0 + panel_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    return frame


def draw_speed_badge(
    frame: np.ndarray,
    det: Detections,
    track_history: dict,
    fps: float,
) -> np.ndarray:
    if fps <= 0:
        return frame
    if det.tracker_id is None or len(det.tracker_id) == 0:
        return frame

    for i, tid in enumerate(det.tracker_id):
        if tid is None:
            continue
        history = track_history.get(int(tid), [])
        if len(history) < 3:
            continue
        dx = history[-1][0] - history[-3][0]
        dy = history[-1][1] - history[-3][1]
        speed_px = (dx**2 + dy**2) ** 0.5 * (fps / 2.0)

        x1, y1, x2, y2 = det.xyxy[i].astype(int)
        badge_txt = f"{speed_px:.0f} px/s"
        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        bx = int((x1 + x2) / 2) - tw // 2
        by = y1 - 6
        if by > th + 4:
            cv2.rectangle(frame, (bx - 3, by - th - 3), (bx + tw + 3, by + 3),
                          (20, 20, 20), -1)
            cv2.putText(frame, badge_txt, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_AMBER, 1, cv2.LINE_AA)
    return frame


def _make_writer(path: str, fps: float, w: int, h: int):
    for codec in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"  Video codec : {codec}")
            return writer
        writer.release()
    print("  ⚠  Could not open any video writer.")
    return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        MAIN LOOP                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run(cfg: dict) -> None:
    weights = Path(cfg["weights"])
    if not weights.exists():
        sys.exit(f"❌  Weights not found: {weights}")

    print(f"\n{'─'*60}")
    print(f"  COW TRACKER  —  YOLOv8 + ByteTrack + Supervision")
    print(f"{'─'*60}")
    print(f"  Weights : {weights}")
    print(f"  Source  : {cfg['source']}")
    print(f"  Conf    : {cfg['conf']}   IoU: {cfg['iou']}   imgsz: {cfg['imgsz']}")
    print(f"{'─'*60}\n")

    model  = YOLO(str(weights))
    device = cfg["device"] if cfg["device"] else ("0" if _cuda_available() else "cpu")
    print(f"  Running on : {device}\n")

    # ── Video source ──────────────────────────────────────────────────────────
    source = cfg["source"]
    source_str = str(source).strip()
    is_webcam = isinstance(source, int) or source_str.isdigit()
    cap_source = int(source_str) if is_webcam else source_str
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open source: {source}")

    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"  Video : {vid_w}×{vid_h} @ {vid_fps:.1f} fps  ({total_frames} frames)\n")

    # ── ROI: interactive selector if points not provided ──────────────────────
    poly_cfg  = cfg.get("polygon_zone", {})
    use_poly  = poly_cfg.get("enabled", False)
    poly_pts  = poly_cfg.get("points", None)

    if use_poly and poly_pts is None:
        ret, first_frame = cap.read()
        if not ret:
            sys.exit("❌  Cannot read first frame for ROI selection.")
        poly_pts = select_roi_interactive(first_frame, "ROI Selector — Cow Tracker")
        if poly_pts is None:
            use_poly = False          # user pressed ESC → full-frame mode
        # Rewind so we don't skip the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem           = Path(str(source)).stem if not is_webcam else "webcam"
    out_video_path = out_dir / f"cow_tracked_{stem}.mp4"
    out_csv_path   = out_dir / f"cow_log_{stem}.csv"

    # ── Writer ────────────────────────────────────────────────────────────────
    writer = None
    if cfg["save"]:
        writer = _make_writer(str(out_video_path), vid_fps, vid_w, vid_h)
        if writer:
            print(f"  Saving to : {out_video_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_file = csv_writer = None
    if cfg["save_csv"]:
        csv_file   = open(out_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "timestamp_s", "tracker_id",
            "x1", "y1", "x2", "y2", "conf",
            "cows_in_roi", "total_unique_cows",
        ])
        print(f"  CSV log   : {out_csv_path}")

    print()

    # ── Tracker + annotators ──────────────────────────────────────────────────
    tracker = ByteTrack(
        track_activation_threshold = cfg["track_thresh"],
        lost_track_buffer          = cfg["track_buffer"],
        minimum_matching_threshold = cfg["match_thresh"],
        frame_rate                 = int(vid_fps),
        minimum_consecutive_frames = cfg["min_consec_frames"],
    )

    box_annotator   = RoundBoxAnnotator(color=PAL, thickness=2, roundness=0.3)
    label_annotator = LabelAnnotator(
        color=PAL, text_color=Color(255, 255, 255),
        text_scale=0.45, text_thickness=1, text_padding=5,
        text_position=sv.Position.TOP_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )
    trace_annotator = TraceAnnotator(
        color=PAL, position=sv.Position.BOTTOM_CENTER,
        trace_length=cfg["trace_length"], thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )

    # ── Polygon zone (supervision — used for annotation only) ─────────────────
    poly_zone = poly_annotator = None
    if use_poly and poly_pts is not None:
        poly_zone = PolygonZone(
            polygon=poly_pts,
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
        )
        poly_annotator = PolygonZoneAnnotator(
            zone=poly_zone, color=C_ZONE,
            thickness=2, text_scale=0.7,
            text_color=Color(0, 0, 0),
            opacity=0.15,
        )
        print(f"  ROI polygon : {len(poly_pts)} points")
        print(f"  Strategy    : crop-then-detect (model only sees inside ROI)")
    else:
        print("  ROI polygon : DISABLED — processing full frame")

    print()

    # ── State ─────────────────────────────────────────────────────────────────
    all_track_ids = set()
    track_history = defaultdict(list)
    frame_idx     = 0
    prev_time     = time.time()
    fps_smooth    = vid_fps
    FPS_ALPHA     = 0.1

    print("  Processing frames … (press Q in preview window to quit)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            now        = time.time()
            inst_fps   = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = FPS_ALPHA * inst_fps + (1 - FPS_ALPHA) * fps_smooth
            prev_time  = now

            # ── CROP-THEN-DETECT ──────────────────────────────────────────────
            # If a polygon ROI is defined:
            #   1. Crop + mask the frame to the ROI bounding box
            #   2. Run YOLO only on that small crop  → better recall, no
            #      distractions from cows outside the zone
            #   3. Shift detected boxes back to full-frame coordinates
            # If no ROI, run on the full frame as before.

            if use_poly and poly_pts is not None:
                crop, crop_x0, crop_y0 = _crop_and_mask(frame, poly_pts)
                results    = model(crop,
                                   conf=cfg["conf"], iou=cfg["iou"],
                                   imgsz=cfg["imgsz"], device=device,
                                   verbose=False)[0]
                detections = Detections.from_ultralytics(results)
                # Shift boxes back to full-frame space
                detections = _offset_detections(detections, crop_x0, crop_y0)

                # Optional polygon-zone trigger (consistency check / annotator)
                if poly_zone is not None and len(detections) > 0:
                    mask       = poly_zone.trigger(detections)
                    detections = detections[mask]
            else:
                results    = model(frame,
                                   conf=cfg["conf"], iou=cfg["iou"],
                                   imgsz=cfg["imgsz"], device=device,
                                   verbose=False)[0]
                detections = Detections.from_ultralytics(results)

            # ── ByteTrack ─────────────────────────────────────────────────────
            detections = tracker.update_with_detections(detections)

            # ── Track history ─────────────────────────────────────────────────
            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                for i, tid in enumerate(detections.tracker_id):
                    if tid is None:
                        continue
                    x1, y1, x2, y2 = detections.xyxy[i]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track_history[int(tid)].append((cx, cy))
                    if len(track_history[int(tid)]) > 120:
                        track_history[int(tid)] = track_history[int(tid)][-120:]
                    all_track_ids.add(int(tid))

            # ── Labels: ID + confidence ───────────────────────────────────────
            labels = []
            for i in range(len(detections)):
                tid  = detections.tracker_id[i] if detections.tracker_id is not None else "?"
                conf = detections.confidence[i]  if detections.confidence  is not None else 0.0
                labels.append(f"#{tid}  {conf:.0%}")

            # ── Annotate ──────────────────────────────────────────────────────
            annotated = frame.copy()

            if poly_annotator is not None:
                annotated = poly_annotator.annotate(annotated)

            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            annotated = draw_speed_badge(annotated, detections, track_history, fps_smooth)
            annotated = draw_hud(
                frame          = annotated,
                frame_idx      = frame_idx,
                fps            = fps_smooth,
                n_current      = len(detections),
                n_total_unique = len(all_track_ids),
                zone_label     = poly_cfg.get("label", "Zone"),
                alpha          = cfg["hud_alpha"],
            )

            # ── Save frame ────────────────────────────────────────────────────
            if writer is not None:
                writer.write(annotated)

            # ── CSV ───────────────────────────────────────────────────────────
            if csv_writer is not None:
                ts = frame_idx / vid_fps
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    for i, tid in enumerate(detections.tracker_id):
                        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                        conf = detections.confidence[i] if detections.confidence is not None else 0
                        csv_writer.writerow([
                            frame_idx, f"{ts:.3f}", int(tid if tid is not None else -1),
                            x1, y1, x2, y2, f"{conf:.4f}",
                            len(detections), len(all_track_ids),
                        ])

            # ── Preview ───────────────────────────────────────────────────────
            if cfg["show"]:
                cv2.imshow("Cow Tracker  —  press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  [Q pressed] — stopping early.")
                    break

            # ── Progress ──────────────────────────────────────────────────────
            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_idx:>6}"
                      f"  FPS {fps_smooth:>5.1f}"
                      f"  Cows(ROI) {len(detections):>3}"
                      f"  UniqueIDs {len(all_track_ids):>4}"
                      + (f"  [{pct:.1f}%]" if pct > 0 else ""))

    except KeyboardInterrupt:
        print("\n  [Ctrl+C] — interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total frames processed : {frame_idx}")
    print(f"  Unique cow IDs tracked : {len(all_track_ids)}")
    if cfg["save"] and writer is not None:
        print(f"  Annotated video saved  : {out_video_path}")
    if cfg["save_csv"]:
        print(f"  CSV log saved          : {out_csv_path}")
    print(f"{'═'*60}\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        UTILITIES                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cow Detection · Tracking · Count — YOLOv8 + ByteTrack"
    )
    p.add_argument("--weights", type=str,   help="Path to best.pt")
    p.add_argument("--source",  type=str,   help="Video file or 0 for webcam")
    p.add_argument("--conf",    type=float, help="Confidence threshold")
    p.add_argument("--iou",     type=float, help="NMS IoU threshold")
    p.add_argument("--imgsz",   type=int,   help="Inference image size")
    p.add_argument("--device",  type=str,   help="Device: '', 'cpu', '0', 'cuda:0'")
    p.add_argument("--show",    action="store_true", help="Show live preview")
    p.add_argument("--roi",     type=str,
                   help="Polygon points as x1,y1;x2,y2;x3,y3")
    p.add_argument("--full-frame", action="store_true",
                   help="Disable ROI selector and process full frame")
    p.add_argument("--no-save", action="store_true", help="Do not save output video")
    p.add_argument("--no-csv",  action="store_true", help="Do not save CSV log")
    p.add_argument("--out-dir", type=str,   help="Output directory")
    return p.parse_args(_normalize_cli_tokens(sys.argv[1:]))


def _normalize_cli_tokens(argv: list[str]) -> list[str]:
    """
    Accepts a common typo style like --roi"x1,y1;..." by converting
    the merged token back into ["--roi", "x1,y1;..."] before argparse.
    """
    normalized: list[str] = []
    for token in argv:
        if token.startswith("--roi") and token != "--roi" and not token.startswith("--roi="):
            value = token[len("--roi"):]
            if value:
                normalized.extend(["--roi", value])
                continue
        normalized.append(token)
    return normalized


def _parse_roi_points(roi_text: str) -> np.ndarray:
    raw_points = [chunk.strip() for chunk in roi_text.split(";") if chunk.strip()]
    points = []
    for chunk in raw_points:
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid ROI point '{chunk}'. Use x,y format.")
        x, y = int(parts[0]), int(parts[1])
        points.append((x, y))

    if len(points) < 3:
        raise ValueError("ROI needs at least 3 points.")

    return np.array(points, dtype=np.int64)


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.weights is not None:  cfg["weights"]    = args.weights
    if args.source is not None:   cfg["source"]     = args.source
    if args.conf is not None:     cfg["conf"]       = args.conf
    if args.iou is not None:      cfg["iou"]        = args.iou
    if args.imgsz is not None:    cfg["imgsz"]      = args.imgsz
    if args.device is not None:   cfg["device"]     = args.device
    if args.show:     cfg["show"]       = True
    if args.full_frame:
        cfg.setdefault("polygon_zone", {})
        cfg["polygon_zone"]["enabled"] = False
        cfg["polygon_zone"]["points"] = None
    if args.roi is not None:
        try:
            roi_points = _parse_roi_points(args.roi)
        except ValueError as e:
            raise SystemExit(f"❌  Invalid --roi value: {e}")
        cfg.setdefault("polygon_zone", {})
        cfg["polygon_zone"]["enabled"] = True
        cfg["polygon_zone"]["points"] = roi_points
    if args.no_save:  cfg["save"]       = False
    if args.no_csv:   cfg["save_csv"]   = False
    if args.out_dir is not None:  cfg["output_dir"] = args.out_dir
    return cfg


if __name__ == "__main__":
    args = _parse_args()
    cfg  = _apply_cli_overrides(CONFIG.copy(), args)
    run(cfg)