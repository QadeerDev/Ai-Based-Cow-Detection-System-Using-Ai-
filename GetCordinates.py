import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\muham\Desktop\Linked In Optimization\Week 02\Cows Detection\frames\frame_00000.jpg"
SAVE_PATH  = r"C:\Users\muham\Desktop\Linked In Optimization\Week 02\Cows Detection\roi_preview.jpg"

# ── Display window size — adjust to fit your screen ────────────────
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# ── Load image + compute scale factors ─────────────────────────────
orig  = cv2.imread(IMAGE_PATH)
OH, OW = orig.shape[:2]

SCALE_X = OW / DISPLAY_WIDTH    # clicked px → original px
SCALE_Y = OH / DISPLAY_HEIGHT

# Working display copy (resized)
display     = cv2.resize(orig, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
display_clone = display.copy()

# Collected points in ORIGINAL resolution
points_orig    = []   # what gets printed / saved
# Collected points in DISPLAY resolution (for drawing)
points_display = []

def click(event, x, y, flags, param):
    global display

    if event == cv2.EVENT_LBUTTONDOWN:
        # Map display coords → original coords
        ox = int(x * SCALE_X)
        oy = int(y * SCALE_Y)

        points_orig.append((ox, oy))
        points_display.append((x, y))
        print(f"  P{len(points_orig)}: display({x},{y})  →  original({ox},{oy})")

        # Draw on display image
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display, f"P{len(points_orig)}({ox},{oy})",
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

        if len(points_display) > 1:
            cv2.line(display, points_display[-2], points_display[-1], (0, 255, 0), 2)

        # Close and fill after 4th point
        if len(points_display) == 4:
            cv2.line(display, points_display[-1], points_display[0], (0, 255, 0), 2)
            overlay = display.copy()
            roi_d   = np.array(points_display, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_d], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

            # Save full-resolution preview with polygon drawn
            roi_orig = np.array(points_orig, dtype=np.int32)
            save_img = orig.copy()
            save_overlay = save_img.copy()
            cv2.fillPoly(save_overlay, [roi_orig], (0, 255, 0))
            cv2.addWeighted(save_overlay, 0.2, save_img, 0.8, 0, save_img)
            cv2.polylines(save_img, [roi_orig], True, (0, 255, 0), 4)
            for i, (px, py) in enumerate(points_orig):
                cv2.circle(save_img, (px, py), 12, (0, 0, 255), -1)
                cv2.putText(save_img, f"P{i+1}({px},{py})", (px+10, py-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imwrite(SAVE_PATH, save_img)

            print(f"\n✅ ROI complete!")
            print(f"\n   Copy this into your notebook:")
            print(f"\n   ROI_POINTS = np.array({points_orig}, dtype=np.int32)")
            print(f"\n   Full-res preview saved to: {SAVE_PATH}")

        cv2.imshow(WIN, display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points_orig:
            removed = points_orig.pop()
            points_display.pop()
            print(f"  ↩  Removed point: original{removed}")

            # Redraw from scratch
            display[:] = display_clone.copy()
            for i, (dx, dy) in enumerate(points_display):
                ox, oy = points_orig[i]
                cv2.circle(display, (dx, dy), 6, (0, 0, 255), -1)
                cv2.putText(display, f"P{i+1}({ox},{oy})",
                            (dx+8, dy-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 2)
                if i > 0:
                    cv2.line(display, points_display[i-1], (dx,dy), (0,255,0), 2)
            cv2.imshow(WIN, display)

# ── Launch window ───────────────────────────────────────────────────
WIN = "ROI Picker  |  Left click = add point  |  Right click = undo  |  Q = quit"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, DISPLAY_WIDTH, DISPLAY_HEIGHT)

print(f"Original resolution : {OW} x {OH}")
print(f"Display resolution  : {DISPLAY_WIDTH} x {DISPLAY_HEIGHT}")
print(f"Scale factors       : X={SCALE_X:.3f}  Y={SCALE_Y:.3f}")
print()
print("Instructions:")
print("  Left click  — add point  (clockwise: TL → TR → BR → BL)")
print("  Right click — undo last point")
print("  Press Q     — quit\n")

cv2.imshow(WIN, display)
cv2.setMouseCallback(WIN, click)
cv2.waitKey(0)
cv2.destroyAllWindows()