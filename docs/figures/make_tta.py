#!/usr/bin/env python3
"""TTA figure for the narrowflip profile. Does not modify the pipeline figure.

Six views: scales 0.9, 1.0, 1.1, each with and without a horizontal flip.
Flipped scores are flipped back, every view is resized to the original H×W,
the six score maps are averaged, then argmax. No sliding window.

Run from anywhere: python docs/figures/make_tta.py
Writes docs/figures/tta_narrowflip.svg next to this file.
"""

import base64
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tta_narrowflip.svg"
REG = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
BOLD = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
COLOR_IMG = ROOT / "data/LuSNAR/Moon_1/color/1693274724086398208.png"
LABEL_IMG = ROOT / "data/LuSNAR/Moon_1/label/1693274724086398208.png"

BLUE, PEACH, GREEN, FEAT = "#dae2f3", "#fae4d5", "#d2e7d2", "#fff1dc"
INK, STROKE, DASH = "#1a1a1a", "#2a2a2a", "#6f6f6f"
ARROW, BOX_SW, DASH_SW = 1.45, 1.25, 1.35


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;")


class Fig:
    def __init__(self):
        self.parts = []

    def add(self, svg):
        self.parts.append(svg)

    def text(self, x, y, text, size, bold=False, anchor="middle"):
        weight = ' font-weight="700"' if bold else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="central" '
            f'fill="{INK}" font-family="Arial" font-size="{size}"{weight}>{esc(text)}</text>'
        )

    def vtext(self, x, y, lines, size=13):
        step = size + 5
        n = len(lines)
        for i, line in enumerate(lines):
            cx = x + (i - (n - 1) / 2) * step
            self.add(
                f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="central" '
                f'fill="{INK}" font-family="Arial" font-size="{size}" font-weight="700" '
                f'transform="rotate(-90 {cx:.1f} {y:.1f})">{esc(line)}</text>'
            )

    def rect(self, x, y, w, h, fill, rx=0, sw=BOX_SW, stroke=STROKE, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def line(self, points, head=True):
        marker = ' marker-end="url(#arrow)"' if head else ""
        attrs = f'stroke="{INK}" stroke-width="{ARROW}" stroke-linecap="butt" stroke-linejoin="miter"'
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attrs}{marker}/>')
        else:
            pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in points)
            self.add(f'<polyline points="{pts}" fill="none" {attrs}{marker}/>')


def jpeg_uri(image, size):
    image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build():
    fig = Fig()
    # One row of six views. Arrows leave each thumbnail downward, so they never share a line.
    # Widths show the scale; the right edge of every thumbnail is aligned.
    scales = (0.9, 1.0, 1.1, 0.9, 1.0, 1.1)
    mirrored = (False, False, False, True, True, True)
    slot = 64
    group_extra = 44
    thumb_h = 32
    max_w = 56
    widths = [round(max_w * scale / 1.1) for scale in scales]
    img_w, img_h = 78, 48
    img_x = 16
    thumb_top = 64
    thumb_cy = thumb_top + thumb_h / 2
    left0 = img_x + img_w + 28
    rights = [left0 + max_w + i * slot + (group_extra if i >= 3 else 0) for i in range(6)]
    lefts = [right - width for right, width in zip(rights, widths)]

    model_y = thumb_top + thumb_h + 28
    model_h = 36
    img_y = model_y + (model_h - img_h) / 2
    model_x = lefts[0]
    model_w = rights[-1] - lefts[0]
    # Unflip sits under the flipped three only. Resize lines up on every column.
    op_w, op_h = 46, 22
    hflip_y = model_y + model_h + 16
    resize_y = hflip_y + op_h + 14
    hxw_bottom = resize_y + op_h
    # The collecting polyline sits in the gap under H×W, not on the box edges.
    mean_w, mean_h = 62, 40
    arg_w, arg_h = 68, 32
    result_w = mean_w + 14 + arg_w + 16 + img_w
    mid = (lefts[0] + rights[-1]) / 2
    mean_x = mid - result_w / 2
    mean_y = hxw_bottom + 40
    bus_y = hxw_bottom + (mean_y - hxw_bottom) / 2
    arg_x = mean_x + mean_w + 14
    arg_y = mean_y + (mean_h - arg_h) / 2
    mask_x = arg_x + arg_w + 16
    mask_y = mean_y + (mean_h - img_h) / 2
    out_cy = mean_y + mean_h / 2
    canvas_w = max(rights[-1] + 16, mask_x + img_w + 8 + 46 + 10)
    canvas_h = max(mean_y + mean_h, mask_y + img_h) + 14

    full_crop_h = int(round(1024 * img_h / img_w))
    full = Image.open(COLOR_IMG).convert("RGB").crop((0, 36, 1024, 36 + full_crop_h))
    label = Image.open(LABEL_IMG).convert("RGB").crop((0, 36, 1024, 36 + full_crop_h))
    small_h = int(round(1024 * thumb_h / max_w))
    small = Image.open(COLOR_IMG).convert("RGB").crop((0, 36, 1024, 36 + small_h))
    flipped = small.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    color_uri = jpeg_uri(full, (img_w * 2, img_h * 2))
    label_uri = jpeg_uri(label, (img_w * 2, img_h * 2))
    view_uris = [jpeg_uri(small, (width * 2, thumb_h * 2)) for width in widths[:3]]
    flip_uris = [jpeg_uri(flipped, (width * 2, thumb_h * 2)) for width in widths[3:]]

    font_reg = base64.b64encode(Path(REG).read_bytes()).decode()
    font_bold = base64.b64encode(Path(BOLD).read_bytes()).decode()
    fig.add(
        f'''<defs>
  <style>
    @font-face {{ font-family: "Arial"; src: url("data:font/ttf;base64,{font_reg}") format("truetype"); font-weight: 400; }}
    @font-face {{ font-family: "Arial"; src: url("data:font/ttf;base64,{font_bold}") format("truetype"); font-weight: 700; }}
  </style>
  <marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="4.2" markerHeight="4.2" orient="auto" markerUnits="strokeWidth">
    <path d="M1.2,1.2 L10.6,6 L1.2,10.8 Z" fill="{INK}"/>
  </marker>
</defs>'''
    )
    fig.add(f'<rect width="{canvas_w:.0f}" height="{canvas_h:.0f}" fill="#ffffff"/>')

    def thumb(x, y, w, h, href, clip):
        fig.add(f'<clipPath id="{clip}"><rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="2"/></clipPath>')
        fig.add(
            f'<image href="{href}" x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" clip-path="url(#{clip})"/>'
        )
        fig.rect(x, y, w, h, "none", rx=2, sw=1.15)

    fig.text(img_x + img_w / 2, img_y - 11, "Image", 13, bold=True)
    thumb(img_x, img_y, img_w, img_h, color_uri, "cI")
    thumb(mask_x, mask_y, img_w, img_h, label_uri, "cM")
    fig.text(mask_x + img_w + 8, mask_y + img_h / 2, "Mask", 13, bold=True, anchor="start")

    # Labels sit above the trunk. Drops run from the trunk to the thumbnail and do not cross text.
    trunk_y = thumb_top - 18
    label_y = trunk_y - 14
    trunk_x = img_x + img_w + 14
    img_cy = img_y + img_h / 2
    fig.line([(img_x + img_w, img_cy), (trunk_x, img_cy)], head=False)
    fig.line([(trunk_x, img_cy), (trunk_x, trunk_y)], head=False)
    labels = ["0.9×", "1.0×", "1.1×", "0.9×", "1.0×", "1.1×"]
    centers = [left + width / 2 for left, width in zip(lefts, widths)]
    fig.line([(trunk_x, trunk_y), (centers[-1], trunk_y)], head=False)
    for i, (left, width, flip, label, cx) in enumerate(zip(lefts, widths, mirrored, labels, centers)):
        href = (flip_uris if flip else view_uris)[i % 3]
        thumb(left, thumb_top, width, thumb_h, href, f"v{i}")
        fig.text(cx, label_y, label, 11)
        fig.line([(cx, trunk_y), (cx, thumb_top)])

    pad = 8
    box_x = lefts[3] - pad
    box_y = thumb_top - pad
    box_w = (lefts[-1] + widths[-1]) - lefts[3] + 2 * pad
    box_h = (thumb_top + thumb_h + pad) - box_y
    fig.rect(box_x, box_y, box_w, box_h, "none", sw=DASH_SW, stroke=DASH, dash="5 3.5")
    gutter = (rights[2] + lefts[3]) / 2
    fig.vtext(gutter, thumb_cy, ["Hflip"], 12)

    fig.rect(model_x, model_y, model_w, model_h, GREEN, rx=8)
    fig.text(model_x + model_w / 2, model_y + model_h / 2, "Model", 13, bold=True)
    for cx in centers:
        fig.line([(cx, thumb_top + thumb_h), (cx, model_y)])

    for cx in centers[3:]:
        fig.rect(cx - op_w / 2, hflip_y, op_w, op_h, BLUE, rx=6)
        fig.text(cx, hflip_y + op_h / 2, "Hflip", 12, bold=True)
    for cx in centers:
        fig.rect(cx - op_w / 2, resize_y, op_w, op_h, FEAT, rx=0, sw=1.1)
        fig.text(cx, resize_y + op_h / 2, "H×W", 12, bold=True)
    for cx in centers[:3]:
        fig.line([(cx, model_y + model_h), (cx, resize_y)])
    for cx in centers[3:]:
        fig.line([(cx, model_y + model_h), (cx, hflip_y)])
        fig.line([(cx, hflip_y + op_h), (cx, resize_y)])
    for cx in centers:
        fig.line([(cx, resize_y + op_h), (cx, bus_y)], head=False)
    fig.line([(centers[0], bus_y), (centers[-1], bus_y)], head=False)
    drop_x = mean_x + mean_w / 2
    fig.line([(drop_x, bus_y), (drop_x, mean_y)])

    fig.rect(mean_x, mean_y, mean_w, mean_h, PEACH, rx=8)
    fig.text(mean_x + mean_w / 2, out_cy - 8, "Mean", 13, bold=True)
    fig.text(mean_x + mean_w / 2, out_cy + 9, "6 views", 11)
    fig.rect(arg_x, arg_y, arg_w, arg_h, GREEN, rx=8)
    fig.text(arg_x + arg_w / 2, out_cy, "argmax", 13, bold=True)
    fig.line([(mean_x + mean_w, out_cy), (arg_x, out_cy)])
    fig.line([(arg_x + arg_w, out_cy), (mask_x, out_cy)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">\n'
        + "\n".join(fig.parts) + "\n</svg>\n"
    )
    OUT.write_text(svg)
    print(
        f"{canvas_w:.0f}x{canvas_h:.0f} thumbs={thumb_top:.0f} model={model_y:.0f}-{model_y + model_h:.0f} "
        f"bus={bus_y:.0f}"
    )


if __name__ == "__main__":
    build()
