#!/usr/bin/env python3
"""Pipeline figure. Decoder titles are vertical. Four scales stay on their own rows.

Run from anywhere: python docs/figures/make_pipeline.py
Writes docs/figures/vit_adapter_mask2former.svg next to this file.
"""

import base64
import io
from pathlib import Path

from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "vit_adapter_mask2former.svg"
REG = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
BOLD = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
COLOR_IMG = ROOT / "data/LuSNAR/Moon_1/color/1693274724086398208.png"
LABEL_IMG = ROOT / "data/LuSNAR/Moon_1/label/1693274724086398208.png"

BLUE, PEACH, GREEN, FEAT = "#dae2f3", "#fae4d5", "#d2e7d2", "#fff1dc"
INK, STROKE, DASH = "#1a1a1a", "#2a2a2a", "#6f6f6f"
ARROW, BOX_SW, DASH_SW = 1.45, 1.25, 1.35
SQ = 32


def face(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, size)


def text_w(text, size, bold=False):
    return face(size, bold).getlength(text)


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
        """Each line is rotated -90 about its own center, so the words read upward."""
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


def jpeg_uri(path, crop, size):
    image = Image.open(path).convert("RGB").crop(crop).resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64(path):
    return "data:font/ttf;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def trapezoid(fig, top_l, top_r, bot_l, bot_r, y0, y1, radius):
    pts = [(top_l, y0), (top_r, y0), (bot_r, y1), (bot_l, y1)]

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    def mul(v, s):
        return (v[0] * s, v[1] * s)

    def length(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5 or 1.0

    def norm(v):
        ell = length(v)
        return (v[0] / ell, v[1] / ell)

    cmds = []
    for i, cur in enumerate(pts):
        prev, nxt = pts[i - 1], pts[(i + 1) % 4]
        din = min(radius, length(sub(cur, prev)) * 0.4)
        dout = min(radius, length(sub(nxt, cur)) * 0.4)
        before = sub(cur, mul(norm(sub(cur, prev)), din))
        after = add(cur, mul(norm(sub(nxt, cur)), dout))
        cmds.append(("M" if i == 0 else "L") + f"{before[0]:.1f},{before[1]:.1f}")
        cmds.append(f"Q{cur[0]:.1f},{cur[1]:.1f} {after[0]:.1f},{after[1]:.1f}")
    fig.add(
        f'<path d="{" ".join(cmds)} Z" fill="{BLUE}" stroke="{STROKE}" stroke-width="{BOX_SW}"/>'
    )


def build():
    fig = Fig()
    in_sizes = [(64, 22), (50, 18), (40, 16), (30, 13)]
    out_sizes = [(58, 20), (46, 17), (36, 14), (28, 12)]
    left_labels = ["D,H/4,W/4", "D,H/8,W/8", "D,H/16,W/16", "D,H/32,W/32"]
    right_labels = ["C,H/4,W/4", "C,H/8,W/8", "C,H/16,W/16", "C,H/32,W/32"]

    img_w, img_h = 78, 48
    img_y = 20
    pitch = 56
    # Four rows are mirrored about the Pixel Decoder center: -1.5, -0.5, +0.5, +1.5 pitch.
    vit_top = img_y + img_h + 22
    rows = [vit_top + SQ / 2 + i * pitch for i in range(4)]
    center = (rows[0] + rows[3]) / 2
    # One bottom outlet. The trunk drops clear of the second row, then branches.
    vit_bot = rows[1] - 18
    vit_cx = 64
    top_half, bot_half = 46, 28

    def right_edge(y):
        t = (y - vit_top) / (vit_bot - vit_top)
        return (vit_cx + top_half) + t * ((vit_cx + bot_half) - (vit_cx + top_half))

    side_x = right_edge(rows[0]) + 0.8
    # Adapter column sits 6px left of the feature column's anchor.
    feat_sq = side_x + 26
    sq_x = feat_sq - 6
    sq_cx = sq_x + SQ / 2

    # Same height so the gaps between the four rows are equal, not just the centers.
    feat_h = 16
    in_sizes = [(w, feat_h) for w, _ in in_sizes]
    out_sizes = [(w, feat_h) for w, _ in out_sizes]
    label_size = 11
    label_w = max(text_w(s, label_size, False) for s in left_labels + right_labels)
    in_right = feat_sq + SQ + 16 + label_w
    # Equal arrow length on both sides of the Pixel Decoder.
    side_gap = 16
    pix_w = 46
    pix_x = in_right + side_gap
    pix_pad = 18
    pix_y = center - (rows[3] - rows[0]) / 2 - pix_pad
    pix_h = (rows[3] - rows[0]) + 2 * pix_pad

    out_left = pix_x + pix_w + side_gap
    tr_w = 62
    tr_x = out_left + max(w for w, _ in out_sizes) + 20
    tr_pad = 18
    tr_mid = (rows[1] + rows[3]) / 2
    tr_y = tr_mid - (rows[3] - rows[1]) / 2 - tr_pad
    tr_h = (rows[3] - rows[1]) + 2 * tr_pad

    head_w = max(text_w(s, 13, True) for s in ("Mask", "Class", "head")) + 22
    head_h = 40
    head_y = rows[0] - head_h / 2
    mh_x = tr_x
    ch_x = tr_x + tr_w + 8
    mask_cx = (mh_x + ch_x + head_w) / 2
    mask_x = mask_cx - img_w / 2
    mask_y = img_y
    join_y = (mask_y + img_h + head_y) / 2
    img_x = vit_cx - img_w / 2
    canvas_w = max(ch_x + head_w, tr_x + tr_w, mask_x + img_w) + 14
    canvas_h = max(pix_y + pix_h, tr_y + tr_h, rows[3] + SQ / 2 + 12) + 10

    crop_h = int(round(1024 * img_h / img_w))
    crop = (0, 36, 1024, 36 + crop_h)
    color_uri = jpeg_uri(COLOR_IMG, crop, (img_w * 2, img_h * 2))
    label_uri = jpeg_uri(LABEL_IMG, crop, (img_w * 2, img_h * 2))

    fig.add(
        f'''<defs>
  <style>
    @font-face {{ font-family: "Arial"; src: url("data:font/ttf;base64,{base64.b64encode(Path(REG).read_bytes()).decode()}") format("truetype"); font-weight: 400; }}
    @font-face {{ font-family: "Arial"; src: url("data:font/ttf;base64,{base64.b64encode(Path(BOLD).read_bytes()).decode()}") format("truetype"); font-weight: 700; }}
  </style>
  <marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="4.2" markerHeight="4.2" orient="auto" markerUnits="strokeWidth">
    <path d="M1.2,1.2 L10.6,6 L1.2,10.8 Z" fill="{INK}"/>
  </marker>
</defs>'''
    )
    fig.add(f'<rect width="{canvas_w:.0f}" height="{canvas_h:.0f}" fill="#ffffff"/>')

    def thumb(x, y, href, clip):
        fig.add(f'<clipPath id="{clip}"><rect x="{x:.1f}" y="{y:.1f}" width="{img_w}" height="{img_h}" rx="2"/></clipPath>')
        fig.add(f'<image href="{href}" x="{x:.1f}" y="{y:.1f}" width="{img_w}" height="{img_h}" clip-path="url(#{clip})"/>')
        fig.rect(x, y, img_w, img_h, "none", rx=2, sw=1.15)

    fig.text(img_x + img_w / 2, img_y - 10, "Image", 13, bold=True)
    thumb(img_x, img_y, color_uri, "cI")
    fig.text(mask_x + img_w / 2, mask_y - 10, "Mask", 13, bold=True)
    thumb(mask_x, mask_y, label_uri, "cM")

    top_l, top_r = vit_cx - top_half, vit_cx + top_half
    bot_l, bot_r = vit_cx - bot_half, vit_cx + bot_half
    trapezoid(fig, top_l, top_r, bot_l, bot_r, vit_top, vit_bot, 9)
    fig.text(vit_cx, (vit_top + vit_bot) / 2, "ViT", 16, bold=True)
    fig.line([(vit_cx, img_y + img_h), (vit_cx, vit_top)])

    # Keep the dash off the first square's top edge and off the title.
    box_pad = 10
    box_x = sq_x - box_pad
    box_y = rows[0] - SQ / 2 - box_pad
    box_w = SQ + 2 * box_pad
    box_h = (rows[3] + SQ / 2 + box_pad) - box_y
    fig.rect(box_x, box_y, box_w, box_h, "none", sw=DASH_SW, stroke=DASH, dash="5 3.5")
    fig.text(box_x + box_w / 2, box_y - 30, "ViT", 12, bold=True)
    fig.text(box_x + box_w / 2, box_y - 15, "Adapter", 12, bold=True)
    for y in rows:
        fig.rect(sq_x, y - SQ / 2, SQ, SQ, BLUE, rx=3)

    # Top scale leaves the right side. One bottom outlet, then three branches.
    fig.line([(side_x, rows[0]), (sq_x, rows[0])])
    fig.line([(vit_cx, vit_bot), (vit_cx, rows[3])], head=False)
    for y in rows[1:]:
        fig.line([(vit_cx, y), (sq_x, y)])

    for y, (w, h), label in zip(rows, in_sizes, left_labels):
        fig.rect(in_right - w, y - h / 2, w, h, FEAT, sw=1.1)
        fig.text(in_right, y - h / 2 - 8, label, label_size, anchor="end")
        fig.line([(sq_x + SQ, y), (in_right - w, y)])
        fig.line([(in_right, y), (pix_x, y)])

    fig.rect(pix_x, pix_y, pix_w, pix_h, PEACH, rx=8)
    fig.vtext(pix_x + pix_w / 2, center, ["Pixel Decoder"], 13)

    for y, (w, h), label in zip(rows, out_sizes, right_labels):
        fig.rect(out_left, y - h / 2, w, h, FEAT, sw=1.1)
        fig.text(out_left, y - h / 2 - 8, label, label_size, anchor="start")
        fig.line([(pix_x + pix_w, y), (out_left, y)])

    fig.rect(tr_x, tr_y, tr_w, tr_h, GREEN, rx=8)
    fig.vtext(tr_x + tr_w / 2, tr_mid, ["Transformer", "Decoder"], 13)
    for y, (w, _) in zip(rows[1:], out_sizes[1:]):
        fig.line([(out_left + w, y), (tr_x, y)])

    def head_box(x, lines):
        fig.rect(x, head_y, head_w, head_h, GREEN, rx=8)
        step = 15
        for i, line in enumerate(lines):
            fig.text(x + head_w / 2, rows[0] + (i - 0.5) * step, line, 13, bold=True)

    head_box(mh_x, ["Mask", "head"])
    head_box(ch_x, ["Class", "head"])
    fig.line([(out_left + out_sizes[0][0], rows[0]), (mh_x, rows[0])])
    mh_cx = mh_x + head_w / 2
    fig.line([(mh_cx, tr_y), (mh_cx, head_y + head_h)])
    # Class head leaves the decoder's vertical center, not the top edge.
    ch_cx = ch_x + head_w / 2
    fig.line([(tr_x + tr_w, tr_mid), (ch_cx, tr_mid), (ch_cx, head_y + head_h)])
    fig.line([(mh_x + head_w / 2, head_y), (mh_x + head_w / 2, join_y)], head=False)
    fig.line([(ch_x + head_w / 2, head_y), (ch_x + head_w / 2, join_y)], head=False)
    fig.line([(mh_x + head_w / 2, join_y), (ch_x + head_w / 2, join_y)], head=False)
    fig.line([(mask_cx, join_y), (mask_cx, mask_y + img_h)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">\n'
        + "\n".join(fig.parts) + "\n</svg>\n"
    )
    OUT.write_text(svg)
    print(
        f"{canvas_w:.0f}x{canvas_h:.0f} rows={['%.0f'%r for r in rows]} "
        f"center={center:.1f} pix={pix_y:.1f}-{pix_y+pix_h:.1f} "
        f"tr={tr_y:.1f}-{tr_y+tr_h:.1f} vit={vit_top:.0f}-{vit_bot:.0f}"
    )


if __name__ == "__main__":
    build()
