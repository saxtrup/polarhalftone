"""
File: polarhalftone.py
Author: Lauritz Saxtrup
Date: 2025-10-20
Version: 1.0.0
Description: polarhalftone is a Python toolchain that converts bitmap images into circular halftone line art, then optimizes the resulting SVGs for PrusaSlicer, laser engraving, or plotter workflows.

It works by sampling brightness in polar coordinates, generating a ring-based halftone pattern that preserves tonal variation while producing clean, slice-ready vector geometry.

License: MIT License
Contact: lauritz@example.com

Dependencies:
- numpy
- matplotlib
- cv2
- svgwrite

Usage:

For experimenting and viewing in Inkscape:
python polarhalftone.py input.png --center_x -150 --clean

Or, more conveniently, use Inkscape to convert it to PNG
inkscape "$(ls -t *.svg | head -n 1)" --export-type=png

Once you are satisfied, generate an output that PrusaSlicer accepts by adding the --print option.


Notes:
- Use --clean to remove anomalous segments (>500px by default).

Changelog:
- 2025-10-20: Initial creation by Lauritz Saxtrup
- 2025-10-20: Added cleanup and print compatibility by Lauritz Saxtrup
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import svgwrite
import datetime
import argparse
import sys
import xml.etree.ElementTree as ET
import os

# --- Image Loading & CLAHE ---
def load_and_orient(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: Input file not found: {image_path}")
        sys.exit(1)
    try:
        img = plt.imread(image_path)
    except Exception as e:
        print(f"Error: Could not read image '{image_path}': {e}")
        sys.exit(1)
    if img is None:
        print(f"Error: Image '{image_path}' could not be loaded (unsupported format or empty file).")
        sys.exit(1)
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    print(f"Loaded image: {w}x{h}")
    return img

def grayscale_clahe(img):
    if len(img.shape) == 3:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = img.copy()
    gray = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# --- Progress bar helper ---
def print_progress_bar(fraction, bar_length=40):
    fraction = min(max(fraction, 0.0), 1.0)
    filled_length = int(bar_length * fraction)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\rProgress: |{bar}| {fraction*100:6.2f}%", end="")
    if fraction >= 1.0:
        print()
    sys.stdout.flush()

# --- Cohen–Sutherland line clipping ---
INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

def compute_outcode(x, y, xmin, xmax, ymin, ymax):
    code = INSIDE
    if x < xmin: code |= LEFT
    elif x > xmax: code |= RIGHT
    if y < ymin: code |= BOTTOM
    elif y > ymax: code |= TOP
    return code

def clip_line(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    outcode0 = compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
    outcode1 = compute_outcode(x1, y1, xmin, xmax, ymin, ymax)
    accept = False
    while True:
        if not (outcode0 | outcode1):
            accept = True
            break
        elif outcode0 & outcode1:
            break
        else:
            outcode_out = outcode0 if outcode0 else outcode1
            if outcode_out & TOP:
                if abs(y1 - y0) < 1e-10:
                    break
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif outcode_out & BOTTOM:
                if abs(y1 - y0) < 1e-10:
                    break
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif outcode_out & RIGHT:
                if abs(x1 - x0) < 1e-10:
                    break
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            elif outcode_out & LEFT:
                if abs(x1 - x0) < 1e-10:
                    break
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin
            # Bounds check after calculation
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                break
            if outcode_out == outcode0:
                x0, y0 = x, y
                outcode0 = compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
            else:
                x1, y1 = x, y
                outcode1 = compute_outcode(x1, y1, xmin, xmax, ymin, ymax)
    if accept:
        return x0, y0, x1, y1
    return None

# --- Add Inkscape page color metadata ---
def set_inkscape_pagecolor(svg_path, color="#000000", opacity="1.0"):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        ns = {
            "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
            "inkscape": "http://www.inkscape.org/namespaces/inkscape",
        }
        namedview = root.find("sodipodi:namedview", ns)
        if namedview is None:
            namedview = ET.SubElement(root, "{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}namedview")
        namedview.set("pagecolor", color)
        namedview.set("{http://www.inkscape.org/namespaces/inkscape}pageopacity", opacity)
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        print(f"Warning: could not set Inkscape page color ({e})")

# --- Helper functions for cleanup ---
def parse_svg_path(d):
    """Parse an SVG path 'd' attribute into a list of (command, x, y) tuples."""
    commands = []
    coords = []
    i = 0
    while i < len(d):
        while i < len(d) and d[i].isspace():
            i += 1
        if i >= len(d):
            break
        if d[i].isalpha():
            cmd = d[i]
            i += 1
        else:
            if commands:
                cmd = commands[-1][0]
            else:
                continue
        while i < len(d) and d[i].isspace():
            i += 1
        while i < len(d) and not d[i].isalpha():
            num = ''
            while i < len(d) and (d[i].isdigit() or d[i] in '.-'):
                num += d[i]
                i += 1
            if num:
                coords.append(float(num))
            while i < len(d) and d[i] in ' ,':
                i += 1
        while len(coords) >= 2:
            x, y = coords[:2]
            commands.append((cmd, x, y))
            coords = coords[2:]
    return commands

def analyze_segment_length(x0, y0, x1, y1):
    """Calculate the length of a line segment."""
    return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

# --- SVG Export with Cleanup ---
def etching_svg(gray, N, min_thick, max_thick, epsilon, center_x, center_y, output_file=None,
                base_arc_len=2.0, verbose=False, progress=False, stroke_tolerance=0.01,
                minify_svg=True, print_mode=False, clean=False, threshold=500):
    h, w = gray.shape
    if center_x is None:
        center_x = w / 2
    if center_y is None:
        center_y = h / 2
    dwg = svgwrite.Drawing(output_file if output_file else "etched_output.svg", size=(f"{w}", f"{h}"))
    ET.register_namespace('', 'http://www.w3.org/2000/svg')  # Ensure default namespace for compatibility
    xmin, xmax = 0, w
    ymin, ymax = 0, h
    stroke_color = "black" if print_mode else "white"
    base_dr = w / N  # Target N rings across the width
    print(f"Generating SVG with base_dr {base_dr:.1f} for approx {N} rings across width...")
    
    min_dr = 0.5
    r = 0.0
    ring_count = 0
    max_r = max(np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)])
    while r < max_r:
        num_segments = max(24, int(2 * np.pi * r / base_arc_len))
        angles = np.linspace(0, 2 * np.pi, num_segments)
        xs = center_x + r * np.cos(angles)
        ys = center_y + r * np.sin(angles)
        xs_int = np.clip(xs.astype(int), 0, w - 1)
        ys_int = np.clip(ys.astype(int), 0, h - 1)
        avg_brightness = np.mean(gray[ys_int, xs_int]) / 255.0
        dr = base_dr * (avg_brightness + epsilon)
        dr = max(dr, min_dr)
        if verbose:
            print(f"[Ring {ring_count+1:3d}] Radius={r:.1f}, Segments={num_segments}, Avg Brightness={avg_brightness:.3f}, dr={dr:.3f}")
        clipped_segments = []
        for i in range(num_segments - 1):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i + 1], ys[i + 1]
            clipped = clip_line(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
            if clipped is not None:
                x0c, y0c, x1c, y1c = clipped
                iy = np.clip(int(round(y0c)), 0, h - 1)
                ix = np.clip(int(round(x0c)), 0, w - 1)
                local_brightness = gray[iy, ix] / 255.0
                local_stroke = min_thick + (max_thick - min_thick) * local_brightness
                local_stroke = min(local_stroke, dr * 0.9)
                clipped_segments.append(((x0c, y0c), (x1c, y1c), local_stroke))
        current_points = []
        current_stroke = None
        for start, end, stroke in clipped_segments:
            if current_stroke is None or abs(stroke - current_stroke) > stroke_tolerance:
                if current_points:
                    path_data = [(f"M {current_points[0][0]:.6f} {current_points[0][1]:.6f}")]
                    path_data += [f"L {px:.6f} {py:.6f}" for px, py in current_points[1:]]
                    dwg.add(dwg.path(path_data, stroke=stroke_color, stroke_width=current_stroke,
                                     stroke_linecap="round", stroke_linejoin="round"))
                current_points = [start, end]
                current_stroke = stroke
            else:
                current_points.append(end)
        if current_points:
            path_data = [(f"M {current_points[0][0]:.6f} {current_points[0][1]:.6f}")]
            path_data += [f"L {px:.6f} {py:.6f}" for px, py in current_points[1:]]
            dwg.add(dwg.path(path_data, stroke=stroke_color, stroke_width=current_stroke,
                             stroke_linecap="round", stroke_linejoin="round"))
        r += dr
        ring_count += 1
        if progress:
            print_progress_bar(r / max_r)
    if progress:
        print()
    
    # Cleanup phase if enabled
    if clean:
        try:
            tree = ET.ElementTree(ET.fromstring(dwg.tostring()))  # Convert dwg to ElementTree
            root = tree.getroot()
            width = float(root.get('width', 1024))
            height = float(root.get('height', 1024))
            image_size = (width, height)
            print(f"Performing in-line cleanup with size {image_size} and threshold {threshold}px")
            modified = False
            for elem in list(root.iter('{http://www.w3.org/2000/svg}path')):  # Use list to avoid iteration issues
                d = elem.get('d', '')
                if not d:
                    continue
                commands = parse_svg_path(d)
                if len(commands) < 2:
                    continue
                new_paths = []
                current_path = []
                start = None
                for cmd, x, y in commands:
                    if cmd.upper() == 'M':
                        if current_path:
                            new_paths.append(" ".join(current_path))
                            current_path = []
                        current_path.append(f"M {x:.6f} {y:.6f}")
                        start = (x, y)
                    elif cmd.upper() == 'L':
                        if start:
                            length = analyze_segment_length(start[0], start[1], x, y)
                            if length > threshold:
                                if verbose:
                                    print(f"Excluding anomalous segment: Start ({start[0]:.1f}, {start[1]:.1f}) to End ({x:.1f}, {y:.1f}), Length: {length:.1f}px")
                                if current_path:
                                    new_paths.append(" ".join(current_path))
                                    current_path = []
                                current_path = [f"M {x:.6f} {y:.6f}"]
                            else:
                                if verbose:
                                    print(f"Keeping segment: Start ({start[0]:.1f}, {start[1]:.1f}) to End ({x:.1f}, {y:.1f}), Length: {length:.1f}px")
                                current_path.append(f"L {x:.6f} {y:.6f}")
                            start = (x, y)
                if current_path:
                    new_paths.append(" ".join(current_path))
                if len(new_paths) > 1 or (len(new_paths) == 1 and new_paths[0] != d):
                    root.remove(elem)
                    for new_d in new_paths:
                        new_elem = ET.SubElement(root, '{http://www.w3.org/2000/svg}path')
                        for attr, value in elem.attrib.items():
                            new_elem.set(attr, value)
                        new_elem.set('d', new_d)
                    modified = True
                    if verbose:
                        print(f"Modified path: Original d '{d[:50]}...' (length {len(d)}) -> Split into {len(new_paths)} paths")
            if modified or True:  # Force save even if no modifications for testing
                temp_output = output_file + '.tmp'
                tree.write(temp_output, encoding='utf-8', xml_declaration=True, short_empty_elements=True)
                if not print_mode:
                    set_inkscape_pagecolor(temp_output)
                os.replace(temp_output, output_file)
                print(f"Cleaned SVG saved to {output_file}")
            else:
                print(f"No modifications needed during cleanup")
        except Exception as e:
            print(f"Cleanup failed: {e}")
            dwg.save(pretty=not minify_svg)
            if not print_mode:
                set_inkscape_pagecolor(output_file)
            print(f"Fallback SVG saved to {output_file}")
    else:
        dwg.save(pretty=not minify_svg)
        if not print_mode:
            set_inkscape_pagecolor(output_file)
        print(f"Saved SVG: {output_file}")
    if verbose:
        print(f"Total rings drawn: {ring_count}")

# --- Headless Export ---
def generate_etching(args):
    if args.N <= 0:
        print("Error: Number of rings (--N) must be positive")
        sys.exit(1)
    if args.min_thick < 0 or args.max_thick < args.min_thick:
        print("Error: Invalid thickness values (min_thick >= 0 and max_thick >= min_thick)")
        sys.exit(1)
    if args.base_N <= 0:
        print("Error: Base number of rings (--base_N) must be positive")
        sys.exit(1)
    img = load_and_orient(args.input)
    gray = grayscale_clahe(img)
    try:
        center_x = None if args.center_x.lower() == "none" else float(args.center_x)
        center_y = None if args.center_y.lower() == "none" else float(args.center_y)
    except ValueError:
        print("Error: center_x and center_y must be 'none' or valid numbers")
        sys.exit(1)
    thickness_scale = args.base_N / args.N if args.N != 0 else 1.0
    min_thick_scaled = max(0.1, args.min_thick * thickness_scale)
    max_thick_scaled = args.max_thick * thickness_scale
    print(f"Scaling thicknesses by {thickness_scale:.2f} (based on base_N={args.base_N} and N={args.N})")
    print(f"Effective min_thick: {min_thick_scaled:.2f}, max_thick: {max_thick_scaled:.2f}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.output if args.output else f"etched_output_{timestamp}.svg"
    etching_svg(
        gray, args.N, min_thick_scaled, max_thick_scaled, args.epsilon,
        center_x, center_y, fname, args.base_arc_len, args.verbose,
        args.progress, args.stroke_tolerance, not args.no_minify, args.print, args.clean, args.threshold
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an etched SVG from an image")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--output", help="Output SVG filename")
    parser.add_argument("--N", type=int, default=80, help="Approximate number of rings across width at midline")
    parser.add_argument("--base_N", type=int, default=100, help="Reference N for scaling thicknesses (default: 100)")
    parser.add_argument("--min_thick", type=float, default=0.5, help="Minimum stroke thickness")
    parser.add_argument("--max_thick", type=float, default=7.0, help="Maximum stroke thickness")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Brightness adjustment factor")
    parser.add_argument("--center_x", type=str, default="none", help="X coordinate of center ('none' to auto-center)")
    parser.add_argument("--center_y", type=str, default="none", help="Y coordinate of center ('none' to auto-center)")
    parser.add_argument("--base_arc_len", type=float, default=4.0, help="Base arc segment length")
    parser.add_argument("--stroke_tolerance", type=float, default=0.01, help="Tolerance for grouping strokes into paths")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bar")
    parser.add_argument("--no-minify", action="store_true", help="Disable SVG minification (pretty-print instead)")
    parser.add_argument("--print", action="store_true", help="Use black stroke color without metadata for printing")
    parser.add_argument("--clean", action="store_true", help="Perform in-line cleanup of anomalous segments")
    parser.add_argument("--threshold", type=float, default=500, help="Threshold in pixels for anomalous segments during cleanup (default: 500)")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    generate_etching(args)
