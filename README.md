# polarhalftone

**polarhalftone** is a Python toolchain that converts bitmap images into **circular halftone line art**, then optimizes the resulting SVGs for **PrusaSlicer**, **laser engraving**, or **plotter** workflows.

It works by sampling brightness in polar coordinates, generating a ring-based halftone pattern that preserves tonal variation while producing clean, slice-ready vector geometry.

---

## ✦ Features

- 🌀 **Polar Halftone Generation**
  - Converts grayscale or color images into ring-based halftone structures.
  - Adjustable number of rings, line thickness scaling, and tonal mapping.
- ⚙️ **PrusaSlicer-Safe SVG Output**
  - Removes malformed or anomalous path segments.
  - Splits long paths to avoid rendering or slicing issues.
  - Preserves all original SVG attributes and numeric precision.
- 🎨 **High-Fidelity Vector Output**
  - Floating-point coordinates (6-decimal precision).
  - Consistent stroke geometry suitable for engraving or plotting.
- 🧩 **CLI-Friendly**
  - Command-line arguments for full control of thresholds, scaling, and verbosity.

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/polarhalftone.git
cd polarhalftone
pip install -r requirements.txt
