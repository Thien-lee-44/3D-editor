# 3D Scene Editor & Synthetic Data Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3%2B-5586A4?logo=opengl&logoColor=white)
![PySide6](https://img.shields.io/badge/Qt-PySide6-41CD52?logo=qt&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A robust, lightweight, and open-source 3D Graphics Engine and Level Editor built entirely in Python. Designed for educational purposes, rapid prototyping, and procedural geometry experimentation, this engine leverages the power of OpenGL for rendering and PySide6 (Qt) for a highly interactive, professional-grade authoring interface.

At its core, the engine is driven by a strict Entity-Component System (ECS) architecture, ensuring that data and logic are cleanly decoupled, making the system highly extensible and maintainable.

---

## Core Architecture & Features

### 1. Software Architecture
- **Entity-Component System (ECS):** A decoupled architecture where Entities act as containers and functionality is dynamically added through Components (Transform, Mesh, Light, Camera, Animation, Semantic).
- **The Ultimate Dynamic Facade:** A centralized interface (`Engine` class) that strictly decouples and streamlines communication between the Editor UI and the core Engine backend.
- **Structural Caching & Render Queues:** Optimized scene management that flattens hierarchical data and sorts renderables (Opaque/Transparent) to ensure a fast, artifact-free rendering loop.
- **Memento Pattern:** Comprehensive state management that allows users to capture, store, and restore previous versions of the scene (Undo/Redo).

### 2. Graphics & Rendering
- **Forward Rendering Pipeline:** Custom GLSL shaders implementing the Phong shading model for consistent and realistic surface illumination.
- **Dynamic Lighting:** Full support for various light sources, including Directional (Sun), Point, and Spot lights with adjustable attenuation and cutoff parameters.
- **Advanced Material System:** Full multi-texturing support including Diffuse, Specular, Ambient (AO), Emission, Shininess, Opacity, Bump/Normal, and Reflection Maps.
- **Procedural Geometry:** Integrated generator capable of creating smooth 3D surfaces directly from user-defined mathematical equations.

### 3. Editor & User Interface
- **Interactive 3D Viewport:** Hardware-accelerated canvas featuring intuitive 3D Gizmos (Translate, Rotate, Scale), GPU Color-Picking raycasting, and a navigation HUD.
- **Anti-Lag Inspector:** A context-aware editor that provides real-time control over an entity's components with optimized performance and safe thread execution.
- **Asset Browser:** User-friendly file management supporting drag-and-drop for importing `.obj` and `.ply` models and applying textures directly onto entities.
- **Project Serialization:** Robust system for saving and loading entire project states, including scene hierarchy and UI configurations via JSON.
- **OBJ Exporter:** Ability to bake and export the current 3D scene into standard Wavefront (`.obj` and `.mtl`) formats for external use.

### 4. Animation & Timeline System
- **Non-Linear Editor (NLE) Timeline:** Professional Dope Sheet interface featuring a precise time ruler and interactive keyframe tracks.
- **Advanced Keyframe Manipulation:** Box selection, duplication, time-scaling, and moving of keyframe ranges using mouse and modifier keys (Ctrl/Alt/Shift).
- **Real-time Interpolation:** Smooth mathematical interpolation of Transform, Light, and Material properties synchronized to the target framerate.

---

## Synthetic Data Engine & AI Benchmarking

### Synthetic Export Outputs
The generator exports all core artifacts expected by modern Computer Vision standards:
- `images/*.jpg` (RGB frames)
- `depth/*.png` and `depth/*.npy` (Depth map visualization + raw depth array)
- `masks/instance/*.png` and `masks/semantic/*.png`
- `labels/*.txt` (YOLO detection)
- `annotations/instances_coco.json` (COCO)
- `annotations/voc/*.xml` (Pascal VOC, additional format)
- `metadata/frames.json`, `metadata/frames.ndjson`, `metadata/objects.csv`
- `annotations/export_manifest.json` (Index of exported formats)

### Run CV Benchmark from Preview UI
In **Synthetic Data Preview** mode, use the right panel section **CV Benchmark (Preview)**:
1. Select/confirm the dataset output directory (must contain `dataset.yaml`).
2. Choose a task (`auto`, `detect`, `segment`) and an optional model.
3. Click **RUN CV BENCHMARK (CURRENT DATASET)**.
> *The UI shows benchmark metrics and artifact paths for summary/CSV/JSON plus Ground Truth vs. Prediction comparison outputs*.

---

## Installation & Setup

### Prerequisites
- **Python 3.8 or higher**.
- A dedicated GPU or integrated graphics capable of supporting **OpenGL 3.3+**.

### Step-by-Step Guide

**1. Clone the repository:**
```bash
git clone -b synthetic-data-engine https://github.com/Thien-lee-44/3D-editor.git
cd 3D-editor
```

**2. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch the Editor:**
```bash
python run.py
```

## Synthetic Export Outputs

The generator now exports all core artifacts:

- `images/*.jpg` (RGB frames)
- `depth/*.png` and `depth/*.npy` (depth map visualization + raw depth array)
- `masks/instance/*.png` and `masks/semantic/*.png`
- `labels/*.txt` (YOLO detection)
- `annotations/instances_coco.json` (COCO)
- `annotations/voc/*.xml` (Pascal VOC, additional format)
- `metadata/frames.json`, `metadata/frames.ndjson`, `metadata/objects.csv`
- `annotations/export_manifest.json` (index of exported formats)

## Run CV Benchmark from Preview UI

In **Synthetic Data Preview** mode, use the right panel section **CV Benchmark (Preview)**:

1. Select/confirm the dataset output directory (must contain `dataset.yaml`).
2. Choose task (`auto`, `detect`, `segment`) and optional model.
3. Click **RUN CV BENCHMARK (CURRENT DATASET)**.

The UI shows benchmark metrics and artifact paths for summary/CSV/JSON plus GT-vs-Pred comparison outputs.

## User Guide & Controls

### Viewport Navigation

| Action | Input |
|---|---|
| Orbit Camera | Hold Right Mouse Button + Drag |
| Pan Camera | Hold Middle Mouse Button + Drag |
| Zoom | Mouse Scroll Wheel |
| Free-Flight | Hold Right Mouse Button + W, A, S, D, Q, E |

### Editor Interaction

| Action | Input |
|---|---|
| Select Entity | Left Click on any object (GPU Raycasting) or select from Hierarchy |
| Transform Tool | Use Toolbar radio buttons (Move, Rotate, Scale) |
| Copy / Paste | Ctrl + C / Ctrl + V |
| Cut Entity | Ctrl + X |
| Delete Entity | Delete / Backspace |
| Undo / Redo | Ctrl + Z / Ctrl + Y |

## Project Directory Structure

```plaintext
3D_Editor/
├── projects/                   # Saved project workspaces (.json)
├── assets/                     # Raw physical assets & Resources
│   ├── models/                 # Predefined primitives and Editor proxies
│   ├── scene/                  # Sample environment assets and materials
│   ├── shaders/                # GLSL vertex and fragment shaders
│   └── textures/               # Default image maps
└── src/
    ├── app/                    # Application Entry, Configurations (SSOT), Exceptions
    ├── engine/                 # ================= CORE RUNTIME ENGINE =================
    │   ├── engine.py           # Dynamic Facade API
    │   ├── core/               # Input & Transform Managers
    │   ├── geometry/           # Primitives & Procedural Math Surfaces
    │   ├── graphics/           # OpenGL Buffers, Shaders, Materials, Render Queues
    │   ├── resources/          # Asset loading pipeline and RAM/VRAM Caching
    │   ├── scene/              # ECS Architecture (Entities, Components, SceneManager)
    │   │   ├── components/     # (Transform, Mesh, Light, Camera, Semantic, Animation)
    │   │   └── managers/       # Specialized ECS System Managers
    │   └── synthetic/          # AI Data Generator, CV Benchmarks, Exporters (YOLO/COCO)
    └── ui/                     # ================= PYSIDE6 AUTHORING GUI =================
        ├── controllers/        # Logic Controllers (Asset, Project, Timeline workflows)
        ├── error_handler.py    # Centralized UI Thread Error Dispatcher
        ├── main_window.py      # Main Application Window and Layout Orchestrator
        ├── views/              # Main Window, Dialogs, Viewport Canvas, and Dock Panels
        └── widgets/            # Custom UI Components (Inspector, Timeline, Dope Sheet)
```

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page to get involved.

## License

This project is open-source and available under the MIT License.
