# 🚗 YOLOv8 Bidirectional Vehicle Counting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An intelligent, real-time vehicle counting system powered by YOLOv8 for bidirectional traffic analysis**

[Features](#features) • [Demo](#demo) • [Installation](#installation--setup) • [Usage](#usage) • [Configuration](#configuration)

</div>

---

## 📖 About The Project

This project utilizes the state-of-the-art **YOLOv8** (You Only Look Once) object detection model to perform real-time detection, tracking, and counting of vehicles from CCTV or video footage. It specifically counts vehicles moving in two defined directions (Incoming and Outgoing) and categorizes them by type.

Perfect for traffic monitoring, parking management, and intelligent transportation systems.

---

## 📚 Documentation

For detailed theory, technical documentation, and comprehensive explanation of the code implementation, please refer to:

**[📄 Documentation.pdf](./Documentation.pdf)** - Complete guide covering:
- YOLOv8 architecture and theory
- Object tracking algorithms (ByteTrack)
- Implementation details and code explanation
- Mathematical foundations
- Advanced configuration options

---

## 🎬 Demo

Watch the system in action! The video demonstrates real-time vehicle detection, tracking, and counting:

**[📹 Click Here to Watch Demo Video](https://youtu.be/d_X8XPncZn8)**

<div align="center">
  
[![YOLOv8 Vehicle Counting Demo](https://img.youtube.com/vi/d_X8XPncZn8/0.jpg)](https://youtu.be/d_X8XPncZn8)

*Click the thumbnail above to watch the demo video*

</div>

---

## ✨ Features


*   ✅ **High-Accuracy Detection**: Uses the `yolov8l.pt` (Large) model for superior detection accuracy.
*   🎯 **Real-Time Tracking**: Implements **ByteTrack** for robust object tracking across frames, ensuring vehicles are not recounted.
*   🔄 **Bidirectional Counting**: Distinguishes between "Incoming" (downward moving) and "Outgoing" (upward moving) traffic based on a configurable virtual line.
*   🚙 **Multi-Class Classification**: Detects and counts specific vehicle types:
    *   Car
    *   Motorbike
    *   Bus
    *   Truck
*   📊 **Live Dashboard**: Displays real-time counts and totals directly on the video feed with a large, readable dashboard.
*   📁 **Data Export**: Automatically saves the final counting session results to an **Excel (.xlsx)** file for reporting and analysis.
*   🔧 **Noise Filtering**: Includes size filtering to ignore small objects or false positives.

---

## 🛠️ Requirements

| Requirement | Details |
|------------|---------|
| **Python Version** | 3.8 or higher |
| **Operating System** | Windows / Linux / macOS |
| **GPU** | Optional (Recommended for faster processing) |

### Dependencies

The following Python packages are required (all listed in `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 framework |
| `opencv-python` | Computer Vision tasks |
| `pandas` | Data manipulation and Excel export |
| `numpy` | Numerical operations |
| `torch` | Deep Learning backend |
| `openpyxl` | Excel file support |

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/yolov8-vehicle-counting.git
cd yolov8-vehicle-counting
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Model Weights

The system uses `yolov8l.pt`. If not present, the `ultralytics` library will automatically download it on the first run. Ensure you have an internet connection or place the `.pt` file in the root directory.

---

## ⚙️ Configuration

Before running the script, adjust the configuration paths in `YOLO_PROJECT/Incoming_Outgoing.py` to match your local environment.

### Configuration Parameters

Open `YOLO_PROJECT/Incoming_Outgoing.py` and modify:

```python
# --- Paths and System Setup ---
VIDEO_PATH = r"C:\path\to\your\video.mp4"  # Replace with your video file path
EXCEL_PATH = r"C:\path\to\output\results.xlsx" # Replace with desired Excel output path

# --- Counting Line Setup ---
LINE_POSITION = 1500 # Adjust the Y-coordinate for the counting line based on your video resolution
```

| Parameter | Description |
|-----------|-------------|
| `VIDEO_PATH` | Absolute path to your CCTV footage or input video |
| `EXCEL_PATH` | Where you want the final report to be saved |
| `LINE_POSITION` | The vertical pixel position of the counting line (adjust based on video resolution) |

---

## 🎮 Usage

### Running the System

To start the vehicle counting system, run the main script from the root directory:

```bash
python YOLO_PROJECT/Incoming_Outgoing.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Stop processing, close window, and save Excel report |

---

## 📂 Project Structure

```
yolov8-vehicle-counting/
│
├── YOLO_PROJECT/
│   └── Incoming_Outgoing.py    # Core detection, tracking & counting logic
│
├── Documentation.pdf            # Detailed theory and technical documentation
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── yolov8l.pt                   # YOLOv8 model weights (auto-downloaded)
```

---

## 📊 Output

After the session ends (video finishes or user quits), the program generates an Excel file at the specified `EXCEL_PATH`.

### Output Format

| Column | Description |
|--------|-------------|
| **Direction** | Incoming or Outgoing |
| **Vehicle Type** | Car, Bus, Motorbike, Truck |
| **Count** | Total number of unique vehicles counted for that category |

### Sample Output

| Direction | Vehicle Type | Count |
|-----------|-------------|-------|
| Incoming  | Car         | 45    |
| Incoming  | Motorbike   | 12    |
| Outgoing  | Bus         | 3     |
| Outgoing  | Truck       | 8     |

---

## ⚠️ Limitations

While this system provides robust vehicle counting capabilities, please be aware of the following limitations:

*   **Fixed Counting Line**: The system uses a single horizontal line for counting. This may not be optimal for:
    *   Curved roads or intersections
    *   Multi-lane highways where vehicles change lanes frequently
    *   Scenarios requiring multiple counting zones

*   **Occlusion & Overlapping**: Detection accuracy may decrease when:
    *   Vehicles are heavily overlapping or occluding each other
    *   Dense traffic conditions with significant vehicle clustering
    *   Large vehicles (buses/trucks) block smaller vehicles (motorbikes/cars)

*   **Environmental & Hardware Dependencies**: Performance is affected by:
    *   **Video Quality**: Low resolution, poor lighting, or severe weather conditions (heavy rain, fog, snow) can reduce detection accuracy
    *   **Camera Angle**: Extreme angles or high-mounted cameras may impact tracking reliability
    *   **Processing Power**: Real-time processing requires adequate CPU/GPU resources; slower hardware may cause frame drops

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
*   [OpenCV](https://opencv.org/) - Computer vision library
*   [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking

---

<div align="center">

**Made with ❤️ for Traffic Analysis**

⭐ **If this project helped you, please give it a star!** ⭐

</div>

