# SEGY Viewer for Unix Systems

A comprehensive GUI application for browsing and visualizing SEGY seismic data files with advanced processing capabilities.

## Features

### 🗂️ **File Management**
- Browse directories and automatically detect SEGY files
- Support for multiple SEGY extensions (`.segy`, `.sgy`, `.seg`, `.dat`)
- Handles Smart Solo filename formats with dots (e.g., `00001030.00005001.00003416.segy`)
- Navigation between files with keyboard shortcuts

### 📊 **Visualization Modes**
- **Grayscale View**: Traditional seismic section display
- **Wiggle View**: Classic wiggle traces with positive fill
- Real-time switching between display modes

### 🔧 **Data Processing**
- **Background Removal**: 30-trace sliding window for noise suppression
- **Data Normalization**: Trace-by-trace amplitude normalization
- **DC Bias Removal**: Automatic mean removal from traces
- **Full Data Loading**: Complete traces loaded for zoom capability

### 🎛️ **Interactive Controls**
- **Display Clipping Slider**: Adjust contrast (0.1% to 25% percentile clipping)
- **Sample Zoom**: Focus on specific time windows (e.g., first 100 samples for time break analysis)
- **Real-time Updates**: All controls update display immediately

### ⌨️ **Navigation**
- **Keyboard Shortcuts**:
  - `→` or `Space`: Next file
  - `←`: Previous file
  - `R`: Refresh file list
  - `O`: Open file dialog
- **Mouse Controls**: Click to select, double-click to load

## Requirements

```python
segyio
matplotlib
numpy
tkinter (usually included with Python)
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/SEGY_viewer_unix.git
cd SEGY_viewer_unix
```

2. Install dependencies:
```bash
pip install segyio matplotlib numpy
```

## Usage

### Basic Usage
```bash
python main.py
```

### Command Line Options
```bash
# Open specific file
python main.py /path/to/file.segy

# Open directory
python main.py /path/to/segy/directory/
```

### Workflow
1. **Browse**: Use the directory browser or command line to open SEGY files
2. **Navigate**: Use Previous/Next buttons or keyboard arrows to browse files
3. **Analyze**: 
   - Toggle background removal for noise suppression
   - Adjust clipping slider for optimal contrast
   - Set sample limit for time break analysis (e.g., 100 samples)
   - Switch between grayscale and wiggle display modes
4. **Zoom**: Use matplotlib toolbar for detailed examination

## Key Features for Seismic Analysis

### Time Break Analysis
- Set "Max Samples" to 50-100 to focus on first arrivals
- Use wiggle view for detailed waveform analysis
- Background removal helps isolate signal from noise

### Reflection Analysis  
- Use full data view or zoom to deeper times (500-2000 samples)
- Grayscale view excellent for structural interpretation
- Adjust clipping for optimal reflection visibility

### Quality Control
- Rapidly browse through shot files using keyboard navigation
- Consistent amplitude normalization for comparison
- Real-time processing controls for different analysis needs

## Smart Solo Integration

Specifically designed to handle Smart Solo seismic system outputs:
- Supports multi-dot filename formats
- Optimized for shot gather analysis
- Background removal tuned for nodal acquisition noise patterns

## License

[Your chosen license]

## Contributing

[Contributing guidelines if desired]