#!/usr/bin/env python3
"""
SEGY Viewer for Unix Systems
A GUI application for browsing and visualizing SEGY seismic data files
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import segyio

# Supported SEGY file extensions (case-sensitive for better performance)
SEGY_EXTENSIONS = ('.segy', '.sgy', '.seg', '.SEGY', '.SGY', '.SEG', 
                   '.Segy', '.Sgy', '.Seg', '.dat', '.DAT')

class SEGYReader:
    """Class to handle SEGY file reading and data extraction"""
    
    def __init__(self):
        self.current_file = None
        self.data = None
        self.header = None
        
    def load_file(self, filepath):
        """Load a SEGY file and extract data using the proven method"""
        try:
            with segyio.open(filepath, 'r', ignore_geometry=True) as f:
                # Get basic file information
                trace_count = f.tracecount
                n_samples = f.samples.size
                
                print(f"Loading SEGY file: {os.path.basename(filepath)}")
                print(f"Total traces in file: {trace_count}")
                print(f"Samples per trace: {n_samples}")
                
                # Load all traces in full length
                traces = []
                
                for i in range(trace_count):
                    trace_data = f.trace[i]
                    traces.append(trace_data)
                
                if traces:
                    self.data = np.array(traces)
                    print(f"Loaded data shape: {self.data.shape}")
                    print(f"Data range: {np.min(self.data):.2e} to {np.max(self.data):.2e}")
                else:
                    raise ValueError("No traces found in SEGY file")
                
                # Extract header information using the same approach as your script
                samples = f.samples
                
                # Get sample rate safely
                try:
                    sample_interval = f.bin[segyio.BinField.Interval]
                    sample_rate = sample_interval / 1000.0 if sample_interval and sample_interval > 0 else 1.0
                except:
                    sample_rate = 1.0  # Default sample rate
                
                self.header = {
                    'samples': samples,
                    'sample_rate': sample_rate,
                    'traces': trace_count,
                    'samples_per_trace': n_samples,
                    'filename': os.path.basename(filepath),
                    'total_traces_in_file': trace_count
                }
                
                self.current_file = filepath
                return True
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SEGY file:\n{str(e)}")
            return False
    
    def get_trace_data(self, trace_idx=None):
        """Get trace data for visualization"""
        if self.data is None:
            return None
            
        if trace_idx is not None:
            # Return specific trace
            if trace_idx < len(self.data):
                return self.data[trace_idx]
            else:
                return None
        else:
            # Return all data for 2D visualization (traces x samples)
            return self.data

class SEGYViewer:
    """Main GUI application for SEGY file viewing"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SEGY Viewer")
        self.root.geometry("1200x800")
        
        # Initialize SEGY reader
        self.segy_reader = SEGYReader()
        
        # Current directory and file list
        self.current_directory = os.getcwd()
        self.segy_files = []
        self.current_file_index = 0
        
        # Setup GUI
        self.setup_gui()
        self.refresh_file_list()
        
    def setup_gui(self):
        """Create the main GUI layout"""
        # Create main frames
        self.create_menu()
        self.create_toolbar()
        self.create_main_panels()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Directory", command=self.open_directory)
        file_menu.add_command(label="Open File", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh", command=self.refresh_file_list)
        view_menu.add_command(label="Next File", command=self.next_file)
        view_menu.add_command(label="Previous File", command=self.previous_file)
        
    def create_toolbar(self):
        """Create toolbar with navigation buttons"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Directory selection
        ttk.Label(toolbar, text="Directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.dir_var = tk.StringVar(value=self.current_directory)
        dir_entry = ttk.Entry(toolbar, textvariable=self.dir_var, width=50)
        dir_entry.pack(side=tk.LEFT, padx=(0, 5))
        dir_entry.bind('<Return>', lambda e: self.change_directory())
        
        ttk.Button(toolbar, text="Browse", command=self.open_directory).pack(side=tk.LEFT, padx=(0, 10))
        
        # Navigation buttons
        ttk.Button(toolbar, text="Previous", command=self.previous_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next", command=self.next_file).pack(side=tk.LEFT, padx=2)
        
        # File info
        self.file_info_var = tk.StringVar(value="No file loaded")
        ttk.Label(toolbar, textvariable=self.file_info_var).pack(side=tk.RIGHT, padx=10)
        
    def create_main_panels(self):
        """Create main content panels"""
        # Create paned window for resizable panels
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - File browser
        self.create_file_browser(main_paned)
        
        # Right panel - Visualization
        self.create_visualization_panel(main_paned)
        
    def create_file_browser(self, parent):
        """Create file browser panel"""
        browser_frame = ttk.LabelFrame(parent, text="SEGY Files", width=300)
        parent.add(browser_frame, weight=1)
        
        # File listbox with scrollbar
        list_frame = ttk.Frame(browser_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Bind selection event
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        self.file_listbox.bind('<Double-Button-1>', self.on_file_double_click)
        
        # Button frame for file operations
        button_frame = ttk.Frame(browser_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(button_frame, text="Load Selected File", 
                  command=self.load_current_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Auto-load First", 
                  command=self.auto_load_first_file).pack(side=tk.LEFT, padx=2)
        
        # File count label
        self.file_count_var = tk.StringVar(value="0 files")
        ttk.Label(browser_frame, textvariable=self.file_count_var).pack(pady=2)
        
    def create_visualization_panel(self, parent):
        """Create visualization panel with matplotlib"""
        viz_frame = ttk.LabelFrame(parent, text="Seismic Data Visualization")
        parent.add(viz_frame, weight=3)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create clipping control frame
        clipping_frame = ttk.Frame(viz_frame)
        clipping_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # Clipping percentile slider
        ttk.Label(clipping_frame, text="Display Clipping:").pack(side=tk.LEFT)
        
        self.clipping_var = tk.DoubleVar(value=5.0)  # Default 5th percentile
        self.clipping_scale = ttk.Scale(clipping_frame, from_=0.1, to=25.0, 
                                       variable=self.clipping_var, 
                                       orient=tk.HORIZONTAL, length=150,
                                       command=self.on_clipping_change)
        self.clipping_scale.pack(side=tk.LEFT, padx=5)
        
        self.clipping_label = ttk.Label(clipping_frame, text="5.0%")
        self.clipping_label.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(clipping_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Sample limit control
        ttk.Label(clipping_frame, text="Max Samples:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.sample_limit_var = tk.StringVar(value="")  # Empty means show all
        self.sample_limit_entry = ttk.Entry(clipping_frame, textvariable=self.sample_limit_var, 
                                           width=8)
        self.sample_limit_entry.pack(side=tk.LEFT, padx=2)
        self.sample_limit_entry.bind('<Return>', self.on_sample_limit_change)
        self.sample_limit_entry.bind('<FocusOut>', self.on_sample_limit_change)
        
        ttk.Button(clipping_frame, text="Apply", 
                  command=self.apply_sample_limit).pack(side=tk.LEFT, padx=2)
        ttk.Button(clipping_frame, text="Show All", 
                  command=self.reset_sample_limit).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(clipping_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Display mode toggle
        ttk.Label(clipping_frame, text="Display:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.display_mode_var = tk.StringVar(value="grayscale")
        self.display_mode_combo = ttk.Combobox(clipping_frame, textvariable=self.display_mode_var,
                                              values=["grayscale", "wiggle"], 
                                              state="readonly", width=10)
        self.display_mode_combo.pack(side=tk.LEFT, padx=2)
        self.display_mode_combo.bind('<<ComboboxSelected>>', self.on_display_mode_change)
        
        # Separator
        ttk.Separator(clipping_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Background removal toggle
        self.background_removal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(clipping_frame, text="Background Removal (30 traces)", 
                       variable=self.background_removal_var,
                       command=self.on_background_removal_change).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(clipping_frame, text="Reset", 
                  command=self.reset_clipping).pack(side=tk.RIGHT, padx=5)
        
        # Create navigation toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.nav_toolbar.update()
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'No SEGY file loaded\n\nDouble-click a file in the list\nor use "Load Selected File" button', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_title('SEGY Data Viewer')
        
    def open_directory(self):
        """Open directory dialog and change to selected directory"""
        directory = filedialog.askdirectory(initialdir=self.current_directory)
        if directory:
            self.current_directory = directory
            self.dir_var.set(directory)
            self.refresh_file_list()
            
    def change_directory(self):
        """Change to directory specified in entry widget"""
        new_dir = self.dir_var.get()
        if os.path.isdir(new_dir):
            self.current_directory = new_dir
            self.refresh_file_list()
        else:
            messagebox.showerror("Error", f"Directory not found: {new_dir}")
            self.dir_var.set(self.current_directory)
            
    def refresh_file_list(self):
        """Refresh the list of SEGY files in current directory"""
        self.segy_files = []
        
        if os.path.isdir(self.current_directory):
            for file in os.listdir(self.current_directory):
                if file.endswith(SEGY_EXTENSIONS):
                    self.segy_files.append(file)
        
        # Sort files alphabetically
        self.segy_files.sort()
        
        # Update listbox
        self.file_listbox.delete(0, tk.END)
        for file in self.segy_files:
            self.file_listbox.insert(tk.END, file)
            
        # Update file count
        self.file_count_var.set(f"{len(self.segy_files)} files")
        
        # Reset current file index
        self.current_file_index = 0
        
    def on_file_select(self, event):
        """Handle file selection in listbox"""
        selection = event.widget.curselection()
        if selection:
            self.current_file_index = selection[0]
            
    def on_file_double_click(self, event):
        """Handle double-click on file to load it"""
        selection = event.widget.curselection()
        if selection:
            self.current_file_index = selection[0]
            self.load_current_file()
            
    def open_file(self):
        """Open file dialog to select a specific SEGY file"""
        filetypes = [
            ('SEGY files', '*.segy *.sgy *.seg *.SEGY *.SGY *.SEG *.dat *.DAT'),
            ('All files', '*.*')
        ]
        filename = filedialog.askopenfilename(
            initialdir=self.current_directory,
            title="Select SEGY file",
            filetypes=filetypes
        )
        if filename:
            # Update directory and file list
            self.current_directory = os.path.dirname(filename)
            self.dir_var.set(self.current_directory)
            self.refresh_file_list()
            
            # Select the file in the list
            basename = os.path.basename(filename)
            if basename in self.segy_files:
                self.current_file_index = self.segy_files.index(basename)
                self.file_listbox.selection_clear(0, tk.END)
                self.file_listbox.selection_set(self.current_file_index)
                self.load_current_file()
                
    def load_current_file(self):
        """Load and visualize the currently selected SEGY file"""
        if not self.segy_files:
            messagebox.showinfo("Info", "No SEGY files available. Please select a directory with SEGY files.")
            return
            
        if self.current_file_index >= len(self.segy_files):
            messagebox.showerror("Error", "Invalid file index. Please select a file from the list.")
            return
            
        filename = self.segy_files[self.current_file_index]
        filepath = os.path.join(self.current_directory, filename)
        
        # Show loading message
        self.file_info_var.set(f"Loading {filename}...")
        self.root.update()
        
        if self.segy_reader.load_file(filepath):
            self.visualize_data()
            self.update_file_info()
        else:
            self.file_info_var.set("Failed to load file")
            
    def auto_load_first_file(self):
        """Automatically load the first file in the list"""
        if self.segy_files:
            self.current_file_index = 0
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(0)
            self.file_listbox.see(0)
            self.load_current_file()
        else:
            messagebox.showinfo("Info", "No SEGY files found in the current directory")
            
    def visualize_data(self):
        """Visualize the loaded SEGY data"""
        data = self.segy_reader.get_trace_data()
        if data is None:
            return
            
        # Clear previous plot and colorbar
        self.ax.clear()
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass  # Ignore errors when removing colorbar
            self.colorbar = None
        
        # Reset the figure layout to prevent shrinking
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        # Plot seismic data
        if len(data.shape) == 2 and data.shape[0] > 1:
            # 2D seismic section (multiple traces)
            # Apply some basic processing to improve visualization
            data_viz = data.copy()
            
            # Remove DC bias (mean) from each trace
            for i in range(data_viz.shape[0]):
                data_viz[i] = data_viz[i] - np.mean(data_viz[i])
            
            # Apply background removal if enabled
            if self.background_removal_var.get():
                data_viz = self.apply_background_removal(data_viz)
                
            # Normalize the data
            data_viz = self.normalize_data(data_viz)
            
            # Store original data shape for zoom reference
            self.original_shape = data_viz.shape
            
            # Apply sample limit for display (zoom functionality)
            sample_limit_text = self.sample_limit_var.get().strip()
            if sample_limit_text and sample_limit_text.isdigit():
                sample_limit = min(int(sample_limit_text), data_viz.shape[1])
                data_viz = data_viz[:, :sample_limit]
            
            # Calculate percentile-based clipping using slider value
            clipping_percent = self.clipping_var.get()
            vmin, vmax = np.percentile(data_viz, [clipping_percent, 100-clipping_percent])
            
            # Check display mode
            display_mode = self.display_mode_var.get()
            
            if display_mode == "wiggle":
                # Wiggle trace display
                self.plot_wiggle_traces(data_viz, vmin, vmax)
            else:
                # Grayscale display (default)
                im = self.ax.imshow(data_viz.T, aspect='auto', cmap='gray', 
                                  interpolation='bilinear', origin='upper',
                                  vmin=vmin, vmax=vmax)
                self.ax.set_xlabel('Trace Number')
                self.ax.set_ylabel('Sample Number')
                
                # Add colorbar safely
                try:
                    self.colorbar = self.fig.colorbar(im, ax=self.ax, label='Amplitude')
                except:
                    pass  # Continue without colorbar if it fails
            
        else:
            # 1D trace or single trace
            trace_data = data[0] if len(data.shape) == 2 else data
            
            # Remove DC bias
            trace_data = trace_data - np.mean(trace_data)
            
            # Normalize single trace
            max_abs = np.max(np.abs(trace_data))
            if max_abs > 0:
                trace_data = trace_data / max_abs
            
            # Apply sample limit for display (zoom functionality)
            sample_limit_text = self.sample_limit_var.get().strip()
            if sample_limit_text and sample_limit_text.isdigit():
                sample_limit = min(int(sample_limit_text), len(trace_data))
                trace_data = trace_data[:sample_limit]
            
            self.ax.plot(trace_data)
            self.ax.set_xlabel('Sample Number')
            self.ax.set_ylabel('Amplitude')
            self.ax.grid(True, alpha=0.3)
            
        # Set title with file info
        header = self.segy_reader.header
        title = f"{header['filename']}\n"
        title += f"Traces: {header['traces']}, Samples: {header['samples_per_trace']}"
        if 'sample_rate' in header and header['sample_rate']:
            title += f", Sample Rate: {header['sample_rate']:.1f} ms"
        
        # Add zoom info if sample limit is applied
        sample_limit_text = self.sample_limit_var.get().strip()
        if sample_limit_text and sample_limit_text.isdigit():
            title += f" (Showing samples 0-{sample_limit_text})"
            
        self.ax.set_title(title)
        
        # Reset figure layout and refresh canvas
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()
        
    def update_file_info(self):
        """Update file information display"""
        if self.segy_reader.header:
            info = f"File {self.current_file_index + 1}/{len(self.segy_files)}: {self.segy_reader.header['filename']}"
            self.file_info_var.set(info)
        
    def next_file(self):
        """Load next SEGY file"""
        if self.segy_files and self.current_file_index < len(self.segy_files) - 1:
            self.current_file_index += 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_file_index)
            self.file_listbox.see(self.current_file_index)
            self.load_current_file()
            
    def previous_file(self):
        """Load previous SEGY file"""
        if self.segy_files and self.current_file_index > 0:
            self.current_file_index -= 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_file_index)
            self.file_listbox.see(self.current_file_index)
            self.load_current_file()
            
    def on_clipping_change(self, value):
        """Handle clipping slider change"""
        clipping_value = float(value)
        self.clipping_label.config(text=f"{clipping_value:.1f}%")
        # Re-visualize data with new clipping if a file is loaded
        if self.segy_reader.data is not None:
            self.visualize_data()
            
    def reset_clipping(self):
        """Reset clipping to default value"""
        self.clipping_var.set(5.0)
        self.clipping_label.config(text="5.0%")
        # Re-visualize data with default clipping if a file is loaded
        if self.segy_reader.data is not None:
            self.visualize_data()
            
    def on_sample_limit_change(self, event=None):
        """Handle sample limit entry change"""
        self.apply_sample_limit()
        
    def apply_sample_limit(self):
        """Apply the sample limit from the entry box"""
        sample_text = self.sample_limit_var.get().strip()
        if sample_text:
            try:
                sample_limit = int(sample_text)
                if sample_limit > 0:
                    # Re-visualize data with new sample limit if a file is loaded
                    if self.segy_reader.data is not None:
                        self.visualize_data()
                else:
                    messagebox.showerror("Error", "Sample limit must be a positive number")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for sample limit")
        else:
            # Empty means show all samples
            if self.segy_reader.data is not None:
                self.visualize_data()
                
    def reset_sample_limit(self):
        """Reset sample limit to show all samples"""
        self.sample_limit_var.set("")
        # Re-visualize data without sample limit if a file is loaded
        if self.segy_reader.data is not None:
            self.visualize_data()
            
    def on_display_mode_change(self, event=None):
        """Handle display mode change"""
        # Re-visualize data with new display mode if a file is loaded
        if self.segy_reader.data is not None:
            self.visualize_data()
            
    def on_background_removal_change(self):
        """Handle background removal toggle"""
        # Re-visualize data with/without background removal if a file is loaded
        if self.segy_reader.data is not None:
            self.visualize_data()
            
    def apply_background_removal(self, data_viz):
        """Apply background removal using a 30-trace sliding window"""
        window_size = 30
        n_traces, n_samples = data_viz.shape
        
        if n_traces < window_size:
            # If fewer traces than window size, use all traces for background
            background = np.mean(data_viz, axis=0)
            return data_viz - background[np.newaxis, :]
        
        # Create output array
        processed_data = data_viz.copy()
        
        # Apply sliding window background removal
        for i in range(n_traces):
            # Define window boundaries
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n_traces, i + window_size // 2 + 1)
            
            # Ensure window is exactly 30 traces when possible
            if end_idx - start_idx < window_size:
                if start_idx == 0:
                    end_idx = min(n_traces, window_size)
                else:
                    start_idx = max(0, n_traces - window_size)
            
            # Calculate background from window (excluding current trace)
            window_traces = np.concatenate([
                data_viz[start_idx:i, :], 
                data_viz[i+1:end_idx, :]
            ], axis=0) if i > start_idx and i < end_idx - 1 else data_viz[start_idx:end_idx, :]
            
            if window_traces.shape[0] > 0:
                background = np.mean(window_traces, axis=0)
                processed_data[i, :] = data_viz[i, :] - background
            
        return processed_data
        
    def normalize_data(self, data_viz):
        """Normalize seismic data for consistent amplitude display"""
        # Choose normalization method based on data characteristics
        normalization_method = "trace"  # Options: "trace", "global", "rms"
        
        if normalization_method == "trace":
            # Trace-by-trace normalization (preserves relative amplitudes within each trace)
            normalized_data = data_viz.copy()
            for i in range(data_viz.shape[0]):
                trace_max = np.max(np.abs(data_viz[i, :]))
                if trace_max > 0:
                    normalized_data[i, :] = data_viz[i, :] / trace_max
                    
        elif normalization_method == "global":
            # Global normalization (preserves relative amplitudes between traces)
            global_max = np.max(np.abs(data_viz))
            normalized_data = data_viz / global_max if global_max > 0 else data_viz
            
        elif normalization_method == "rms":
            # RMS normalization (balances trace energies)
            normalized_data = data_viz.copy()
            for i in range(data_viz.shape[0]):
                trace_rms = np.sqrt(np.mean(data_viz[i, :] ** 2))
                if trace_rms > 0:
                    normalized_data[i, :] = data_viz[i, :] / trace_rms
        
        return normalized_data
            
    def plot_wiggle_traces(self, data_viz, vmin, vmax):
        """Plot seismic data as wiggle traces"""
        n_traces, n_samples = data_viz.shape
        
        # Limit number of traces for performance (max 200 traces)
        max_traces = min(200, n_traces)
        if n_traces > max_traces:
            # Subsample traces evenly
            trace_indices = np.linspace(0, n_traces-1, max_traces, dtype=int)
            data_viz = data_viz[trace_indices, :]
            n_traces = max_traces
        else:
            trace_indices = np.arange(n_traces)
        
        # Calculate scaling factor
        max_amp = np.max(np.abs(data_viz))
        if max_amp > 0:
            scale_factor = 0.8 / max_amp  # Scale to 80% of trace spacing
        else:
            scale_factor = 1.0
        
        # Plot each trace
        for i in range(n_traces):
            trace = data_viz[i, :]
            trace_offset = trace_indices[i] if n_traces < 200 else i * (n_traces / max_traces)
            
            # Scale and offset the trace
            scaled_trace = trace * scale_factor
            
            # Plot the zero line
            self.ax.plot([trace_offset, trace_offset], [0, n_samples-1], 
                        color='lightgray', linewidth=0.5)
            
            # Plot the wiggle trace
            self.ax.plot(trace_offset + scaled_trace, np.arange(n_samples), 
                        color='black', linewidth=0.8)
            
            # Fill positive amplitudes
            self.ax.fill_betweenx(np.arange(n_samples), trace_offset, 
                                trace_offset + scaled_trace,
                                where=(scaled_trace > 0), 
                                color='black', alpha=0.7)
        
        # Set labels and limits
        self.ax.set_xlabel('Trace Number')
        self.ax.set_ylabel('Sample Number')
        self.ax.set_xlim(-0.5, (n_traces-1) + 0.5)
        self.ax.set_ylim(n_samples-1, 0)  # Invert y-axis
        self.ax.grid(True, alpha=0.3)

def main():
    """Main function to run the SEGY viewer application"""
    root = tk.Tk()
    
    # Bring window to front and ensure it gets focus
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()
    
    # Set up keyboard shortcuts
    def on_key_press(event):
        if event.keysym == 'Right' or event.keysym == 'space':
            app.next_file()
        elif event.keysym == 'Left':
            app.previous_file()
        elif event.keysym == 'r':
            app.refresh_file_list()
        elif event.keysym == 'o':
            app.open_file()
            
    root.bind('<Key>', on_key_press)
    root.focus_set()  # Make sure window can receive key events
    
    # Create and run application
    app = SEGYViewer(root)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.isfile(arg_path) and arg_path.endswith(SEGY_EXTENSIONS):
            # Open specific file
            app.current_directory = os.path.dirname(os.path.abspath(arg_path))
            app.dir_var.set(app.current_directory)
            app.refresh_file_list()
            basename = os.path.basename(arg_path)
            if basename in app.segy_files:
                app.current_file_index = app.segy_files.index(basename)
                app.file_listbox.selection_set(app.current_file_index)
                app.load_current_file()
        elif os.path.isdir(arg_path):
            # Open directory
            app.current_directory = os.path.abspath(arg_path)
            app.dir_var.set(app.current_directory)
            app.refresh_file_list()
    
    root.mainloop()

if __name__ == "__main__":
    main()

