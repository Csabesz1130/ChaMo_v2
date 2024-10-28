import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json

from ..io_utils.atf_handler import ATFHandler
from ..filtering.traditional_filters import (SavitzkyGolayFilter, FFTFilter, ButterworthFilter)
from ..filtering.adaptive_filters import AdaptivePatternFilter
from ..filtering.utils import calculate_signal_metrics
from .filter_controls import FilterControls

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("ChaMo v2 - Channel Analysis Tool")
        self.root.state('zoomed')

        self.atf_handler: Optional[ATFHandler] = None
        self.original_data: Optional[np.ndarray] = None
        self.filtered_data: Optional[np.ndarray] = None
        self.time_data: Optional[np.ndarray] = None
        
        self.filters = {
            'savgol': SavitzkyGolayFilter(),
            'fft': FFTFilter(),
            'butterworth': ButterworthFilter(),
            'adaptive': AdaptivePatternFilter()
        }

        self._create_menu()
        self._create_main_layout()
        self._setup_plot()

        self.status_var = tk.StringVar(value="Ready")
        self._create_status_bar()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ATF File", command=self._load_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Filtered Data", command=self._save_filtered_data, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)
        view_menu.add_command(label="Show Statistics", command=self._show_statistics_window)

        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._load_file())
        self.root.bind('<Control-s>', lambda e: self._save_filtered_data())

    def _create_main_layout(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side='left', fill='both', expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side='right', fill='y')

        self.filter_controls = FilterControls(self.control_frame, self._handle_control_event)

    def _setup_plot(self):
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def _create_status_bar(self):
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side='bottom', fill='x')

    def _load_file(self):
        try:
            filepath = filedialog.askopenfilename(
                title="Select ATF file",
                filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
            )
            
            if not filepath:
                return

            self.status_var.set(f"Loading {Path(filepath).name}...")
            self.root.update()

            self.atf_handler = ATFHandler(filepath)
            if not self.atf_handler.load_atf():
                raise ValueError("Failed to load ATF file")

            self.time_data = self.atf_handler.get_time_data()
            self.original_data = self.atf_handler.get_current_data()
            self.filtered_data = None

            self._update_plot()
            
            self.filter_controls.set_time_range(self.time_data[0], self.time_data[-1])
            
            stats = calculate_signal_metrics(self.original_data)
            self.filter_controls.update_statistics(stats)

            self.status_var.set("File loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            self.status_var.set("Error loading file")

    def _update_plot(self):
        if self.original_data is None:
            return

        self.ax.clear()
        
        self.ax.plot(self.time_data, self.original_data, label='Original Signal', alpha=0.5)
        
        if self.filtered_data is not None:
            self.ax.plot(self.time_data, self.filtered_data, label='Filtered Signal', linestyle='--')
        
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Current (pA)')
        self.ax.set_title('Signal Analysis')
        self.ax.grid(True)
        self.ax.legend()
        
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _handle_control_event(self, event: Dict[str, Any]):
        event_type = event.get('type', '')

        if event_type == 'apply_filters':
            self._apply_filters(event.get('filters', {}))
        elif event_type == 'reset_view':
            self._reset_view()
        elif event_type == 'reset_filters':
            self.filtered_data = None
            self._update_plot()
        elif event_type == 'interval_update':
            self._update_interval(event.get('start'), event.get('end'))

    def _apply_filters(self, filter_params: Dict[str, Any]):
        if self.original_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return

        try:
            self.status_var.set("Applying filters...")
            self.root.update()

            self.filtered_data = self.original_data.copy()

            if filter_params['savgol']['enabled']:
                self.filtered_data = self.filters['savgol'].filter(
                    self.filtered_data,
                    window_length=filter_params['savgol']['window_length'],
                    polyorder=filter_params['savgol']['polyorder']
                )

            if filter_params['fft']['enabled']:
                self.filtered_data = self.filters['fft'].filter(
                    self.filtered_data,
                    threshold=filter_params['fft']['threshold']
                )

            if filter_params['butterworth']['enabled']:
                self.filtered_data = self.filters['butterworth'].filter(
                    self.filtered_data,
                    cutoff=filter_params['butterworth']['cutoff'],
                    order=filter_params['butterworth']['order'],
                    fs=self.atf_handler.get_sampling_rate()
                )

            if filter_params['adaptive']['enabled']:
                self.filtered_data = self.filters['adaptive'].filter(
                    self.filtered_data,
                    window_size=filter_params['adaptive']['window_size'],
                    overlap=filter_params['adaptive']['overlap'],
                    learning_rate=filter_params['adaptive']['learning_rate']
                )

            self._update_plot()
            
            if self.filtered_data is not None:
                stats = calculate_signal_metrics(self.filtered_data)
                self.filter_controls.update_statistics(stats)

            if filter_params['adaptive']['enabled']:
                pattern_stats = self.filters['adaptive'].get_pattern_statistics()
                self.filter_controls.update_pattern_statistics(pattern_stats)

            self.status_var.set("Filters applied successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error applying filters: {str(e)}")
            self.status_var.set("Error applying filters")

    def _update_interval(self, start: float, end: float):
        if self.time_data is None:
            return

        self.ax.set_xlim(start, end)
        self.canvas.draw_idle()

    def _reset_view(self):
        if self.time_data is not None:
            self.ax.set_xlim(self.time_data[0], self.time_data[-1])
            self.canvas.draw_idle()

    def _save_filtered_data(self):
        if self.filtered_data is None:
            messagebox.showwarning("Warning", "No filtered data to save")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if filepath:
                data = np.column_stack((self.time_data, self.original_data, self.filtered_data))
                np.savetxt(filepath, data, delimiter=',', header='Time,Original,Filtered', comments='')
                messagebox.showinfo("Success", "Data saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving data: {str(e)}")

    def _show_statistics_window(self):
        if self.original_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return

        stats_window = tk.Toplevel(self.root)
        stats_window.title("Signal Statistics")
        stats_window.geometry("400x300")

        original_stats = calculate_signal_metrics(self.original_data)
        filtered_stats = calculate_signal_metrics(self.filtered_data) if self.filtered_data is not None else None

        text = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill='both', expand=True)

        text.insert('end', "Original Signal Statistics:\n\n")
        for key, value in original_stats.items():
            text.insert('end', f"{key}: {value:.3f}\n")

        if filtered_stats:
            text.insert('end', "\nFiltered Signal Statistics:\n\n")
            for key, value in filtered_stats.items():
                text.insert('end', f"{key}: {value:.3f}\n")

        text.configure(state='disabled')

    def _show_batch_process(self):
        """Show batch processing window"""
        messagebox.showinfo("Info", "Batch processing not implemented yet")

    def _export_settings(self):
        """Export current settings to file"""
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filepath:
                settings = {
                    filter_name: filter_obj.get_config()
                    for filter_name, filter_obj in self.filters.items()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(settings, f, indent=4)
                    
                messagebox.showinfo("Success", "Settings exported successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error exporting settings: {str(e)}")

    def _import_settings(self):
        """Import settings from file"""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filepath:
                with open(filepath, 'r') as f:
                    settings = json.load(f)

                for filter_name, filter_config in settings.items():
                    if filter_name in self.filters:
                        self.filters[filter_name].set_config(filter_config)

                messagebox.showinfo("Success", "Settings imported successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error importing settings: {str(e)}")

    def _show_documentation(self):
        """Show documentation window"""
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("600x400")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(doc_window)
        text_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10,
                      yscrollcommand=scrollbar.set)
        text.pack(side='left', fill='both', expand=True)
        
        scrollbar.config(command=text.yview)

        # Documentation text
        doc_text = """
ChaMo v2 - Channel Analysis Tool

This tool provides advanced signal processing capabilities for analyzing ion channel recordings.

Features:
- ATF file format support
- Multiple filtering methods:
  * Savitzky-Golay filter: Smooths data while preserving higher moments
  * FFT-based filtering: Removes frequency-domain noise
  * Butterworth filter: Low-pass filtering with custom cutoff
  * Adaptive Pattern Recognition: Learns and removes recurring noise patterns

Signal Processing Capabilities:
- Real-time signal visualization
- Interactive parameter adjustment
- Multiple filter combination
- Interval selection and analysis
- Pattern learning and recognition
- Automated noise reduction

Data Analysis Features:
- Signal statistics calculation
- SNR estimation
- Pattern detection
- Batch processing support
- Data export in multiple formats

Filter Parameters:

1. Savitzky-Golay Filter:
   - Window Length: Controls smoothing window (5-101 points)
   - Polynomial Order: Determines fitting accuracy (2-5)
   - Best for: Smooth noise reduction while preserving peaks

2. FFT Filter:
   - Threshold: Controls frequency component removal (0.01-1.0)
   - Best for: Removing periodic noise and specific frequencies

3. Butterworth Filter:
   - Cutoff Frequency: Sets frequency threshold (0.01-1.0)
   - Filter Order: Controls roll-off steepness (1-10)
   - Best for: General low-pass filtering

4. Adaptive Pattern Filter:
   - Window Size: Pattern detection length (100-2000 points)
   - Overlap: Window overlap ratio (0.1-0.9)
   - Learning Rate: Pattern adaptation speed (0.01-0.5)
   - Best for: Recurring noise patterns and adaptive learning

Usage Tips:
1. Start with loading an ATF file
2. Examine the signal characteristics
3. Choose appropriate filters based on noise type
4. Adjust parameters while observing results
5. Use interval selection for detailed analysis
6. Save settings for similar recordings
7. Export processed data for further analysis

File Formats:
- Input: Axon Text File (.atf)
- Output: CSV, JSON for processed data
- Settings: JSON for filter configurations

Keyboard Shortcuts:
- Ctrl+O: Open file
- Ctrl+S: Save filtered data
- Ctrl+R: Reset view
- Ctrl+Z: Undo last filter
- Space: Toggle interval selection

For more information and updates, visit:
https://github.com/yourusername/chamo_v2

© 2024 Your Institution
"""
        text.insert('1.0', doc_text)
        text.configure(state='disabled')

    def _show_about(self):
        """Show about dialog"""
        about_text = """
ChaMo v2 - Channel Analysis Tool
Version 2.0.0

A sophisticated signal processing tool designed for 
analyzing ion channel recordings.

Developed by:
Your Name/Institution

Copyright © 2024
All rights reserved.
"""
        messagebox.showinfo("About ChaMo v2", about_text)

    def _show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.status_var.set("Error: " + message)

    def _check_data_loaded(self) -> bool:
        """Check if data is loaded and show warning if not"""
        if self.original_data is None:
            self._show_error("No data loaded")
            return False
        return True

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()