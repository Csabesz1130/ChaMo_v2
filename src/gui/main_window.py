"""
ChaMo_v2: Main application window implementation.
Provides the main GUI window for the signal analysis application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json

from ..io_utils.atf_handler import ATFHandler
from ..filtering.traditional_filters import (SavitzkyGolayFilter, 
                                          FFTFilter, 
                                          ButterworthFilter)
from ..filtering.adaptive_filters import AdaptivePatternFilter
from ..filtering.utils import calculate_signal_metrics
from .filter_controls import FilterControls

class MainWindow:
    def __init__(self, root):
        """Initialize the main window"""
        self.root = root
        self.root.title("ChaMo v2 - Channel Analysis Tool")
        self.root.state('zoomed')  # Start maximized

        # Data storage
        self.atf_handler: Optional[ATFHandler] = None
        self.original_data: Optional[np.ndarray] = None
        self.filtered_data: Optional[np.ndarray] = None
        self.time_data: Optional[np.ndarray] = None
        
        # Filter instances
        self.filters = {
            'savgol': SavitzkyGolayFilter(),
            'fft': FFTFilter(),
            'butterworth': ButterworthFilter(),
            'adaptive': AdaptivePatternFilter()
        }

        # Create main layout first
        self._setup_menu()  # Changed from _create_menu to _setup_menu
        self._create_main_layout()
        self._setup_plot()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self._create_status_bar()

        # Initialize keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _setup_menu(self):
        """Setup the main menu bar"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open File...", 
                                command=lambda: self._load_file(),
                                accelerator="Ctrl+O")
        self.file_menu.add_command(label="Save Filtered Data...", 
                                command=lambda: self._save_filtered_data(),
                                accelerator="Ctrl+S")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", 
                                command=self.root.quit,
                                accelerator="Esc")

        # View menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Reset View", 
                                command=lambda: self._reset_view(),
                                accelerator="Ctrl+R")
        self.view_menu.add_command(label="Show Statistics", 
                                command=lambda: self._show_statistics_window())

        # Tools menu
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=self.tools_menu)
        self.tools_menu.add_command(label="Batch Process", 
                                command=lambda: self._show_batch_process())
        self.tools_menu.add_command(label="Export Settings", 
                                command=lambda: self._export_settings())
        self.tools_menu.add_command(label="Import Settings", 
                                command=lambda: self._import_settings())

        # Help menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Documentation", 
                                command=lambda: self._show_documentation())
        self.help_menu.add_command(label="About", 
                                command=lambda: self._show_about())
        
    def _load_file(self, event=None):
        """Load an ATF file"""
        try:
            filepath = filedialog.askopenfilename(
                title="Select ATF file",
                filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
            )
            
            if not filepath:
                return

            self.status_var.set(f"Loading {Path(filepath).name}...")
            self.root.update()

            # Load ATF file
            self.atf_handler = ATFHandler(filepath)
            if not self.atf_handler.load_atf():
                raise ValueError("Failed to load ATF file")

            # Get data
            self.time_data = self.atf_handler.get_time_data()
            self.original_data = self.atf_handler.get_current_data()
            self.filtered_data = None

            # Update plot
            self._update_plot()
            
            # Update controls
            if hasattr(self, 'filter_controls'):
                self.filter_controls.set_time_range(
                    self.time_data[0], 
                    self.time_data[-1]
                )
                
                # Update statistics
                stats = calculate_signal_metrics(self.original_data)
                self.filter_controls.update_statistics(stats)

            self.status_var.set("File loaded successfully")
            return True

        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            return False

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self._load_file())
        self.root.bind('<Control-s>', lambda e: self._save_filtered_data())
        self.root.bind('<Control-r>', lambda e: self._reset_view())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        # Add more shortcuts
        self.root.bind('<Control-z>', lambda e: self._undo_last_action())
        self.root.bind('<Control-y>', lambda e: self._redo_last_action())
        self.root.bind('<space>', lambda e: self._toggle_interval_selection())

    # Add these utility methods
    def _undo_last_action(self):
        """Undo last filter operation"""
        # Implement undo functionality
        pass

    def _redo_last_action(self):
        """Redo last undone operation"""
        # Implement redo functionality
        pass

    def _toggle_interval_selection(self):
        """Toggle interval selection mode"""
        if hasattr(self, 'filter_controls'):
            current = self.filter_controls.use_interval.get()
            self.filter_controls.use_interval.set(not current)

    def _create_menu(self):
        """Create the main menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self._load_file)
        file_menu.add_command(label="Save Filtered Data", 
                            command=self._save_filtered_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)
        view_menu.add_command(label="Show Statistics", 
                            command=self._show_statistics_window)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Batch Process", 
                             command=self._show_batch_process)
        tools_menu.add_command(label="Export Settings", 
                             command=self._export_settings)
        tools_menu.add_command(label="Import Settings", 
                             command=self._import_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", 
                            command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create the main application layout"""
        try:
            # Main container using grid
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.grid(row=0, column=0, sticky='nsew')
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.main_frame.grid_columnconfigure(0, weight=3)  # Plot gets more space
            self.main_frame.grid_columnconfigure(1, weight=1)  # Controls get less space

            # Plot frame
            self.plot_frame = ttk.Frame(self.main_frame)
            self.plot_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

            # Controls frame with scrollbar
            controls_container = ttk.Frame(self.main_frame)
            controls_container.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
            
            # Create scrollable frame for controls
            canvas = tk.Canvas(controls_container)
            scrollbar = ttk.Scrollbar(controls_container, orient="vertical", command=canvas.yview)
            self.control_frame = ttk.Frame(canvas)

            # Configure scrolling
            self.control_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Pack canvas and scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Add filter controls
            self.filter_controls = FilterControls(self.control_frame, self._handle_control_event)
            
        except Exception as e:
            messagebox.showerror("Layout Error", f"Error creating layout: {str(e)}")
            raise

    def _handle_control_event(self, event: Dict[str, Any]):
        """Handle control panel events"""
        event_type = event.get('type', '')

        if event_type == 'update_view':
            self._update_view_mode(event.get('mode'))
        elif event_type == 'event_detection':
            self._handle_event_detection(event)
        elif event_type == 'update_measurements':
            self._update_measurements(event)
        elif event_type == 'export':
            self._handle_export(event.get('target'))
        elif event_type == 'apply_filters':
            self._apply_filters(event.get('filters', {}))
        elif event_type == 'reset_view':
            self._reset_view()
        elif event_type == 'reset_filters':
            self.filtered_data = None
            self._update_plot()
        elif event_type == 'interval_update':
            self._update_interval(event.get('start'), event.get('end'))

    def _apply_filters(self, filter_params: Dict[str, Any]):
        """Apply selected filters to the data"""
        if self.original_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return

        try:
            self.status_var.set("Applying filters...")
            self.root.update()

            # Start with original data
            self.filtered_data = self.original_data.copy()

            # Apply Savitzky-Golay filter
            if filter_params['savgol']['enabled']:
                self.filtered_data = self.filters['savgol'].filter(
                    self.filtered_data,
                    window_length=filter_params['savgol']['window_length'],
                    polyorder=filter_params['savgol']['polyorder']
                )

            # Apply FFT filter
            if filter_params['fft']['enabled']:
                self.filtered_data = self.filters['fft'].filter(
                    self.filtered_data,
                    threshold=filter_params['fft']['threshold']
                )

            # Apply Butterworth filter
            if filter_params['butterworth']['enabled']:
                self.filtered_data = self.filters['butterworth'].filter(
                    self.filtered_data,
                    cutoff=filter_params['butterworth']['cutoff'],
                    order=filter_params['butterworth']['order'],
                    fs=self.atf_handler.get_sampling_rate()
                )

            # Apply Adaptive filter
            if filter_params['adaptive']['enabled']:
                self.filtered_data = self.filters['adaptive'].filter(
                    self.filtered_data,
                    window_size=filter_params['adaptive']['window_size'],
                    overlap=filter_params['adaptive']['overlap'],
                    learning_rate=filter_params['adaptive']['learning_rate']
                )

            # Update plot and statistics
            self._update_plot()
            if self.filtered_data is not None:
                stats = calculate_signal_metrics(self.filtered_data)
                self.filter_controls.update_statistics(stats)

            # Update pattern statistics if adaptive filter was used
            if filter_params['adaptive']['enabled']:
                pattern_stats = self.filters['adaptive'].get_pattern_statistics()
                self.filter_controls.update_pattern_statistics(pattern_stats)

            self.status_var.set("Filters applied successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error applying filters: {str(e)}")
            self.status_var.set("Error applying filters")

    def _update_interval(self, start: float, end: float):
        """Update plot to show selected interval"""
        if self.time_data is None:
            return

        self.ax.set_xlim(start, end)
        self.canvas.draw_idle()

    def _reset_view(self):
        """Reset plot view to show all data"""
        if self.time_data is not None:
            self.ax.set_xlim(self.time_data[0], self.time_data[-1])
            self.canvas.draw_idle()

    def _save_filtered_data(self):
        """Save filtered data to file"""
        if self.filtered_data is None:
            messagebox.showwarning("Warning", "No filtered data to save")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if filepath:
                data = np.column_stack((self.time_data, 
                                      self.original_data, 
                                      self.filtered_data))
                np.savetxt(filepath, data, delimiter=',',
                          header='Time,Original,Filtered',
                          comments='')
                messagebox.showinfo("Success", "Data saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving data: {str(e)}")

    def _show_statistics_window(self):
        """Show detailed statistics window"""
        if self.original_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return

        stats_window = tk.Toplevel(self.root)
        stats_window.title("Signal Statistics")
        stats_window.geometry("400x300")

        # Calculate statistics
        original_stats = calculate_signal_metrics(self.original_data)
        filtered_stats = (calculate_signal_metrics(self.filtered_data) 
                        if self.filtered_data is not None else None)

        # Create text widget
        text = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill='both', expand=True)

        # Add statistics
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
    
    def _update_view_mode(self, mode: str):
        """Handle view mode changes"""
        if self.original_data is None:
            return

        self.ax.clear()
        
        if mode == "overlay":
            self._show_overlay_view()
        elif mode == "sidebyside":
            self._show_sidebyside_view()
        elif mode == "difference":
            self._show_difference_view()
            
        self.canvas.draw_idle()

    def _show_overlay_view(self):
        """Show original and filtered signals overlaid"""
        self.ax.plot(self.time_data, self.original_data, 
                    label='Original Signal', alpha=0.5)
        
        if self.filtered_data is not None:
            self.ax.plot(self.time_data, self.filtered_data, 
                        label='Filtered Signal', linestyle='--')
        
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Current (pA)')
        self.ax.set_title('Signal Analysis - Overlay View')
        self.ax.grid(True)
        self.ax.legend()

    def _show_sidebyside_view(self):
        """Show original and filtered signals side by side"""
        # Clear current axis and create two subplots
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        
        # Plot original signal
        ax1.plot(self.time_data, self.original_data, label='Original Signal')
        ax1.set_title('Original Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Current (pA)')
        ax1.grid(True)
        
        # Plot filtered signal if available
        if self.filtered_data is not None:
            ax2.plot(self.time_data, self.filtered_data, 
                    label='Filtered Signal', color='orange')
        ax2.set_title('Filtered Signal')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Current (pA)')
        ax2.grid(True)
        
        self.fig.tight_layout()

    def _show_difference_view(self):
        """Show difference between original and filtered signals"""
        if self.filtered_data is None:
            self.ax.plot(self.time_data, self.original_data, 
                        label='Original Signal')
        else:
            difference = self.original_data - self.filtered_data
            self.ax.plot(self.time_data, difference, 
                        label='Difference', color='red')
            
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Difference (pA)')
            self.ax.set_title('Signal Difference (Original - Filtered)')
            self.ax.grid(True)
            self.ax.legend()

    def _handle_event_detection(self, event: Dict[str, Any]):
        """Handle event detection"""
        if not self.original_data is not None:
            return
            
        if event['enabled']:
            events = self._detect_events(event['threshold'])
            self.filter_controls.update_event_statistics(events)
            self._plot_events(events)
        else:
            self._clear_event_markers()

    def _detect_events(self, threshold: float) -> Dict[str, Any]:
        """Detect events in the signal"""
        if self.filtered_data is not None:
            data = self.filtered_data
        else:
            data = self.original_data
            
        # Calculate baseline and std
        baseline = np.median(data)
        noise_std = np.std(data - baseline)
        
        # Find events (peaks above threshold * std)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(data, height=baseline + threshold * noise_std,
                                     width=1, distance=10)
        
        # Calculate event properties
        events = {
            'event_count': len(peaks),
            'peak_indices': peaks,
            'peak_heights': properties['peak_heights'],
            'avg_amplitude': np.mean(properties['peak_heights']) if len(peaks) > 0 else 0,
            'baseline': baseline,
            'noise_std': noise_std
        }
        
        return events

    def _plot_events(self, events: Dict[str, Any]):
        """Plot detected events"""
        if hasattr(self, 'event_markers'):
            for marker in self.event_markers:
                marker.remove()
                
        self.event_markers = []
        
        # Plot peak markers
        marker = self.ax.plot(self.time_data[events['peak_indices']], 
                            events['peak_heights'], 'ro', 
                            label='Detected Events')[0]
        self.event_markers.append(marker)
        
        # Plot baseline
        baseline = self.ax.axhline(y=events['baseline'], color='g', 
                                 linestyle='--', label='Baseline')
        self.event_markers.append(baseline)
        
        self.ax.legend()
        self.canvas.draw_idle()

    def _clear_event_markers(self):
        """Clear event markers from plot"""
        if hasattr(self, 'event_markers'):
            for marker in self.event_markers:
                marker.remove()
            self.event_markers = []
            self.canvas.draw_idle()

    def _update_measurements(self, event: Dict[str, Any]):
        """Handle measurement updates"""
        if self.original_data is None:
            return
            
        if event['show_peaks']:
            self._show_peak_markers()
        else:
            self._hide_peak_markers()
            
        if event['show_baseline']:
            self._show_baseline()
        else:
            self._hide_baseline()
            
        self.canvas.draw_idle()

    def _show_peak_markers(self):
        """Show peak markers on the plot"""
        if self.filtered_data is not None:
            data = self.filtered_data
        else:
            data = self.original_data
            
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, height=np.mean(data), distance=10)
        
        if hasattr(self, 'peak_markers'):
            self._hide_peak_markers()
            
        self.peak_markers = self.ax.plot(self.time_data[peaks], data[peaks], 
                                       'ro', label='Peaks')[0]
        self.ax.legend()

    def _hide_peak_markers(self):
        """Hide peak markers"""
        if hasattr(self, 'peak_markers'):
            self.peak_markers.remove()
            delattr(self, 'peak_markers')
            self.ax.legend()

    def _show_baseline(self):
        """Show baseline on the plot"""
        if self.filtered_data is not None:
            data = self.filtered_data
        else:
            data = self.original_data
            
        baseline = np.median(data)
        
        if hasattr(self, 'baseline_line'):
            self._hide_baseline()
            
        self.baseline_line = self.ax.axhline(y=baseline, color='g', 
                                           linestyle='--', label='Baseline')
        self.ax.legend()

    def _hide_baseline(self):
        """Hide baseline"""
        if hasattr(self, 'baseline_line'):
            self.baseline_line.remove()
            delattr(self, 'baseline_line')
            self.ax.legend()

    def _handle_export(self, target: str):
        """Handle export requests"""
        if self.original_data is None:
            messagebox.showwarning("Warning", "No data to export")
            return
            
        if target == 'data':
            self._export_data()
        elif target == 'statistics':
            self._export_statistics()
        elif target == 'plot':
            self._export_plot()

    def _export_data(self):
        """Export data to CSV"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                data = {
                    'Time': self.time_data,
                    'Original': self.original_data
                }
                
                if self.filtered_data is not None:
                    data['Filtered'] = self.filtered_data
                    
                import pandas as pd
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", "Data exported successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def _export_statistics(self):
        """Export statistics to JSON"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                stats = {
                    'original': calculate_signal_metrics(self.original_data)
                }
                
                if self.filtered_data is not None:
                    stats['filtered'] = calculate_signal_metrics(self.filtered_data)
                    
                with open(filepath, 'w') as f:
                    json.dump(stats, f, indent=4)
                    
                messagebox.showinfo("Success", "Statistics exported successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")

    def _export_plot(self):
        """Export plot as image"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), 
                      ("PDF files", "*.pdf"),
                      ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Plot exported successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

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