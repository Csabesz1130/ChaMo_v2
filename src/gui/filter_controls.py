"""
ChaMo_v2: Filter control panel implementation.
Provides the user interface for controlling various filter parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Dict, Any
import json
from pathlib import Path

class FilterControls(ttk.LabelFrame):
    def __init__(self, parent, callback: Callable):
        super().__init__(parent, text="Signal Analysis Controls")
        self.callback = callback
        
        # Create notebook for better organization
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create separate tabs
        self.filter_frame = ttk.Frame(self.notebook)
        self.view_frame = ttk.Frame(self.notebook)
        self.analysis_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.filter_frame, text="Filters")
        self.notebook.add(self.view_frame, text="View")
        self.notebook.add(self.analysis_frame, text="Analysis")

    def _init_variables(self):
        """Initialize all control variables"""
        # Savitzky-Golay parameters
        self.use_savgol = tk.BooleanVar(value=False)
        self.savgol_window = tk.IntVar(value=51)
        self.savgol_polyorder = tk.IntVar(value=3)
        self.savgol_window_value = tk.StringVar(value="51")
        self.savgol_polyorder_value = tk.StringVar(value="3")

        # FFT parameters
        self.use_fft = tk.BooleanVar(value=False)
        self.fft_threshold = tk.DoubleVar(value=0.2)
        self.fft_threshold_value = tk.StringVar(value="0.20")

        # Butterworth parameters
        self.use_butter = tk.BooleanVar(value=False)
        self.butter_cutoff = tk.DoubleVar(value=0.1)
        self.butter_order = tk.IntVar(value=5)
        self.butter_cutoff_value = tk.StringVar(value="0.10")
        self.butter_order_value = tk.StringVar(value="5")

        # Adaptive filter parameters
        self.use_adaptive = tk.BooleanVar(value=False)
        self.window_size = tk.IntVar(value=1000)
        self.overlap = tk.DoubleVar(value=0.5)
        self.learning_rate = tk.DoubleVar(value=0.1)
        self.window_size_value = tk.StringVar(value="1000")
        self.overlap_value = tk.StringVar(value="0.50")
        self.learning_rate_value = tk.StringVar(value="0.10")

        # Interval selection variables
        self.use_interval = tk.BooleanVar(value=False)
        self.interval_start = tk.DoubleVar(value=0.0)
        self.interval_end = tk.DoubleVar(value=1.0)
        self.interval_start_value = tk.StringVar(value="0.00")
        self.interval_end_value = tk.StringVar(value="1.00")
        self.start_scale = None  # Initialize as None
        self.end_scale = None    # Initialize as None

        # Statistics display
        self.stats_text = tk.StringVar(value="No data loaded")
        self.pattern_stats = tk.StringVar(value="No patterns learned")

        # View mode variables
        self.view_mode = tk.StringVar(value="overlay")
        
        # Event detection variables
        self.use_event_detection = tk.BooleanVar(value=False)
        self.event_threshold = tk.DoubleVar(value=2.0)
        
        # Measurement variables
        self.show_peaks = tk.BooleanVar(value=False)
        self.show_baseline = tk.BooleanVar(value=False)
        
        # Initialize statistics text widget
        self.stats_text = None  # Will be created in _create_statistics_panel


    def _create_export_controls(self):
        export_frame = ttk.LabelFrame(self.analysis_frame, text="Export")
        export_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Data...", 
                command=self._export_data).pack(fill='x', pady=2)
        ttk.Button(export_frame, text="Export Statistics...", 
                command=self._export_stats).pack(fill='x', pady=2)
        ttk.Button(export_frame, text="Export Plot...", 
                command=self._export_plot).pack(fill='x', pady=2)
        
    def _update_view(self, *args):
        """Handle view mode changes"""
        view_params = {
            'type': 'update_view',
            'mode': self.view_mode.get()
        }
        self.callback(view_params)

    def _handle_event_detection(self, *args):
        """Handle event detection changes"""
        if self.use_event_detection.get():
            event_params = {
                'type': 'event_detection',
                'enabled': True,
                'threshold': self.event_threshold.get()
            }
            self.callback(event_params)
        else:
            self.callback({'type': 'event_detection', 'enabled': False})

    def _handle_measurement_changes(self, *args):
        """Handle measurement display changes"""
        measure_params = {
            'type': 'update_measurements',
            'show_peaks': self.show_peaks.get(),
            'show_baseline': self.show_baseline.get()
        }
        self.callback(measure_params)

    def _export_data(self):
        """Handle data export"""
        export_params = {
            'type': 'export',
            'target': 'data'
        }
        self.callback(export_params)

    def _export_stats(self):
        """Handle statistics export"""
        export_params = {
            'type': 'export',
            'target': 'statistics'
        }
        self.callback(export_params)

    def _export_plot(self):
        """Handle plot export"""
        export_params = {
            'type': 'export',
            'target': 'plot'
        }
        self.callback(export_params)

    def update_event_statistics(self, event_stats: Dict[str, Any]):
        """Update event statistics display"""
        if not event_stats:
            return
            
        stats_text = (
            f"Event Analysis:\n"
            f"Events detected: {event_stats.get('event_count', 0)}\n"
            f"Average duration: {event_stats.get('avg_duration', 0):.2f} ms\n"
            f"Average amplitude: {event_stats.get('avg_amplitude', 0):.2f} pA\n"
            f"Peak frequency: {event_stats.get('frequency', 0):.2f} Hz\n"
        )
        
        # Insert into stats text widget
        self.stats_text.insert('end', stats_text + "\n")
        self.stats_text.see('end')

    def update_measurement_display(self, measurements: Dict[str, Any]):
        """Update measurement statistics display"""
        if not measurements:
            return
            
        measure_text = (
            f"Measurements:\n"
            f"Peak count: {measurements.get('peak_count', 0)}\n"
            f"Mean peak amplitude: {measurements.get('mean_peak_amplitude', 0):.2f} pA\n"
            f"Baseline mean: {measurements.get('baseline_mean', 0):.2f} pA\n"
            f"Baseline std: {measurements.get('baseline_std', 0):.2f} pA\n"
        )
        
        # Insert into stats text widget
        self.stats_text.insert('end', measure_text + "\n")
        self.stats_text.see('end')

    def clear_statistics(self):
        """Clear statistics display"""
        self.stats_text.delete('1.0', 'end')

    def _bind_events(self):
        """Bind all control events"""
        # View controls
        self.view_mode.trace_add('write', self._update_view)
        
        # Event detection controls
        self.use_event_detection.trace_add('write', self._handle_event_detection)
        self.event_threshold.trace_add('write', self._handle_event_detection)
        
        # Measurement controls
        self.show_peaks.trace_add('write', self._handle_measurement_changes)
        self.show_baseline.trace_add('write', self._handle_measurement_changes)

    def _create_real_time_display(self):
        """Create real-time measurement display"""
        realtime_frame = ttk.LabelFrame(self.view_frame, text="Real-time Measurements")
        realtime_frame.pack(fill='x', padx=5, pady=5)
        
        self.current_value = tk.StringVar(value="Current: -- pA")
        self.baseline_value = tk.StringVar(value="Baseline: -- pA")
        self.peak_value = tk.StringVar(value="Last Peak: -- pA")
        
        ttk.Label(realtime_frame, textvariable=self.current_value).pack(anchor='w')
        ttk.Label(realtime_frame, textvariable=self.baseline_value).pack(anchor='w')
        ttk.Label(realtime_frame, textvariable=self.peak_value).pack(anchor='w')

    def update_real_time_values(self, values: Dict[str, float]):
        """Update real-time measurement display"""
        if 'current' in values:
            self.current_value.set(f"Current: {values['current']:.2f} pA")
        if 'baseline' in values:
            self.baseline_value.set(f"Baseline: {values['baseline']:.2f} pA")
        if 'peak' in values:
            self.peak_value.set(f"Last Peak: {values['peak']:.2f} pA")

    def _create_traditional_controls(self):
        """Create controls for traditional filters"""
        # Main traditional filters frame
        traditional_frame = ttk.LabelFrame(self, text="Traditional Filters")
        traditional_frame.pack(fill='x', padx=5, pady=5, ipadx=5, ipady=5)

        # Savitzky-Golay Filter
        savgol_frame = ttk.LabelFrame(traditional_frame, text="Savitzky-Golay Filter", relief="solid")
        savgol_frame.pack(fill='x', padx=5, pady=5)
        
        # Enable checkbox in its own frame
        enable_frame = ttk.Frame(savgol_frame)
        enable_frame.pack(fill='x', padx=5, pady=2)
        ttk.Checkbutton(enable_frame, text="Enable Savitzky-Golay Filter", 
                    variable=self.use_savgol,
                    command=self._apply_filters).pack(anchor='w')
        
        # Window length control
        window_frame = ttk.Frame(savgol_frame)
        window_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(window_frame, text="Window Length:", width=15).pack(side='left')
        ttk.Scale(window_frame, from_=5, to=101, variable=self.savgol_window,
                orient='horizontal',
                command=lambda v: self._update_savgol_window(float(v))
                ).pack(side='left', fill='x', expand=True)
        ttk.Label(window_frame, textvariable=self.savgol_window_value,
                width=6).pack(side='right')

        # Polynomial order control
        poly_frame = ttk.Frame(savgol_frame)
        poly_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(poly_frame, text="Polynomial Order:", width=15).pack(side='left')
        ttk.Scale(poly_frame, from_=2, to=5, variable=self.savgol_polyorder,
                orient='horizontal',
                command=lambda v: self._update_savgol_poly(float(v))
                ).pack(side='left', fill='x', expand=True)
        ttk.Label(poly_frame, textvariable=self.savgol_polyorder_value,
                width=6).pack(side='right')

        # FFT Filter
        fft_frame = ttk.LabelFrame(traditional_frame, text="FFT Filter", relief="solid")
        fft_frame.pack(fill='x', padx=5, pady=5)
        
        # Enable checkbox in its own frame
        enable_frame = ttk.Frame(fft_frame)
        enable_frame.pack(fill='x', padx=5, pady=2)
        ttk.Checkbutton(enable_frame, text="Enable FFT Filter",
                    variable=self.use_fft,
                    command=self._apply_filters).pack(anchor='w')
        
        # Threshold control
        thresh_frame = ttk.Frame(fft_frame)
        thresh_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(thresh_frame, text="Threshold:", width=15).pack(side='left')
        ttk.Scale(thresh_frame, from_=0.01, to=1.0, variable=self.fft_threshold,
                orient='horizontal',
                command=lambda v: self._update_fft_threshold(float(v))
                ).pack(side='left', fill='x', expand=True)
        ttk.Label(thresh_frame, textvariable=self.fft_threshold_value,
                width=6).pack(side='right')

        # Butterworth Filter
        butter_frame = ttk.LabelFrame(traditional_frame, text="Butterworth Filter", relief="solid")
        butter_frame.pack(fill='x', padx=5, pady=5)
        
        # Enable checkbox in its own frame
        enable_frame = ttk.Frame(butter_frame)
        enable_frame.pack(fill='x', padx=5, pady=2)
        ttk.Checkbutton(enable_frame, text="Enable Butterworth Filter",
                    variable=self.use_butter,
                    command=self._apply_filters).pack(anchor='w')
        
        # Cutoff frequency control
        cutoff_frame = ttk.Frame(butter_frame)
        cutoff_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(cutoff_frame, text="Cutoff Frequency:", width=15).pack(side='left')
        ttk.Scale(cutoff_frame, from_=0.01, to=1.0, variable=self.butter_cutoff,
                orient='horizontal',
                command=lambda v: self._update_butter_cutoff(float(v))
                ).pack(side='left', fill='x', expand=True)
        ttk.Label(cutoff_frame, textvariable=self.butter_cutoff_value,
                width=6).pack(side='right')

        # Order control
        order_frame = ttk.Frame(butter_frame)
        order_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(order_frame, text="Filter Order:", width=15).pack(side='left')
        ttk.Scale(order_frame, from_=1, to=10, variable=self.butter_order,
                orient='horizontal',
                command=lambda v: self._update_butter_order(float(v))
                ).pack(side='left', fill='x', expand=True)
        ttk.Label(order_frame, textvariable=self.butter_order_value,
                width=6).pack(side='right')
        
    def _create_view_controls(self):
        # View mode controls
        mode_frame = ttk.LabelFrame(self.view_frame, text="Display Mode")
        mode_frame.pack(fill='x', padx=5, pady=5)
        
        self.view_mode = tk.StringVar(value="overlay")
        ttk.Radiobutton(mode_frame, text="Overlay", 
                        variable=self.view_mode, value="overlay",
                        command=self._update_view).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Side by Side", 
                        variable=self.view_mode, value="sidebyside",
                        command=self._update_view).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Difference", 
                        variable=self.view_mode, value="difference",
                        command=self._update_view).pack(anchor='w')
        
    def _create_analysis_controls(self):
        # Event detection controls
        event_frame = ttk.LabelFrame(self.analysis_frame, text="Event Detection")
        event_frame.pack(fill='x', padx=5, pady=5)
        
        self.use_event_detection = tk.BooleanVar(value=False)
        self.event_threshold = tk.DoubleVar(value=2.0)
        
        ttk.Checkbutton(event_frame, text="Enable Event Detection", 
                        variable=self.use_event_detection).pack(anchor='w')
        ttk.Label(event_frame, text="Threshold:").pack(anchor='w')
        ttk.Scale(event_frame, from_=0.1, to=5.0, variable=self.event_threshold,
                orient='horizontal').pack(fill='x')
        
        # Measurement controls
        measure_frame = ttk.LabelFrame(self.analysis_frame, text="Measurements")
        measure_frame.pack(fill='x', padx=5, pady=5)
        
        self.show_peaks = tk.BooleanVar(value=False)
        self.show_baseline = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(measure_frame, text="Show Peaks", 
                        variable=self.show_peaks).pack(anchor='w')
        ttk.Checkbutton(measure_frame, text="Show Baseline", 
                        variable=self.show_baseline).pack(anchor='w')

    def _create_adaptive_controls(self):
        """Create controls for adaptive filter"""
        adaptive_frame = ttk.LabelFrame(self, text="Adaptive Filter")
        adaptive_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(adaptive_frame, text="Enable", 
                       variable=self.use_adaptive).pack(pady=2)
        
        # Window size control
        window_frame = ttk.Frame(adaptive_frame)
        window_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(window_frame, text="Window Size:").pack(side='left')
        ttk.Scale(window_frame, from_=100, to=2000, variable=self.window_size,
                 orient='horizontal',
                 command=lambda v: self.window_size_value.set(f"{float(v):.0f}")
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(window_frame, textvariable=self.window_size_value,
                 width=6).pack(side='right')

        # Overlap control
        overlap_frame = ttk.Frame(adaptive_frame)
        overlap_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(overlap_frame, text="Overlap:").pack(side='left')
        ttk.Scale(overlap_frame, from_=0.1, to=0.9, variable=self.overlap,
                 orient='horizontal',
                 command=lambda v: self.overlap_value.set(f"{float(v):.2f}")
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(overlap_frame, textvariable=self.overlap_value,
                 width=6).pack(side='right')

        # Learning rate control
        learning_frame = ttk.Frame(adaptive_frame)
        learning_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(learning_frame, text="Learning Rate:").pack(side='left')
        ttk.Scale(learning_frame, from_=0.01, to=0.5, variable=self.learning_rate,
                 orient='horizontal',
                 command=lambda v: self.learning_rate_value.set(f"{float(v):.2f}")
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(learning_frame, textvariable=self.learning_rate_value,
                 width=6).pack(side='right')

        # Pattern statistics
        ttk.Label(adaptive_frame, textvariable=self.pattern_stats,
                 wraplength=300).pack(pady=5)

    def _create_interval_controls(self):
        """Create interval selection controls"""
        interval_frame = ttk.LabelFrame(self, text="Interval Selection")
        interval_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(interval_frame, text="Enable Interval Selection",
                       variable=self.use_interval,
                       command=self._on_interval_toggle).pack(pady=2)
        
        # Start time control
        start_frame = ttk.Frame(interval_frame)
        start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(start_frame, text="Start Time (s):").pack(side='left')
        self.start_scale = ttk.Scale(start_frame, from_=0, to=10,
                                   variable=self.interval_start,
                                   orient='horizontal',
                                   command=self._update_interval_display)
        self.start_scale.pack(side='left', fill='x', expand=True)
        ttk.Label(start_frame, textvariable=self.interval_start_value,
                 width=8).pack(side='right')
        
        # End time control
        end_frame = ttk.Frame(interval_frame)
        end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(end_frame, text="End Time (s):").pack(side='left')
        self.end_scale = ttk.Scale(end_frame, from_=0, to=10,
                                 variable=self.interval_end,
                                 orient='horizontal',
                                 command=self._update_interval_display)
        self.end_scale.pack(side='left', fill='x', expand=True)
        ttk.Label(end_frame, textvariable=self.interval_end_value,
                 width=8).pack(side='right')

    def _create_statistics_panel(self):
        """Create statistics display panel"""
        stats_frame = ttk.LabelFrame(self, text="Statistics")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(stats_frame)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.stats_text = tk.Text(text_frame, height=6, width=40, wrap='word')
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', 
                                command=self.stats_text.yview)
        
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initial text
        self.stats_text.insert('1.0', "No data loaded")
        self.stats_text.configure(state='disabled')  # Make read-only

    def _update_savgol_window(self, value):
        """Update Savitzky-Golay window value"""
        self.savgol_window_value.set(f"{value:.0f}")
        if self.use_savgol.get():
            self._apply_filters()

    def _update_savgol_poly(self, value):
        """Update Savitzky-Golay polynomial order value"""
        self.savgol_polyorder_value.set(f"{value:.0f}")
        if self.use_savgol.get():
            self._apply_filters()

    def _update_fft_threshold(self, value):
        """Update FFT threshold value"""
        self.fft_threshold_value.set(f"{value:.2f}")
        if self.use_fft.get():
            self._apply_filters()

    def _update_butter_cutoff(self, value):
        """Update Butterworth cutoff value"""
        self.butter_cutoff_value.set(f"{value:.2f}")
        if self.use_butter.get():
            self._apply_filters()

    def _update_butter_order(self, value):
        """Update Butterworth order value"""
        self.butter_order_value.set(f"{value:.0f}")
        if self.use_butter.get():
            self._apply_filters()

    def _apply_filters(self):
        """Trigger filter application"""
        self.callback({
            'type': 'apply_filters',
            'filters': {
                'savgol': {
                    'enabled': self.use_savgol.get(),
                    'window_length': self.savgol_window.get(),
                    'polyorder': self.savgol_polyorder.get()
                },
                'fft': {
                    'enabled': self.use_fft.get(),
                    'threshold': self.fft_threshold.get()
                },
                'butterworth': {
                    'enabled': self.use_butter.get(),
                    'cutoff': self.butter_cutoff.get(),
                    'order': self.butter_order.get()
                }
            }
        })

    def _create_buttons(self):
        """Create control buttons"""
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text="Apply Filters",
                  command=self._apply_filters).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Reset View",
                  command=self._reset_view).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Reset Filters",
                  command=self._reset_filters).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Save Settings",
                  command=self._save_settings).pack(side='right', padx=2)
        ttk.Button(button_frame, text="Load Settings",
                  command=self._load_settings).pack(side='right', padx=2)

    def _on_interval_toggle(self):
        """Handle interval selection toggle"""
        try:
            if self.use_interval.get():
                # Ensure scales are properly initialized
                if self.start_scale is not None and self.end_scale is not None:
                    self.callback({'type': 'interval_toggle', 'enabled': True})
            else:
                self.callback({'type': 'interval_toggle', 'enabled': False})
        except Exception as e:
            print(f"Error in interval toggle: {e}")
            self.use_interval.set(False)

    def _update_interval_display(self, _=None):
        """Update interval display and notify callback"""
        start = self.interval_start.get()
        end = self.interval_end.get()
        
        self.interval_start_value.set(f"{start:.2f}")
        self.interval_end_value.set(f"{end:.2f}")
        
        if self.use_interval.get():
            self.callback({
                'type': 'interval_update',
                'start': start,
                'end': end
            })

    def _apply_filters(self):
        """Collect all filter parameters and notify callback"""
        params = {
            'type': 'apply_filters',
            'filters': {
                'savgol': {
                    'enabled': self.use_savgol.get(),
                    'window_length': self.savgol_window.get(),
                    'polyorder': self.savgol_polyorder.get()
                },
                'fft': {
                    'enabled': self.use_fft.get(),
                    'threshold': self.fft_threshold.get()
                },
                'butterworth': {
                    'enabled': self.use_butter.get(),
                    'cutoff': self.butter_cutoff.get(),
                    'order': self.butter_order.get()
                },
                'adaptive': {
                    'enabled': self.use_adaptive.get(),
                    'window_size': self.window_size.get(),
                    'overlap': self.overlap.get(),
                    'learning_rate': self.learning_rate.get()
                }
            },
            'interval': {
                'enabled': self.use_interval.get(),
                'start': self.interval_start.get(),
                'end': self.interval_end.get()
            }
        }
        self.callback(params)
    
    def _reset_view(self):
        """Reset view to original state"""
        self.callback({'type': 'reset_view'})

    def _reset_filters(self):
        """Reset all filter parameters to defaults"""
        # Reset Savitzky-Golay
        self.use_savgol.set(False)
        self.savgol_window.set(51)
        self.savgol_polyorder.set(3)
        self.savgol_window_value.set("51")
        self.savgol_polyorder_value.set("3")

        # Reset FFT
        self.use_fft.set(False)
        self.fft_threshold.set(0.2)
        self.fft_threshold_value.set("0.20")

        # Reset Butterworth
        self.use_butter.set(False)
        self.butter_cutoff.set(0.1)
        self.butter_order.set(5)
        self.butter_cutoff_value.set("0.10")
        self.butter_order_value.set("5")

        # Reset Adaptive
        self.use_adaptive.set(False)
        self.window_size.set(1000)
        self.overlap.set(0.5)
        self.learning_rate.set(0.1)
        self.window_size_value.set("1000")
        self.overlap_value.set("0.50")
        self.learning_rate_value.set("0.10")

        # Reset interval selection
        self.use_interval.set(False)
        self.interval_start.set(0.0)
        self.interval_end.set(1.0)
        self.interval_start_value.set("0.00")
        self.interval_end_value.set("1.00")

        # Notify callback
        self.callback({'type': 'reset_filters'})

    def _save_settings(self):
        """Save current filter settings to file"""
        settings = {
            'savgol': {
                'enabled': self.use_savgol.get(),
                'window_length': self.savgol_window.get(),
                'polyorder': self.savgol_polyorder.get()
            },
            'fft': {
                'enabled': self.use_fft.get(),
                'threshold': self.fft_threshold.get()
            },
            'butterworth': {
                'enabled': self.use_butter.get(),
                'cutoff': self.butter_cutoff.get(),
                'order': self.butter_order.get()
            },
            'adaptive': {
                'enabled': self.use_adaptive.get(),
                'window_size': self.window_size.get(),
                'overlap': self.overlap.get(),
                'learning_rate': self.learning_rate.get()
            },
            'interval': {
                'enabled': self.use_interval.get(),
                'start': self.interval_start.get(),
                'end': self.interval_end.get()
            }
        }

        try:
            settings_path = Path('settings.json')
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def _load_settings(self):
        """Load filter settings from file"""
        try:
            settings_path = Path('settings.json')
            if not settings_path.exists():
                messagebox.showwarning("Warning", "No saved settings found.")
                return

            with open(settings_path, 'r') as f:
                settings = json.load(f)

            # Apply loaded settings
            if 'savgol' in settings:
                self.use_savgol.set(settings['savgol']['enabled'])
                self.savgol_window.set(settings['savgol']['window_length'])
                self.savgol_polyorder.set(settings['savgol']['polyorder'])
                self.savgol_window_value.set(str(settings['savgol']['window_length']))
                self.savgol_polyorder_value.set(str(settings['savgol']['polyorder']))

            if 'fft' in settings:
                self.use_fft.set(settings['fft']['enabled'])
                self.fft_threshold.set(settings['fft']['threshold'])
                self.fft_threshold_value.set(f"{settings['fft']['threshold']:.2f}")

            if 'butterworth' in settings:
                self.use_butter.set(settings['butterworth']['enabled'])
                self.butter_cutoff.set(settings['butterworth']['cutoff'])
                self.butter_order.set(settings['butterworth']['order'])
                self.butter_cutoff_value.set(f"{settings['butterworth']['cutoff']:.2f}")
                self.butter_order_value.set(str(settings['butterworth']['order']))

            if 'adaptive' in settings:
                self.use_adaptive.set(settings['adaptive']['enabled'])
                self.window_size.set(settings['adaptive']['window_size'])
                self.overlap.set(settings['adaptive']['overlap'])
                self.learning_rate.set(settings['adaptive']['learning_rate'])
                self.window_size_value.set(str(settings['adaptive']['window_size']))
                self.overlap_value.set(f"{settings['adaptive']['overlap']:.2f}")
                self.learning_rate_value.set(f"{settings['adaptive']['learning_rate']:.2f}")

            if 'interval' in settings:
                self.use_interval.set(settings['interval']['enabled'])
                self.interval_start.set(settings['interval']['start'])
                self.interval_end.set(settings['interval']['end'])
                self.interval_start_value.set(f"{settings['interval']['start']:.2f}")
                self.interval_end_value.set(f"{settings['interval']['end']:.2f}")

            messagebox.showinfo("Success", "Settings loaded successfully!")
            # Notify callback of loaded settings
            self._apply_filters()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

    def update_statistics(self, stats: Dict[str, Any]):
        """Update statistics display"""
        if not stats:
            stats_text = "No data loaded"
        else:
            stats_text = (
                f"Signal Statistics:\n"
                f"Mean: {stats.get('mean', 0):.2f}\n"
                f"Std: {stats.get('std', 0):.2f}\n"
                f"RMS: {stats.get('rms', 0):.2f}\n"
                f"Peak-to-Peak: {stats.get('peak_to_peak', 0):.2f}\n"
                f"SNR: {stats.get('snr', 0):.2f} dB"
            )
        
        # Update text widget
        self.stats_text.configure(state='normal')  # Allow editing
        self.stats_text.delete('1.0', 'end')
        self.stats_text.insert('1.0', stats_text)
        self.stats_text.configure(state='disabled')  # Make read-only again

    def update_pattern_statistics(self, stats: Dict[str, Any]):
        """Update pattern statistics display"""
        if not stats:
            self.pattern_stats.set("No patterns learned")
            return

        stats_text = (
            f"Learned Patterns: {stats.get('total_patterns', 0)}\n"
            f"Average Confidence: {stats.get('average_confidence', 0):.2f}\n"
            f"Total Observations: {stats.get('total_observations', 0)}"
        )
        self.pattern_stats.set(stats_text)

    def set_time_range(self, start: float, end: float):
        """Set the available time range for interval selection"""
        try:
            # Update scale ranges
            if hasattr(self, 'start_scale') and self.start_scale is not None:
                self.start_scale.configure(to=end)
                self.end_scale.configure(to=end)

            # Update values if interval selection is not active
            if not self.use_interval.get():
                self.interval_start.set(start)
                self.interval_end.set(end)
                self.interval_start_value.set(f"{start:.2f}")
                self.interval_end_value.set(f"{end:.2f}")

        except Exception as e:
            print(f"Error setting time range: {e}")
            # Try to recover by resetting values
            self.interval_start.set(0.0)
            self.interval_end.set(1.0)
            self.interval_start_value.set("0.00")
            self.interval_end_value.set("1.00")