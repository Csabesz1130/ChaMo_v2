# ChaMo_v2

Channel Modeling Software Version 2 - Advanced signal processing and noise reduction for ion channel recordings.

## Project Structure

`
ChaMo_v2/
├── src/
│   ├── io_utils/
│   │   ├── __init__.py
│   │   └── atf_handler.py
│   ├── filtering/
│   │   ├── __init__.py
│   │   ├── base_filter.py
│   │   ├── traditional_filters.py
│   │   ├── adaptive_filters.py
│   │   └── utils.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   └── filter_controls.py
│   └── main.py
├── data/
│   └── noise_patterns/
├── tests/
└── README.md
`

## Setup

1. Create a virtual environment:
   `ash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   `

2. Install requirements:
   `ash
   pip install -r requirements.txt
   `

## Features

- ATF file format support
- Traditional filtering methods (Savitzky-Golay, FFT, Butterworth)
- Adaptive noise cancellation with pattern learning
- Interactive GUI for real-time signal analysis
- Advanced pattern recognition for noise reduction
- Automated signal quality enhancement
- Export capabilities for processed data

## License

© 2024 All rights reserved.
#   C h a M o _ v 2  
 