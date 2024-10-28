"""
ChaMo_v2: Runner script
"""

import os
import sys
from pathlib import Path

# Add the project root directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from src.main import main
    main()
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nChecking directory structure:")
    print(f"Current directory: {current_dir}")
    print("\nContents of current directory:")
    for item in current_dir.iterdir():
        print(f"  {item.name}")
    if (current_dir / 'src').exists():
        print("\nContents of src directory:")
        for item in (current_dir / 'src').iterdir():
            print(f"  {item.name}")
except Exception as e:
    print(f"Error: {e}")