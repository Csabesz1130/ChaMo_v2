import tkinter as tk
import logging
from pathlib import Path

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "chamo.log"),
            logging.StreamHandler()
        ]
    )

def main():
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting ChaMo_v2...")
        root = tk.Tk()
        from src.gui.main_window import MainWindow  # Changed import statement
        app = MainWindow(root)
        
        logger.info("Application started successfully")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
