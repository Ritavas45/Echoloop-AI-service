import sys
import os

# Add src/ directory to Python path so internal imports function seamlessly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator import main

if __name__ == "__main__":
    main()
