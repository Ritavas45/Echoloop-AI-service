"""
Generate and save the implementation guide as markdown
"""

import os
from IMPLEMENTATION import IMPLEMENTATION_GUIDE

# Create docs directory
os.makedirs("./docs", exist_ok=True)

# Save implementation guide
with open("./docs/IMPLEMENTATION.md", "w") as f:
    f.write(IMPLEMENTATION_GUIDE)

print("✓ Implementation guide saved to ./docs/IMPLEMENTATION.md")

# Also save to root
with open("./IMPLEMENTATION.md", "w") as f:
    f.write(IMPLEMENTATION_GUIDE)

print("✓ Implementation guide saved to ./IMPLEMENTATION.md")
