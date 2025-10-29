import subprocess
import os

# Create basic network using SUMO's netgenerate tool
print("Generating network using SUMO netgenerate...")

cmd = [
    'netgenerate',
    '--grid',
    '--grid.x-number=1',
    '--grid.y-number=1', 
    '--grid.x-length=200',
    '--grid.y-length=200',
    '--grid.attach-length=100',
    '--default.lanenumber=2',
    '--default.speed=13.89',
    '--tls.guess=true',
    '--output-file=network/intersection.net.xml'
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] Network generated successfully!")
        print("Now you can run: python src/train.py")
    else:
        print("Error:", result.stderr)
except FileNotFoundError:
    print("ERROR: 'netgenerate' not found. Make sure SUMO is installed and in your PATH.")
    print("After installing SUMO, restart your computer and try again.")