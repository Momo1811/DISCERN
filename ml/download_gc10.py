
import dataset_tools as dtools
from pathlib import Path

# DISCERN raw dataset directory
destination = Path(r"D:\DISCERN\ml\data\raw")

# Create the folder if it does not exist
destination.mkdir(parents=True, exist_ok=True)

print("Starting GC10-DET download...")
print(f"Destination: {destination}")

# Download GC10-DET
dtools.download(
    dataset="GC10-DET",
    dst_dir=str(destination)
)

print("\nDownload process completed.")
print(f"Dataset location: {destination}")
