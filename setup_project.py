from pathlib import Path

# Create folders inside existing project
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Create files
files = [
    "app.py",
    "requirements.txt",
    "README.md"
]

for file in files:
    Path(file).touch(exist_ok=True)

print("Project structure created successfully!")