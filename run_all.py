import subprocess

station_ids = ["84370","84401", "84501", "84593"]

for sid in station_ids:
    print(f"Procesando estación {sid}...")
    subprocess.run(["python", "main.py", sid])
