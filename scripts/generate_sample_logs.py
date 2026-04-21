from pathlib import Path
from src.inference.predict import predict_rul

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / 'logs').exists():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise RuntimeError("Could not find project root containing /logs")
    PROJECT_ROOT = PROJECT_ROOT.parent

LOG_PATH = PROJECT_ROOT / "logs" / "inference.jsonl"

if LOG_PATH.exists():
    LOG_PATH.unlink()

sample_inputs = [
    {"unit_number": 3, "sensor_4": 1392.35, "sensor_11": 47.26, "sensor_15": 8.4046}, # short life engine, high rul
    {"unit_number": 3, "sensor_4": 1416.95, "sensor_11": 47.63, "sensor_15": 8.4479}, # short lived engine, mid rul
    {"unit_number": 3, "sensor_4": 1423.50, "sensor_11": 48.10, "sensor_15": 8.5286}, # short life engine, low rul
    {"unit_number": 1, "sensor_4": 1406.81, "sensor_11": 47.27, "sensor_15": 8.3951}, # mid life engine, high rul
    {"unit_number": 1, "sensor_4": 1413.73, "sensor_11": 48.07, "sensor_15": 8.4595}, # mid life engine, mid rul
    {"unit_number": 1, "sensor_4": 1422.73, "sensor_11": 47.84, "sensor_15": 8.4853}, # mid life engine, low rul
    {"unit_number": 4, "sensor_4": 1400.32, "sensor_11": 47.39, "sensor_15": 8.4309}, # mid life engine, high rul
    {"unit_number": 4, "sensor_4": 1402.75, "sensor_11": 47.53, "sensor_15": 8.4440}, # mid life engine, mid rul
    {"unit_number": 4, "sensor_4": 1419.52, "sensor_11": 47.95, "sensor_15": 8.4826}, # mid life engine, low rul
    {"unit_number": 5, "sensor_4": 1396.29, "sensor_11": 47.11, "sensor_15": 8.3990}, # long life engine, high rul
    {"unit_number": 5, "sensor_4": 1431.89, "sensor_11": 47.82, "sensor_15": 8.5036}, # long life engine, mid rul
    {"unit_number": 5, "sensor_4": 1425.39, "sensor_11": 48.26, "sensor_15": 8.5297} # long life engine, low rul
]

for row in sample_inputs:
    output = predict_rul(row)
    print(
        f"unit={output['unit_number']} | "
        f"rul_pred={output['rul_pred']: .2f} | "
        f"rul_lower={output['rul_lower']: .2f} | "
        f"risk_band={output['risk_band']} | "
        f"action={output['recommended_action']} | "
        f"flags={output['flags']}"
    )