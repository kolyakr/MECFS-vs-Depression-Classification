import json
import random
from pathlib import Path

GENDERS = ["Male", "Female"]
WORK_STATUS = ["Working", "Partially working", "Not working"]
SOCIAL_ACTIVITY = ["Very low", "Low", "Medium", "High", "Very high"]
EXERCISE_FREQ = ["Never", "Rarely", "Often", "Daily"]
MEDITATION = ["Yes", "No"]

rnd = random.Random(42)

items = []
for i in range(200):
    item = {
        "age": rnd.randint(18, 80),
        "gender": rnd.choice(GENDERS),
        "sleep_quality_index": round(rnd.uniform(1, 10), 1),
        "brain_fog_level": round(rnd.uniform(1, 10), 1),
        "physical_pain_score": round(rnd.uniform(1, 10), 1),
        "stress_level": round(rnd.uniform(1, 10), 1),
        "depression_phq9_score": round(rnd.uniform(0, 27), 1),
        "fatigue_severity_scale_score": round(rnd.uniform(0, 10), 1),
        "pem_duration_hours": round(rnd.uniform(0, 72), 1),
        "hours_of_sleep_per_night": round(rnd.uniform(3, 10), 1),
        "pem_present": rnd.choice([0, 1]),
        "work_status": rnd.choice(WORK_STATUS),
        "social_activity_level": rnd.choice(SOCIAL_ACTIVITY),
        "exercise_frequency": rnd.choice(EXERCISE_FREQ),
        "meditation_or_mindfulness": rnd.choice(MEDITATION),
    }
    items.append(item)

out_path = Path(__file__).resolve().parents[1] / "data" / "test_data.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(items)} records to {out_path}")
