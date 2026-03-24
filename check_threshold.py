import mlflow
import sys

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Run ID: {run_id}")

run = mlflow.get_run(run_id)

# 🔍 Print ALL metrics (DEBUG)
print("All metrics:", run.data.metrics)

accuracy = run.data.metrics.get("final_accuracy")

# 🚨 Handle missing metric
if accuracy is None:
    print("❌ ERROR: final_accuracy not found in MLflow!")
    sys.exit(1)

print(f"Final Accuracy: {accuracy}")

if accuracy < 0.85:
    print("❌ Accuracy below 0.85 → Deployment FAILED")
    sys.exit(1)
else:
    print("✅ Accuracy ≥ 0.85 → Deployment SUCCESS")
