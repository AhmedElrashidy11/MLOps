import mlflow
import sys

# Set MLflow tracking URI (same as train.py)
mlflow.set_tracking_uri("file:./mlruns")

# Read run_id from file
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Get run from MLflow
run = mlflow.get_run(run_id)

# Get accuracy metric
accuracy = run.data.metrics.get("final_accuracy", 0)

print(f"Run ID: {run_id}")
print(f"Final Accuracy: {accuracy}")

# Threshold check
if accuracy < 0.85:
    print("Accuracy below 0.85 → Deployment FAILED")
    sys.exit(1)
else:
    print("Accuracy ≥ 0.85 → Deployment SUCCESS")
