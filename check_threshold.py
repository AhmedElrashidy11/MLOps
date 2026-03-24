import mlflow
import sys

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Run ID: {run_id}")

run = mlflow.get_run(run_id)

metrics = run.data.metrics
print("All metrics:", metrics)

if "final_accuracy" not in metrics:
    print("❌ final_accuracy not found")
    print("Available keys:", list(metrics.keys()))
    sys.exit(1)

accuracy = metrics["final_accuracy"]

print(f"Final Accuracy: {accuracy}")

if accuracy < 0.85:
    print("❌ Accuracy below 0.85 → Deployment FAILED")
    sys.exit(1)

print("✅ Accuracy ≥ 0.85 → Deployment SUCCESS")
