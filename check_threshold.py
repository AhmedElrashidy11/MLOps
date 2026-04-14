# import mlflow
# import sys
#
# mlflow.set_tracking_uri("file:./mlruns")
#
# with open("model_info.txt", "r") as f:
#     run_id = f.read().strip()
#
# print(f"Run ID: {run_id}")
#
# run = mlflow.get_run(run_id)
#
# metrics = run.data.metrics
# print("All metrics:", metrics)
#
# if "final_accuracy" not in metrics:
#     print("final_accuracy not found")
#     print("Available keys:", list(metrics.keys()))
#     sys.exit(1)
#
# accuracy = metrics["final_accuracy"]
#
# print(f"Final Accuracy: {accuracy}")
#
# if accuracy < 0.85:
#     print("Accuracy below 0.85 → Deployment FAILED")
#     sys.exit(1)
#
# print("Accuracy ≥ 0.85 → Deployment SUCCESS")


# import mlflow
# import sys
#
# mlflow.set_tracking_uri("file:mlruns")
#
# with open("model_info.txt", "r") as f:
#     run_id = f.read().strip()
#
# print(f"Run ID: {run_id}")
#
# run = mlflow.get_run(run_id)
#
# metrics = run.data.metrics
# print("Metrics:", metrics)
#
# # SAFETY CHECK
# if len(metrics) == 0:
#     print("No metrics found in MLflow run")
#     sys.exit(1)
#
# # Get accuracy safely
# if "final_accuracy" in metrics:
#     accuracy = metrics["final_accuracy"]
# elif "accuracy" in metrics:
#     accuracy = metrics["accuracy"]
# else:
#     print("No accuracy metric found")
#     print(metrics)
#     sys.exit(1)
#
# print(f"Final Accuracy: {accuracy}")
#
# if accuracy < 0.85:
#     print("Accuracy < 0.85 → Deployment FAILED")
#     sys.exit(1)
#
# print("Accuracy ≥ 0.85 → Deployment SUCCESS")


import sys

# Read accuracy from file (SAFE for CI/CD)
with open("accuracy.txt", "r") as f:
    accuracy = float(f.read().strip())

print(f"Final Accuracy: {accuracy}")

if accuracy < 0.85:
    print("Accuracy < 0.85 → Deployment FAILED")
    sys.exit(1)

print("Accuracy ≥ 0.85 → Deployment SUCCESS")
