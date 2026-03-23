import random
import sys

# Simulate accuracy (you can replace with MLflow later)
accuracy = random.uniform(0.7, 0.95)

print(f"Model Accuracy: {accuracy}")

if accuracy < 0.85:
    print("Accuracy below threshold. Failing pipeline.")
    sys.exit(1)
else:
    print("Accuracy is good. Proceeding to deployment.")
