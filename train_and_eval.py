from final_model_config import *
from model_evaluation import *
from model_train import *

# Train the model
print("Starting model training...")
model_train()
print("Model training complete.")

# Evaluate the trained model
print("Starting model evaluation...")
model_evaluation()
print("Model evaluation complete.")

print("Model training and evaluation complete!")
