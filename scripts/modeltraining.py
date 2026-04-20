import pandas as pd
import os
from pycaret.regression import *


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "DataScience_Salaries.csv")
artifacts_dir = os.path.join(project_root, "artifacts")
target = "salary"

df = pd.read_csv(data_path)

reg = setup(
    data=df,
    target=target,
    session_id=42,
    train_size=0.75,
    normalize=True,
    transformation=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9,
    verbose=False
)

best_model = compare_models()
evaluate_model(best_model)

new_data = df.copy().drop(target, axis=1)
predictions = predict_model(best_model, data=new_data)

os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "pipeline")
save_model(best_model, model_path)

predictions_path = os.path.join(artifacts_dir, "predictions.csv")
predictions.to_csv(predictions_path, index=False)

print(f"Model saved to: {model_path}.pkl")
print(f"Predictions saved to: {predictions_path}")