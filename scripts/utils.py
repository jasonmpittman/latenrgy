import json
import os

RESULTS_DIR = "results"

def save_results(model_name, metrics, training_time, energy_consumption):
    """
    Saves the results of model training to a JSON file in the results directory.

    Parameters:
        model_name (str): The name of the model (e.g., 'svm', 'knn').
        metrics (dict): Performance metrics for the model.
        training_time (float): Training time in seconds.
        energy_consumption (float): Energy consumption in kWh.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
    results = {
        "Metrics": metrics,
        "Training Time (s)": training_time,
        "Energy Consumption (kWh)": energy_consumption
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
