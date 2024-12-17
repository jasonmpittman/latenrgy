import os
import numpy as np
import shap

def explain_with_precomputed_shap(model, X_data, shap_dir, model_type="Model", dataset="Dataset"):
    """
    Load precomputed SHAP values and optionally log explainability results.
    No plots are generated as they are unnecessary for this experiment.
    """
    # Path to precomputed SHAP values
    shap_values_path = os.path.join(shap_dir, f"shap_values_{model_type.lower()}_{dataset}.npy")
    print(f"Loading precomputed SHAP values from: {shap_values_path}")
    
    # Load SHAP values
    shap_values = np.load(shap_values_path, allow_pickle=True)
    
    # Verify the shapes align
    if shap_values.shape[0] != X_data.shape[0]:
        print("Warning: SHAP values and input data do not align in shape.")
    else:
        print("SHAP values successfully loaded and verified.")

    # Log the SHAP values for optional future analysis
    shap_summary_path = os.path.join(shap_dir, f"shap_values_{model_type.lower()}_{dataset}_log.txt")
    print(f"Saving SHAP values summary to: {shap_summary_path}")
    with open(shap_summary_path, "w") as f:
        f.write(f"--- SHAP Values for {model_type} on {dataset} Dataset ---\n")
        f.write(f"Shape of SHAP values: {shap_values.shape}\n")
        f.write("SHAP computation successful, no plots generated.\n")
