from sklearn.feature_extraction.text import TfidfVectorizer
import lime.lime_text
import joblib
import torch
import os

def interpret_with_lime(model, X_data, lime_output_dir, instance_indices, vectorizer_path="models/text_vectorizer.pkl"):
    """
    Interpret the model's decisions using LIME for text classification.

    Args:
        model: The trained model.
        X_data: The text data to interpret.
        lime_output_dir: Directory to save LIME results.
        instance_indices: List of indices to interpret.
        vectorizer_path: Path to the text vectorizer file.
    """
    import os
    import joblib
    from lime.lime_text import LimeTextExplainer

    # Load the vectorizer
    print(f"Loading vectorizer from: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    # Wrap the model's predict_proba for LIME
    def predict_proba_pytorch(model, texts):
        X_vectorized = vectorizer.transform(texts)  # Vectorize the texts
        X_tensor = torch.tensor(X_vectorized.toarray(), dtype=torch.float32)
        return model(X_tensor).detach().numpy()

    predict_proba_wrapped = lambda texts: predict_proba_pytorch(model, texts)

    # Ensure the directory exists
    os.makedirs(lime_output_dir, exist_ok=True)

    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

    for i in instance_indices:
        # Decode byte-like objects to strings
        text_instance = X_data[i].decode() if isinstance(X_data[i], bytes) else str(X_data[i])

        print(f"Generating LIME interpretation for text instance {i}...")
        exp = explainer.explain_instance(
            text_instance,
            predict_proba_wrapped,
            num_features=10,
        )

        # Save the explanation
        output_path = os.path.join(lime_output_dir, f"lime_text_instance_{i}.txt")
        with open(output_path, "w") as f:
            f.write(str(exp.as_list()))
        print(f"LIME interpretation saved to: {output_path}")

def predict_proba_pytorch(model, texts, vectorizer):
    """
    Wrapper function for LIME to call PyTorch model's predict_proba.
    Args:
        model: The PyTorch model.
        texts: A list of raw text inputs.
        vectorizer: The preloaded vectorizer for transforming text to numerical input.
    Returns:
        Probabilities as a NumPy array.
    """
    # Vectorize the raw text inputs
    X_vectorized = vectorizer.transform(texts).toarray()

    # Convert to a PyTorch tensor
    X_tensor = torch.tensor(X_vectorized, dtype=torch.float32)

    # Perform inference and return probabilities
    with torch.no_grad():
        outputs = model(X_tensor)
        return outputs.numpy()
