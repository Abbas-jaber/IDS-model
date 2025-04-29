import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Use the same paths as in your training script
dataPath = r"C:\Users\Abbas\Desktop\ids-project"
resultPath = r"C:\Users\Abbas\Desktop\ids-project\results\Evaluation"

def load_test_data(fileName):
    """Load and preprocess test data matching training pipeline"""
    dataFile = os.path.join(dataPath, fileName)
    df = pd.read_csv(dataFile)
    df = df.dropna()
    return df

def preprocess_data(df, encoder=None):
    """Replicate preprocessing from training script"""
    # Separate features and labels
    y = df.pop('Label').values
    X = normalize(df.values)  # Same normalization as training
    
    # Handle label encoding
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(y)
    y_encoded = encoder.transform(y)
    y_categorical = to_categorical(y_encoded)
    
    
    return X, y_categorical, encoder

def evaluate_model(model, X_test, y_test):
    """Calculate and return metrics"""
    # Basic evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions for additional metrics
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    
    # Full classification report
    report = classification_report(y_true, y_pred_classes, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

def save_results(results, model_name, test_file_name):
    """Save test results to file"""
    result_dir = os.path.join(resultPath, "test_results")
    os.makedirs(result_dir, exist_ok=True)
    
    result_file = os.path.join(result_dir, f"{model_name}_on_{test_file_name}.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"Model Evaluation Results\n{'='*30}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Dataset: {test_file_name}\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Loss: {results['loss']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1-Score: {results['f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
    
    print(f"Results saved to {result_file}")

def main(model_path, test_file):
    """Main testing workflow"""
    # Load model
    model = load_model(model_path)
    model_name = os.path.basename(model_path)
    
    # Load and preprocess test data
    test_df = load_test_data(test_file)
    X_test, y_test, _ = preprocess_data(test_df)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Print results to console
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # Save results
    save_results(results, model_name, os.path.basename(test_file))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_model.py <path_to_model.h5> <test_file.csv>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
