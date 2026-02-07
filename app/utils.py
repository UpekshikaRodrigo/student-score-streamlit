from pathlib import Path
import joblib

def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "final_best_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Model file not found!")

    return joblib.load(model_path)
