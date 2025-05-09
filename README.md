# AI-Powered Cognitive Load Detector

🧠 Classifies mental load from biometric inputs using ML.

## Features
- Analyzes HRV, Eye Movement, Skin Conductance
- 4-level classification: Low, Medium, High, Extreme
- Trained with RandomForest
- Optimized for fast inference on AWS Lambda

## Run
```bash
python src/train.py
python -c "from src.predict import predict_load; print(predict_load(60, 0.42, 0.91))"
