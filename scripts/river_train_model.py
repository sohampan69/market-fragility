import os
import pandas as pd
from river import linear_model, preprocessing, metrics, compose, optim
import joblib

# ─────────── Configuration ───────────
DATA_DIR = "data"
MODEL_PATH = "trained_model_v1.joblib"
LOG_PATH = "prediction_log.csv"

# Label mappings
label_mapping = {'BUY': 1, 'SELL': 0, 'HOLD': 2}
reverse_mapping = {v: k for k, v in label_mapping.items()}

# ─────────── Model Setup ───────────
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.SoftmaxRegression(
        optimizer=optim.SGD(lr=0.01),
        l2=0.1
    )
)
metric = metrics.Accuracy()
prediction_log = []

total_rows = 0
correct_preds = 0

print("🔄 Starting training...")

# ─────────── Data Loop ───────────
files = [f for f in os.listdir(DATA_DIR) if f.startswith("features_") and f.endswith(".csv")]

for file in files:
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file))

        required_columns = [
            'volatility', 'log_return', 'RSI', 'SMA', 'EMA',
            'EMA200', 'BB_Middle', 'BB_Std', 'BB_Upper',
            'BB_Lower', 'BB_Width', 'MACD', 'MACD_Signal',
            'Fragile_Zone', 'Low_Vol_Zone', 'fragility_ratio',
            'Signal_Strength'
        ]

        for _, row in df.iterrows():
            if not all(col in row for col in required_columns):
                continue

            features = {
                'volatility': float(row['volatility']),
                'log_return': float(row['log_return']),
                'RSI': float(row['RSI']),
                'SMA': float(row['SMA']),
                'EMA': float(row['EMA']),
                'EMA_200': float(row['EMA200']),
                'BB_MIDDLE': float(row['BB_Middle']),
                'BB_std': float(row['BB_Std']),
                'BB_Upper': float(row['BB_Upper']),
                'BB_Lower': float(row['BB_Lower']),
                'BB_Width': float(row['BB_Width']),
                'MACD': float(row['MACD']),
                'MACD_Signal': float(row['MACD_Signal']),
                'Fragility_zone': float(row['Fragile_Zone']),
                'Low_volatility_zone': float(row['Low_Vol_Zone']),
                'fragility_ratio': float(row['fragility_ratio']),
                'signal_strength': float(row.get('Signal_Strength', 0))
            }

            label = str(row['Label'])
            if label not in label_mapping:
                continue
            encoded_label = label_mapping[label]

            # Predict
            prediction = model.predict_one(features)
            confidence = model.predict_proba_one(features)

            prediction_label = reverse_mapping.get(prediction, "UNKNOWN")
            confidence_score = confidence.get(prediction, 0.0)

            prediction_log.append({
                "Stock": file.replace("features_", "").replace(".csv", ""),
                "Predicted": prediction_label,
                "Actual": label,
                "Correct": prediction == encoded_label,
                "Confidence": round(confidence_score * 100 , 2)
            })

            if prediction == encoded_label:
                correct_preds += 1
            total_rows += 1

            # Train
            model.learn_one(features, encoded_label)

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        continue

# ⏬ Calculate average confidence from the prediction log
confidences = [entry['Confidence'] for entry in prediction_log if 'Confidence' in entry]
if confidences:
    avg_confidence = sum(confidences) / len(confidences)
    print(f"\n🧠 Average Model Confidence: {avg_confidence :.2f}%")
else:
    print("⚠️ No confidence scores recorded.")


# ─────────── Save Results ───────────
if total_rows > 0:
    accuracy = (correct_preds / total_rows) * 100
    print(f"\n✅ Training complete. Accuracy: {accuracy:.2f}% on {total_rows} samples")
    print(f"💾 Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    if prediction_log:
        pd.DataFrame(prediction_log).to_csv(LOG_PATH, index=False)
        print(f"📝 Prediction log saved to {LOG_PATH}")
    else:
        print("⚠️ No predictions were logged. Skipping log save.")
else:
    print("⚠️ No valid samples were processed.")

