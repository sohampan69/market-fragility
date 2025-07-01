import os
import pandas as pd
from river import tree, preprocessing, compose, metrics
import joblib

# ───────────── CONFIG ─────────────
DATA_DIR = "data"
MODEL_PATH = "trained_model_full_v1.joblib"
LOG_PATH = "prediction_log.csv"

label_mapping = {'BUY': 1, 'SELL': 0, 'HOLD': 2}
reverse_mapping = {v: k for k, v in label_mapping.items()}

# ───────────── MODEL SETUP ─────────────
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    tree.HoeffdingTreeClassifier()
)

metric = metrics.Accuracy()
prediction_log = []

total_rows = 0
correct_preds = 0

print("🔁 Training started. Go grab some coffee...")

files = [f for f in os.listdir(DATA_DIR) if f.startswith("features_") and f.endswith(".csv")]

for file in files:
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file))

        required_columns = [
            'Volume' ,
            'log_return', 'volatility' , 'RSI', 'SMA', 'EMA',
            'EMA200', 'BB_Middle', 'BB_Std', 'BB_Upper',
            'BB_Lower', 'BB_Width', 'BB_Squeeze', 'MACD', 'MACD_Signal', 'shock' , 'non-fund_vol' ,
            'rolling_non-fund_vol' , 'fragility_ratio', 'Fragile_Zone' , 'Low_Vol_Zone' ,
            'Signal_Strength', 'Final_Score', 'Label'

        ]

        for _, row in df.iterrows():
            if not all(col in row for col in required_columns):
                continue

            features = {
                'Volume': float(row['Volume']),
                'volatility': float(row['volatility']),
                'log_return': float(row['log_return']),
                'RSI': float(row['RSI']),
                'SMA': float(row['SMA']),
                'EMA': float(row['EMA']),
                'EMA200': float(row['EMA200']),
                'BB_Middle': float(row['BB_Middle']),
                'BB_Std': float(row['BB_Std']),
                'BB_Upper': float(row['BB_Upper']),
                'BB_Lower': float(row['BB_Lower']),
                'BB_Width': float(row['BB_Width']),
                'MACD': float(row['MACD']),
                'MACD_Signal': float(row['MACD_Signal']),
                'shock': float(row['shock']),
                'non-fund_vol': float(row['non-fund_vol']),
                'rolling_non-fund_vol': float(row['rolling_non-fund_vol']),
                'Fragile_Zone': float(row['Fragile_Zone']),
                'Low_Vol_Zone': float(row['Low_Vol_Zone']),
                'fragility_ratio': float(row['fragility_ratio']),
                'Signal_Strength': float(row['Signal_Strength']),
                'Final_Score': float(row['Final_Score']),


            }

            label = str(row['Label'])
            if label not in label_mapping:
                continue
            encoded_label = label_mapping[label]

            # Predict
            prediction = model.predict_one(features)
            confidence = model.predict_proba_one(features)
            confidence_score = confidence.get(prediction, 0.0)

            prediction_log.append({
                "Stock": file.replace("features_", "").replace(".csv", ""),
                "Predicted": reverse_mapping.get(prediction, "UNKNOWN"),
                "Actual": label,
                "Correct": prediction == encoded_label,
                "Confidence": round(confidence_score * 100, 2)
            })

            if prediction == encoded_label:
                correct_preds += 1
            total_rows += 1

            # Learn
            model.learn_one(features, encoded_label)

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        continue

# ───────────── METRICS ─────────────
confidences = [entry['Confidence'] for entry in prediction_log if 'Confidence' in entry]
avg_conf = sum(confidences) / len(confidences) if confidences else 0

if total_rows > 0:
    accuracy = (correct_preds / total_rows) * 100
    print(f"\n✅ Accuracy: {accuracy:.2f}% on {total_rows} samples")
    print(f"🧠 Avg Confidence: {avg_conf:.2f}%")
    print(f"💾 Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    if prediction_log:
        pd.DataFrame(prediction_log).to_csv(LOG_PATH, index=False)
        print(f"📝 Log saved to {LOG_PATH}")
else:
    print("⚠️ No valid rows processed.")
