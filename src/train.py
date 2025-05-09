from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import load_and_clean

df, label_encoder = load_and_clean("data/biometric_data.csv")

X = df[["HRV", "EyeMovement", "SkinConductance"]]
y = df["LoadLabel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "models/cognitive_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
