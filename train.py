import pandas as pd
import numpy as np
import joblib  # Used to save the trained model file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
print("📥 Loading dataset...")
df = pd.read_csv('dataset/students.csv')

# 2. Separate Features (X) and Target Variable (y)
# We drop 'Package' to isolate inputs, and use 'Package' as our target
X = df.drop(columns=['Package'])
y = df['Package']

# 3. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Dataset split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

# 4. Model Initialization & Training
print("⚙️ Training Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model training complete!")

# 5. Make Predictions and Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 --- Model Performance Evaluation ---")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f} LPA")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")

# Check if target metric is met
if r2 > 0.85:
    print("🎉 Success! Your R² score exceeds the 0.85 target blueprint benchmark.")
else:
    print("⚠️ Notice: The model configuration didn't cross the 0.85 R² mark. Double check feature distribution.")

# 6. Save the Trained Model
# Storing it in the model directory so app.py can access it later
joblib.dump(model, 'model/placement_model.pkl')
print("\n💾 Model saved successfully as 'model/placement_model.pkl'!")