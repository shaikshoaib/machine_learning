from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load your dataset
df = pd.read_csv('/Users/ShaikShoaib/Desktop/machine_RF/machine_learning/"Synthetic Machine Data 100k (2).csv"',header= 0)

# Select features and target
features = ['Temperature', 'Pressure', 'Vibration_Level', 'Humidity', 'Power_Consumption']
X = df[features]
y = df['Failure_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
X_test_with_predictions = X_test.copy()
X_test_with_predictions["Actual_Failure_Status"] = y_test.values
X_test_with_predictions["Predicted_Failure_Status"] = y_pred

# Show the first 10 rows of the prediction results
pd.DataFrame(X_test_with_predictions).to_csv('/content/sample_data/predictions.csv')
# pd.DataFrame(y_pred).tocsv('/content/sample_data/predictions.csv', index=False)
X_test_with_predictions.head(10)


