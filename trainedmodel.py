import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os

# --------------------
# Load Data
# --------------------
file_path = r"C:\Users\torih\Downloads\realistic_pet_adoption_data.csv"
df = pd.read_csv(file_path)

# --------------------
# Temperament Expansion
# --------------------
for trait in ['Playful', 'Shy', 'Calm', 'Aggressive', 'Friendly', 'Independent', 'Anxious', 'Affectionate']:
    df[trait] = df['Temperament'].str.contains(trait).astype(int)
df.drop(columns='Temperament', inplace=True)

# --------------------
# Target & Features
# --------------------
y = df['AdoptionSpeed']
X = df.drop(columns='AdoptionSpeed')

categorical_cols = ['Breed', 'Size', 'Sterilized', 'Health', 'GoodWithChildren', 'GoodWithOtherPets']
numeric_cols = ['AgeInMonths', 'PhotoCount', 'DescriptionLength']

# --------------------
# Preprocessing Pipeline
# --------------------
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
], remainder='passthrough')  # temperament traits stay

# --------------------
# Train/Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------
# Build Model Pipeline
# --------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# --------------------
# Train the Model
# --------------------
pipeline.fit(X_train, y_train)

# --------------------
# Evaluate
# --------------------
y_pred = pipeline.predict(X_test)
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------
# Save Model
# --------------------
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
joblib.dump(pipeline, os.path.join(downloads_path, "xgb_model_realistic.pkl"))
print("✅ Realistic model saved to Downloads as xgb_model_realistic.pkl")
