import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

# -----------------------------
# Load Dataset (replace with your real dataset later)
# -----------------------------
print("🔹 Running ETL pipeline...")

# Example dataset (replace this part with pd.read_csv if you have a CSV)
df, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, random_state=42
)
df = pd.DataFrame(df, columns=[f"feature_{i}" for i in range(20)])
df["credit_history"] = ["<0", "1-2", ">=3", "no_check"] * 250  # fake categorical col
df["target"] = y

print(f"✅ Loaded dataset with shape {df.shape}")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify categorical vs numeric
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    print(f"📊 Numeric cols: {list(numeric_cols)}")
    print(f"🔤 Categorical cols: {list(categorical_cols)}")

    # Pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Build pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_processed = pipeline.fit_transform(X)

    return X_processed, y, preprocessor


# -----------------------------
# Split + Save
# -----------------------------
X_processed, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Save to disk
joblib.dump((X_train, X_test, y_train, y_test), "../data/processed/credit_data.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")

print("✅ ETL completed. Processed data + preprocessor saved!")
