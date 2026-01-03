from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["id", "charges"])
    y = df["charges"]

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "region"]
    binary_features = ["smoker"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("bin", "passthrough", binary_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor
