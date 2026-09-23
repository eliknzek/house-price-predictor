"""
House price prediction with Linear Regression.

Pipeline:
    1. Load data from 'housing_data.csv'
    2. Preprocessing:
        2.1 Mean imputation for numeric columns and the target (price)
        2.2 Outlier removal with IsolationForest (random_state=42)
        2.3 One-hot encoding of 'waterfront' (drop='first')
        2.4 StandardScaler on the selected numeric features
    3. 80/20 train/test split (random_state=42) and LinearRegression training
    4. Evaluation on the test set (train shown for overfitting comparison):
       R^2, Adjusted R^2, MAE, MSE, RMSE, MAPE
    5. Prediction for a new, realistic house and a sanity check against
       similar houses in the data
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "housing_data.csv"

TARGET = "price"
CATEGORICAL = ["waterfront"]
SCALED_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "view", "condition", "sqft_above", "sqft_basement", "yr_built",
]


VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and normalize column names to lowercase."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    log(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path.name}")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: fill missing values in all numeric columns (incl. target) with the mean."""
    numeric_cols = df.select_dtypes(include="number").columns
    missing_before = int(df[numeric_cols].isna().sum().sum())
    df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])
    log(f"Imputed {missing_before} missing values with column means")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: drop rows flagged as outliers by IsolationForest."""
    iso = IsolationForest(random_state=RANDOM_STATE)
    labels = iso.fit_predict(df.select_dtypes(include="number"))
    cleaned = df[labels == 1].reset_index(drop=True)
    log(f"IsolationForest removed {len(df) - len(cleaned)} outliers "
          f"({len(cleaned)} rows remain)")
    return cleaned


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: one-hot encode 'waterfront', dropping the first category."""
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[CATEGORICAL])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(CATEGORICAL), index=df.index
    )
    log(f"Encoded {CATEGORICAL} -> {list(encoded_df.columns)}")
    return pd.concat([df.drop(columns=CATEGORICAL), encoded_df], axis=1)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Step 4: StandardScaler on the selected numeric features.

    The scaler is fit on the training set only and then applied to the test
    set, so no test-set statistics leak into training.
    """
    scaler = StandardScaler()
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[SCALED_FEATURES] = scaler.fit_transform(X_train[SCALED_FEATURES])
    X_test[SCALED_FEATURES] = scaler.transform(X_test[SCALED_FEATURES])
    return X_train, X_test, scaler


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# A new, realistic house: a typical 3-bedroom family home in fair condition.
NEW_HOUSE = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 7500,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1500,
    "sqft_basement": 300,   # sqft_above + sqft_basement = sqft_living
    "yr_built": 1985,
}


def predict_price(model, scaler, feature_columns, house: dict) -> float:
    """Predict the price of one house given its raw (unscaled) features."""
    new = pd.DataFrame([house])
    new = new.reindex(columns=feature_columns, fill_value=0)
    new[SCALED_FEATURES] = scaler.transform(new[SCALED_FEATURES])
    return float(model.predict(new)[0])


def predict_new_house(model, scaler, feature_columns, df: pd.DataFrame) -> None:
    """Step 5: predict the price of NEW_HOUSE and compare it to similar houses.

    The new observation goes through the same transformations as the training
    data: same encoded columns (waterfront_* = 0 for a non-waterfront house)
    and the scaler that was fit on the training set.
    """
    predicted = predict_price(model, scaler, feature_columns, NEW_HOUSE)

    # Sanity check: houses in the (cleaned) data that resemble the new one.
    similar = df[
        (df["bedrooms"] == NEW_HOUSE["bedrooms"])
        & (df["sqft_living"].between(NEW_HOUSE["sqft_living"] - 200,
                                     NEW_HOUSE["sqft_living"] + 200))
        & (df["yr_built"].between(NEW_HOUSE["yr_built"] - 15,
                                  NEW_HOUSE["yr_built"] + 15))
    ]["price"]

    print("\n" + "=" * 52)
    print("Prediction for a new house")
    print("-" * 52)
    for name, value in NEW_HOUSE.items():
        print(f"  {name:<16}{value:>14}")
    print(f"\n  Predicted price: ${predicted:,.0f}")

    if similar.empty:
        print("  No similar houses in the data for comparison.")
        return
    q1, median, q3 = similar.quantile([0.25, 0.5, 0.75])
    print(f"\n  {len(similar)} similar houses in the data "
          f"(3 bedrooms, 1,600-2,000 sqft, built 1970-2000):")
    print(f"    median price:  ${median:,.0f}")
    print(f"    middle 50%:    ${q1:,.0f} - ${q3:,.0f}")
    verdict = "REASONABLE" if q1 <= predicted <= q3 else "worth a second look"
    print(f"  -> The prediction is {verdict}: "
          f"{(predicted - median) / median:+.1%} vs. the similar-house median.")
    print("=" * 52)


def train_model(verbose: bool = True) -> dict:
    """Run steps 1-4 and return the trained model and everything needed to use it."""
    global VERBOSE
    VERBOSE = verbose
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Preprocessing
    df = impute_missing(df)
    df = remove_outliers(df)
    df = encode_categorical(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 3. Split, scale, train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)
    log(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
        f"Features: {X_train.shape[1]}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    p = X_train.shape[1]
    metrics = {}
    for split, y_true, y_pred in [("Train", y_train, y_train_pred),
                                  ("Test", y_test, y_test_pred)]:
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        metrics[split] = {
            "R^2": r2,
            "Adjusted R^2": adjusted_r2(r2, len(y_true), p),
            "MAE ($)": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE ($)": mse ** 0.5,
            "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        }

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": X_train.columns,
        "df": df,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main() -> None:
    result = train_model()
    model, scaler, df, metrics = (result[k] for k in ("model", "scaler", "df", "metrics"))
    feature_columns = result["feature_columns"]

    print("\n" + "=" * 52)
    print(f"{'Metric':<16}{'Train':>18}{'Test':>18}")
    print("-" * 52)
    for name in metrics["Test"]:
        tr, te = metrics["Train"][name], metrics["Test"][name]
        fmt = ".4f" if "R^2" in name else (".2f" if "%" in name else ",.0f")
        print(f"{name:<16}{tr:>18{fmt}}{te:>18{fmt}}")
    print("=" * 52)

    test = metrics["Test"]
    print("\nModel quality (evaluated on the TEST set):")
    print(f"  - The model explains {test['R^2']:.1%} of the variance in price "
          f"(Adjusted R^2 = {test['Adjusted R^2']:.4f}).")
    print(f"  - Average absolute error: ${test['MAE ($)']:,.0f} "
          f"(~{test['MAPE (%)']:.1f}% of the actual price).")
    print(f"  - RMSE: ${test['RMSE ($)']:,.0f}; RMSE > MAE means some "
          f"predictions have large errors.")
    gap = metrics["Train"]["R^2"] - test["R^2"]
    print(f"  - Train/Test R^2 gap: {gap:.4f} -> "
          f"{'no significant overfitting' if gap < 0.05 else 'possible overfitting'}.")

    print("\nModel coefficients:")
    coefs = pd.Series(model.coef_, index=feature_columns).sort_values(key=abs, ascending=False)
    for name, value in coefs.items():
        print(f"  {name:<16}{value:>14,.2f}")
    print(f"  {'intercept':<16}{model.intercept_:>14,.2f}")

    # 5. Predict a new house
    predict_new_house(model, scaler, feature_columns, df)


if __name__ == "__main__":
    main()
