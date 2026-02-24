import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
import joblib

DATA_PATH = "data/syracuse_lineup_training.csv"
MODEL_OUT = "data/lineup_model.joblib"

FEATURES = [
    "TS_lineup",
    "USG_lineup",
    "USG_std_lineup",
    "AST_lineup",
    "TOV_lineup",
    "ORB_lineup",
    "DRB_lineup",
    "STL_lineup",
    "BLK_lineup",
    "BPM_lineup",
    "ORtg_lineup",
    "DRtg_lineup",
    "MIN_total",
    "MIN_avg",
]

TARGET = "Net_lineup"  # created in the training CSV


def main():
    df = pd.read_csv(DATA_PATH)

    # Coerce numeric
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET]

    model = Ridge(alpha=1.0)

    # Quick CV sanity check
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"CV R^2 mean: {scores.mean():.3f} | std: {scores.std():.3f}")

    # Train final model on all data
    model.fit(X, y)
    joblib.dump(model, MODEL_OUT)
    print(f"Saved model -> {MODEL_OUT}")

    # Show coefficients
    coef = pd.Series(model.coef_, index=FEATURES).sort_values(ascending=False)
    print("\nTop positive drivers:")
    print(coef.head(8).to_string())

    print("\nTop negative drivers:")
    print(coef.tail(8).to_string())


if __name__ == "__main__":
    main()

