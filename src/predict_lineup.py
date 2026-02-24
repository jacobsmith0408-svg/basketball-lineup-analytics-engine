import pandas as pd
import joblib

PROFILES = "data/syracuse_player_profiles.csv"
MODEL = "data/lineup_model.joblib"


def predict_lineup(season, players):
    df = pd.read_csv(PROFILES)
    model = joblib.load(MODEL)

    lineup = df[(df["Season"] == season) & (df["Player"].isin(players))]

    w = lineup["MP_adv"]

    features = {
        "TS_lineup": (lineup["TS%"] * w).sum() / w.sum(),
        "USG_lineup": (lineup["USG%"] * w).sum() / w.sum(),
        "AST_lineup": (lineup["AST%"] * w).sum() / w.sum(),
        "TOV_lineup": (lineup["TOV%"] * w).sum() / w.sum(),
        "ORB_lineup": (lineup["ORB%"] * w).sum() / w.sum(),
        "DRB_lineup": (lineup["DRB%"] * w).sum() / w.sum(),
        "STL_lineup": (lineup["STL%"] * w).sum() / w.sum(),
        "BLK_lineup": (lineup["BLK%"] * w).sum() / w.sum(),
        "BPM_lineup": (lineup["BPM"] * w).sum() / w.sum(),
    }

    X = pd.DataFrame([features])

    predicted_net = float(model.predict(X)[0])

    return {
        "Predicted_Net_Rating": round(predicted_net, 2)
    }


if __name__ == "__main__":
    players = [
        "JJ Starling",
        "Chris Bell",
        "Justin Taylor",
        "Benny Williams",
        "Naheem McLeod"
    ]

    print(predict_lineup(2024, players))

