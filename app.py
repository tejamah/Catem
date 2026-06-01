from src.data_loader import load_data
from src.catem_scoring import compute_catem_scores


def main():
    print("CATEM Telepresence Evaluation Prototype")
    data = load_data("data/synthetic/catem_sample_data.csv")
    score_df = compute_catem_scores(data)
    print(score_df[[
        "participant_id",
        "session_id",
        "embodiment_score",
        "presence_score",
        "behavior_score",
        "workload_risk_score",
        "system_stability_score",
        "data_quality_score",
        "catem_score",
    ]].head())
    overall_score = score_df["catem_score"].mean()
    print(f"Average literature-weighted CATEM score: {overall_score * 100:.1f}/100")


if __name__ == "__main__":
    main()
