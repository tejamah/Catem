from src.data_loader import load_data
from src.catem_scoring import compute_catem_scores


def main():
    print("CATEM Telepresence Evaluation Prototype")
    data = load_data("data/synthetic/catem_sample_data.csv")
    score_df = compute_catem_scores(data)
    print(score_df.head())
    overall_score = score_df["catem_score"].mean()
    print(f"Average prototype CATEM score: {overall_score:.2f}")


if __name__ == "__main__":
    main()
