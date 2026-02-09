import csv
import os
import matplotlib.pyplot as plt


def read_episode_metrics(csv_path):
    rewards = []
    lengths = []
    success = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["episode_reward"]))
            lengths.append(float(row["episode_length"]))
            success.append(int(row["success"]))
    return rewards, lengths, success


def main():
    csv_path = os.path.join("logs", "ainex_reach", "episode_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rewards, lengths, success = read_episode_metrics(csv_path)
    episodes = list(range(1, len(rewards) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(episodes, rewards, label="Episode Reward", color="tab:blue", alpha=0.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(episodes, lengths, label="Episode Length", color="tab:orange", alpha=0.6)
    ax2.set_ylabel("Length", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    success_rate = sum(success) / max(1, len(success))
    ax1.set_title(f"AINex Reach Episode Metrics (Success rate: {success_rate:.2%})")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
