import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode


def plot(
    scores, mean_scores, lengths, mean_lengths, time_steps, mean_time_steps
):
    plt.clf()  # Clear the current figure

    manager = plt.get_current_fig_manager()
    manager.resize(1152, 864)
    manager.set_window_title("Snake Game Training Plot")

    # Create subplots
    plt.subplot(3, 1, 1)
    plt.title("Scores", fontsize=14)
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Scores", color="blue")
    plt.plot(mean_scores, label="Mean Scores", color="orange")
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.title("Lengths", fontsize=14)
    plt.xlabel("Number of Games")
    plt.ylabel("Length")
    plt.plot(lengths, label="Lengths", color="red")
    plt.plot(mean_lengths, label="Mean Lengths", color="purple")
    plt.text(len(lengths) - 1, lengths[-1], str(lengths[-1]))
    plt.text(len(mean_lengths) - 1, mean_lengths[-1], f"{mean_lengths[-1]:.2f}")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.title("Time Steps", fontsize=14)
    plt.xlabel("Number of Games")
    plt.ylabel("Time Steps")
    plt.plot(time_steps, label="Time Steps", color="green")
    plt.plot(mean_time_steps, label="Mean Time Steps", color="yellow")
    plt.text(len(time_steps) - 1, time_steps[-1], str(time_steps[-1]))
    plt.text(
        len(mean_time_steps) - 1,
        mean_time_steps[-1],
        f"{mean_time_steps[-1]:.2f}",
    )
    plt.legend()
    plt.grid()

    plt.tight_layout(h_pad=2)  # Adjust layout to prevent overlapping
    plt.show(block=False)
    plt.pause(0.1)
