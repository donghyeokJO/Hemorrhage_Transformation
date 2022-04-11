import matplotlib.pyplot as plt


def plot_roc_curve(fper, tper, score, best_threshold, idx, title):
    plt.plot(fper, tper, color="red", label="ROC " + str(score))
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    sens, spec = tper[idx], 1 - fper[idx]
    plt.scatter(
        fper[idx],
        tper[idx],
        marker="X",
        s=100,
        color="green",
        label="Best threshold = %.3f, \nSensitivity = %.3f, \nSpecificity = %.3f"
        % (best_threshold, sens, spec),
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve - %.3f" % title)
    plt.legend(loc="lower right")
    plt.show()
