import matplotlib.pyplot as plt
import numpy as np


def plot_train_hist(model_hist, sec_num, syear, eyear):
    plt.figure()
    plt.plot(model_hist.history["loss"], label="Training loss")
    plt.plot(model_hist.history["val_loss"], label="Validation loss")
    plt.title(f"Training Loss (RMSE) SEC #{sec_num}, {syear}-{eyear}")
    plt.ylabel("Loss (A)")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.savefig(f"plots/training_loss_sec{sec_num}_{syear}-{eyear}.png")


def plot_prediction(predicted, real, sec_num):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(predicted)), predicted, label=f"Predicted")
    plt.plot(np.arange(len(real)), real, label=f"Actual")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(f"SEC Coefficient (A)")
    plt.title(f"Real vs. Predicted coefficient for SEC #{sec_num}")
    plt.savefig(f"plots/CNN-test-sec{sec_num}.png")