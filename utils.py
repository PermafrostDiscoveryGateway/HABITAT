import matplotlib
import matplotlib.pyplot as plt
import glob as glob
import os

matplotlib.style.use('ggplot')


def save_plots(
        train_acc, valid_acc, train_loss, valid_loss,
        acc_PLOT_DIR, loss_PLOT_DIR
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_PLOT_DIR)

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_PLOT_DIR)

def save_plots_kfold(
        train_acc, train_loss, acc_PLOT_DIR, loss_PLOT_DIR
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_PLOT_DIR)

    # Loss plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_PLOT_DIR)    