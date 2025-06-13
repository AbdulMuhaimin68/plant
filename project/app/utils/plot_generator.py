import matplotlib.pyplot as plt
from project.app.utils.monitor import load_history

def generate_training_plot(output_path='static/plot.png'):
    history = load_history()
    acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Train Acc')
    plt.plot(epochs, val_acc, 'g', label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Train Loss')
    plt.plot(epochs, val_loss, 'orange', label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
