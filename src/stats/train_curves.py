
import matplotlib.pyplot as plt

if __name__ == '__main__':
    MODEL_LOG_PATH = '../../projects/nuclear/resources/50nm_model.log'

    val_losses = []
    train_losses = []

    model_log = open(MODEL_LOG_PATH)
    for line in model_log.readlines():

        parts = line.split(',')
        epoch = int(parts[0][1:-1])+1
        val_loss = float(parts[1][1:-1])
        train_loss = float(parts[2][1:-1])

        val_losses.append(val_loss)
        train_losses.append(train_loss)

    plt.ylim(0, 1)
    plt.title('50nm per pixel model training curves')
    plt.ylabel('Smoothed Dice coefficient')
    plt.xlabel('Training epoch')
    plt.plot(val_losses, label='Validation loss')
    plt.plot(train_losses, label='Training loss')
    plt.legend(loc='upper right')
    plt.show()
