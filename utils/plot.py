import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss, savepath):
    fig, ax = plt.subplots(1, 1, clear = True)
    if train_loss:
        ax.plot(list(range(len(train_loss))), 
                train_loss, color = "red", 
                linestyle = "-.",
                label = "Train Loss")
    if val_loss:
        ax.plot(list(range(len(val_loss))), 
                val_loss, color = "green", 
                linestyle = "--",
                label = "Valid Loss")
    fig.savefig(savepath)
    plt.close()