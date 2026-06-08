import matplotlib.pyplot as plt

def plot_loss(train_loss: list, val_loss: list, savepath: str):
    fig, ax = plt.subplots(1, 1, clear = True, figsize = (8,6), dpi = 360)
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
        min_loss = min(val_loss)
        min_index = val_loss.index(min_loss)
        ax.scatter(min_index, min_loss, marker = "*", s = 100)
        ax.text(min_index, min_loss*1.001, s = f"{min_loss:.4f}")
    ax.spines[["top","right"]].set_color(None)
    ax.grid(axis="x")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss Value")
    ax.legend()
    fig.savefig(savepath)
    plt.close()