import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_output(pred, x, y, batch_ind=0, figsize=(14, 12)):
    res = y - pred
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    images = [
        (pred[batch_ind, :, :], "Prediction"),
        (y[batch_ind, :, :].numpy(), "Target"),
        (res[batch_ind, :, :].numpy(), "Residual"),
        (x[batch_ind, 0, :, :].numpy(), "Input image"),
    ]

    for (i, j), (img, title) in zip([(0, 0), (0, 1), (1, 0), (1, 1)], images):
        im = ax[i, j].imshow(img, clim=(0, 1))
        ax[i, j].set_title(title)
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

        # Create an axes on the right side of ax. The width of cax is 5% of ax,
        # and the padding between cax and ax is fixed at 0.05 inch.
        divider = make_axes_locatable(ax[i, j])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create colorbar in the appended axes
        cbar = plt.colorbar(im, cax=cax)
        # Adjust colorbar ticks for better readability
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    plt.show()


# def plot_output(pred, x, y, batch_ind=0):
#     res = y - pred
#     fig, ax = plt.subplots(2, 2)
#     ax[0, 0].imshow(pred[batch_ind, :, :])
#     ax[0, 0].set_title("Prediction")
#     ax[0, 1].imshow(y[batch_ind, :, :].numpy())
#     ax[0, 1].set_title("Target")
#     ax[1, 0].imshow(res[batch_ind, :, :].numpy())
#     ax[1, 0].set_title("Residual")
#     ax[1, 1].imshow(x[batch_ind, 0, :, :].numpy())
#     ax[1, 1].set_title("Input image")
#     plt.show()
