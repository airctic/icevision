__all__ = ["open_img", "show_img", "plot_grid", "plot_grid_preds_actuals"]

from icevision.imports import *


def open_img(fn, gray=False):
    if not os.path.exists(fn):
        raise ValueError(f"File {fn} does not exists")
    color = cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(cv2.imread(str(fn), cv2.IMREAD_UNCHANGED), color)


def show_img(img, ax=None, show: bool = False, **kwargs):
    img = img.squeeze().copy()
    cmap = "gray" if len(img.shape) == 2 else None

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()

    if show:
        plt.show()

    return ax


def plot_grid(fs: List[callable], ncols=1, figsize=None, show=False, **kwargs):
    nrows = math.ceil(len(fs) / ncols)
    figsize = figsize or (12 * ncols, 12 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    axs = np.asarray(axs)
    for f, ax in zip(fs, axs.flatten()):
        f(ax=ax)

    plt.tight_layout()
    if show:
        plt.show()


def plot_grid_preds_actuals(
    actuals, predictions, figsize=None, show=False, annotations=None, **kwargs
):
    fig, axs = plt.subplots(
        nrows=len(actuals),
        ncols=2,
        figsize=figsize or (6, 6 * len(actuals) / 2 / 0.75),
        **kwargs,
    )
    i = 0
    for im, ax in zip(zip(actuals, predictions), axs.reshape(-1, 2)):
        ax[0].imshow(im[0], cmap=None)
        ax[0].set_title("Ground truth")
        ax[1].imshow(im[1], cmap=None)
        ax[1].set_title("Prediction")

        if annotations is None:
            ax[0].set_axis_off()
            ax[1].set_axis_off()
        else:
            ax[0].get_xaxis().set_ticks([])
            ax[0].set_frame_on(False)
            ax[0].get_yaxis().set_visible(False)
            ax[0].set_xlabel(annotations[i][0], ma="left")

            ax[1].get_xaxis().set_ticks([])
            ax[1].set_frame_on(False)
            ax[1].get_yaxis().set_visible(False)
            ax[1].set_xlabel(annotations[i][1], ma="left")

            i += 1

    plt.tight_layout()
    if show:
        plt.show()
    return axs
