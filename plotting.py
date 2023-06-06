import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import cartopy
import cartopy.crs as ccrs
import matplotlib.animation as animation
import tqdm


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


def plot_heatmap(B_interps, central_coords, image_extent, all_poi_lats, all_poi_lons, B_param, minute, norm, save_fig=False):
    projection = ccrs.AlbersEqualArea(central_latitude=central_coords[0], central_longitude=central_coords[1])

    fig, ax = plt.subplots(figsize=(10, 6), sharex=True, subplot_kw={"projection": projection})
    ax.set_extent((image_extent[2], image_extent[3], image_extent[0], image_extent[1]))
    fig.subplots_adjust(hspace=0, left=0.075, right=0.85)
    cax = fig.add_axes([0.90, 0.15, 0.04, 0.7])

    ax.gridlines(draw_labels=True)
    ax.coastlines()

    fig.suptitle(f'd$B_{B_param[2]}$/dt, {minute}m', fontsize=20)

    scatter = ax.scatter(all_poi_lons, all_poi_lats, c=B_interps, cmap="coolwarm", transform=ccrs.PlateCarree())

    cb = fig.colorbar(mappable=scatter, cax=cax, norm=norm)
    cb.ax.set_title('nT/min', fontsize=16)
    cb.ax.tick_params(labelsize=14)

    if save_fig:
        plt.savefig("heatmap.png")


def plot_quiver(B_n, B_e, central_coords, image_extent, all_poi_lats, all_poi_lons, minute, save_fig=False):
    projection = ccrs.AlbersEqualArea(central_latitude=central_coords[0], central_longitude=central_coords[1])

    fig, ax = plt.subplots(figsize=(10, 6), sharex=True, subplot_kw={"projection": projection})
    ax.set_extent((image_extent[2], image_extent[3], image_extent[0], image_extent[1]))

    ax.gridlines(draw_labels=True)
    ax.coastlines()

    fig.suptitle(f"d$B$/dt, {minute}m", fontsize=20)

    ax.quiver(np.array(all_poi_lons), np.array(all_poi_lats), B_e, B_n, color="#46A9A0", transform=ccrs.PlateCarree())

    if save_fig:
        plt.savefig("quiver.png")


def animate_quiver(all_B_n, all_B_e, central_coords, image_extent, all_poi_lats, all_poi_lons, fps=8):
    assert len(all_B_n) == len(all_B_e)

    projection = ccrs.AlbersEqualArea(central_latitude=central_coords[0], central_longitude=central_coords[1])
    fig, ax = plt.subplots(figsize=(10, 6), sharex=True, subplot_kw={"projection": projection})
    ax.set_extent((image_extent[2], image_extent[3], image_extent[0], image_extent[1]))
    fig.subplots_adjust(hspace=0, left=0.075, right=0.85)
    cax = fig.add_axes([0.90, 0.15, 0.04, 0.7])
    magnitudes = np.sqrt(np.square(all_B_e) + np.square(all_B_n))
    vmin = np.percentile(np.ravel(magnitudes), 2)
    vmax = np.percentile(np.ravel(magnitudes), 98)
    norm = Normalize(vmin, vmax)
    colormap = cm.coolwarm

    ax.gridlines(draw_labels=True)
    ax.coastlines()
    x, y = np.array(all_poi_lons), np.array(all_poi_lats)
    u, v = all_B_e.iloc[0], all_B_n.iloc[0]
    Q = ax.quiver(x, y, u, v, color=colormap(norm(magnitudes.iloc[0])), transform=ccrs.PlateCarree())

    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=colormap, norm=norm)
    cb.ax.set_title('nT/min', fontsize=16)
    cb.ax.tick_params(labelsize=14)

    def update_quiver(minute, Q, x, y):
        Q.set_UVC(all_B_e.iloc[minute], all_B_n.iloc[minute])
        Q.set_color(colormap(norm(magnitudes.iloc[minute])))
        fig.suptitle(f"d$B$/dt, {minute}m", fontsize=20)

        return Q,

    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, x, y), frames=len(all_B_e),
                                   interval=50, blit=False)
    anim.save('quiver_animation.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
