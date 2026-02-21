import os

def save_figure(fig, filename, folder="images", dpi=144):
    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, filename)

    fig.savefig(image_path, bbox_inches="tight", dpi=dpi)
    print(f"Figure saved to '{image_path}'.")