import numpy as np
from matplotlib import cm as matplotlib_cm


def matplotlib_scale_as_plotly(name: str, num_entries: int = 255) -> list[tuple[float, str]]:
    """Prepare a matplotlib colormap for use in Plotly.

    Args:
        name: Name of the matplotlib colormap.
        num_entries: Number of entries in the colormap.
            Note that Plotly interpolates between the entries. Defaults to 255.

    Returns:
        A Plotly-compatible colorscale (use with `colorscale=...` in Plotly).
        This consists of a list of tuples, where each tuple contains a float and a string.
        The float is the position in the colorscale (between 0 and 1), and the string is the color
        in the format "rgb(R,G,B)".
    """
    # https://plotly.com/python/v3/matplotlib-colorscales/

    cmap = matplotlib_cm.get_cmap(name)
    h = 1.0 / (num_entries - 1)
    pl_colorscale = []

    for k in range(num_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, f"rgb({C[0]},{C[1]},{C[2]})"])

    return pl_colorscale
