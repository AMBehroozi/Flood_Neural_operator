import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import anuga
import os


def show_nan_locations_in_asc(
    asc_path: str,
    title: str = "NaN / NODATA Locations in ASC File",
    nan_color: str = 'red',
    valid_color: str = 'gray',
    figsize: tuple = (10, 8),
    show_coords: bool = True
):
    """
    Loads ASC file and shows a map of NaN/NODATA cells.
    - Red = NaN / NODATA
    - Gray = valid elevation
    """
    try:
        # ─── Read header ────────────────────────────────────────────────
        header = {}
        with open(asc_path, 'r') as f:
            for _ in range(6):
                line = f.readline().strip()
                if line:
                    key, val = line.split(maxsplit=1)
                    header[key.lower()] = float(val) if '.' in val else int(val)

        ncols   = int(header['ncols'])
        nrows   = int(header['nrows'])
        xll     = header['xllcorner']
        yll     = header['yllcorner']
        cellsize = header['cellsize']
        nodata   = header.get('nodata_value', -9999)

        # ─── Load data ──────────────────────────────────────────────────
        data = np.loadtxt(asc_path, skiprows=6)

        # Mark NODATA as NaN
        data[data == nodata] = np.nan

        # Mask array: True = valid, False = NaN
        valid_mask = ~np.isnan(data)

        # ─── Statistics ─────────────────────────────────────────────────
        total_cells = data.size
        nan_count   = np.sum(np.isnan(data))
        print(f"\nFile: {asc_path}")
        print(f"Grid: {nrows} × {ncols} = {total_cells:,} cells")
        print(f"NaN/NODATA cells: {nan_count:,} ({nan_count/total_cells*100:.2f}%)")
        print(f"Valid cells: {np.sum(valid_mask):,}")

        if nan_count == 0:
            print("→ No NaN / NODATA values found.")
            return

        # ─── Plot mask ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        # Show valid cells in gray, NaN in red
        ax.imshow(~valid_mask, cmap=plt.cm.colors.ListedColormap([nan_color, valid_color]),
                  origin='upper', extent=[xll, xll + ncols*cellsize, yll, yll + nrows*cellsize])

        ax.set_title(f"{title}\nRed = NaN/NODATA ({nan_count:,} cells)")
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.grid(False)

        # Optional: print coordinates of first few NaN cells
        if show_coords:
            nan_rows, nan_cols = np.where(np.isnan(data))
            print("\nFirst 10 NaN cell locations (row, col → map coords):")
            for i in range(min(10, len(nan_rows))):
                r, c = nan_rows[i], nan_cols[i]
                x = xll + c * cellsize + cellsize/2
                y = yll + (nrows - 1 - r) * cellsize + cellsize/2
                print(f"  row {r:4d}, col {c:4d} → x={x:.2f}, y={y:.2f}")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error reading {asc_path}: {e}")




def check_asc_for_nan(
    asc_path: str,
    print_sample: bool = True,
    print_nan_locations: bool = False
):
    """
    Loads an ASC file and checks for NaN values in the elevation grid.
    
    Parameters:
    - asc_path: path to the .asc file
    - print_sample: show a few values from the grid
    - print_nan_locations: print coordinates of first few NaN cells (if any)
    """
    try:
        # ─── Read header (first 6 lines) ────────────────────────────────────────
        header = {}
        with open(asc_path, 'r') as f:
            for _ in range(6):
                line = f.readline().strip()
                if line:
                    key, val = line.split(maxsplit=1)
                    header[key.lower()] = float(val) if '.' in val else int(val)

        required = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize']
        for k in required:
            if k not in header:
                raise ValueError(f"Missing header key: {k}")

        ncols   = int(header['ncols'])
        nrows   = int(header['nrows'])
        cellsize = header['cellsize']
        nodata   = header.get('nodata_value', -9999)

        # ─── Load elevation data ────────────────────────────────────────────────
        data = np.loadtxt(asc_path, skiprows=6)

        # Replace NODATA value with NaN (so we can detect it properly)
        data = np.where(data == nodata, np.nan, data)

        # ─── Statistics ─────────────────────────────────────────────────────────
        total_cells = data.size
        nan_count   = np.sum(np.isnan(data))
        finite_count = np.sum(np.isfinite(data))
        valid_min   = np.nanmin(data) if finite_count > 0 else np.nan
        valid_max   = np.nanmax(data) if finite_count > 0 else np.nan
        valid_mean  = np.nanmean(data) if finite_count > 0 else np.nan

        print(f"\nASC file: {asc_path}")
        print(f"Grid size: {nrows} rows × {ncols} cols = {total_cells:,} cells")
        print(f"Cell size: {cellsize} m")
        print(f"NODATA value in header: {nodata}")
        print(f"NaN count after loading: {nan_count:,} / {total_cells:,} ({nan_count/total_cells*100:.2f}%)")
        print(f"Valid (finite) cells: {finite_count:,}")
        if finite_count > 0:
            print(f"Valid elevation range: {valid_min:.3f} – {valid_max:.3f} m")
            print(f"Valid mean elevation: {valid_mean:.3f} m")

        # ─── Optional: show sample values ───────────────────────────────────────
        if print_sample and total_cells > 0:
            print("\nSample of first few values (top-left corner):")
            print(data[:5, :5])  # first 5×5 block

        # ─── Optional: show locations of NaN cells ──────────────────────────────
        if print_nan_locations and nan_count > 0:
            print("\nFirst few NaN cell locations (row, col):")
            nan_rows, nan_cols = np.where(np.isnan(data))
            for i in range(min(10, len(nan_rows))):
                r, c = nan_rows[i], nan_cols[i]
                x = header['xllcorner'] + c * cellsize
                y = header['yllcorner'] + (nrows - 1 - r) * cellsize  # y decreases downward
                print(f"  row {r}, col {c} → (x={x:.2f}, y={y:.2f})")

        if nan_count == 0:
            print("→ No NaN values detected in the grid.")

    except Exception as e:
        print(f"Error reading {asc_path}: {e}")




def plot_mesh(
    domain,
    dplotter,
    points_csv=None,
    polyline_csv=None,
    radius=100,
    figsize=(10, 10),
    title="ANUGA Mesh with Gauges & Polyline",
    gauge_color='red',
    polyline_color='darkblue',
    mesh_color='gray',
    mesh_alpha=0.5,
    mesh_lw=0.1,
    save_path=None
):
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Plot mesh triangles
    ax.triplot(
        dplotter.triang,
        linewidth=mesh_lw,
        color=mesh_color,
        alpha=mesh_alpha,
        label="Mesh"
    )

    # 2. Collect boundaries by tag
    lines_by_tag = {}

    for (tri_id, face_id), tag in domain.boundary.items():
        nodes = domain.triangles[tri_id]

        if face_id == 0:
            n1, n2 = nodes[1], nodes[2]
        elif face_id == 1:
            n1, n2 = nodes[2], nodes[0]
        else:
            n1, n2 = nodes[0], nodes[1]

        p1 = domain.nodes[n1]
        p2 = domain.nodes[n2]

        if tag not in lines_by_tag:
            lines_by_tag[tag] = {"x": [], "y": []}

        lines_by_tag[tag]["x"].extend([p1[0], p2[0], None])
        lines_by_tag[tag]["y"].extend([p1[1], p2[1], None])

    # 3. Define 40 fixed colors (stable & distinct)
    COLORS_40 = [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple",
        "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan",
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "navy","darkgreen","firebrick","darkgoldenrod","teal",
        "slateblue","indigo","darkcyan","crimson","forestgreen",
        "goldenrod","royalblue","seagreen","orchid","peru",
        "steelblue","tomato","mediumvioletred","darkslategray","olive"
    ]

    tags = sorted(lines_by_tag.keys(), key=str)

    for i, tag in enumerate(tags):
        coords = lines_by_tag[tag]
        color = COLORS_40[i % len(COLORS_40)]  # cycle if >40
        ax.plot(
            coords["x"], coords["y"],
            color=color,
            linewidth=2,
            label=f"BC: {tag}"
        )

    # 4. Plot polyline if provided
    if polyline_csv and os.path.exists(polyline_csv):
        poly_df = pd.read_csv(polyline_csv, header=None, names=["x", "y"])
        poly_df["x"] -= poly_df["x"].min()
        poly_df["y"] -= poly_df["y"].min()
        ax.plot(
            poly_df["x"], poly_df["y"],
            color=polyline_color,
            linewidth=2.5,
            linestyle="--",
            marker="o",
            markersize=4,
            label="Polyline",
            zorder=10
        )

    # 5. Plot gauges if provided
    if points_csv and os.path.exists(points_csv):
        points_df = pd.read_csv(points_csv, header=None, names=["x", "y"])

        for i, row in points_df.iterrows():
            x0, y0 = row["x"], row["y"]

            ax.plot(
                x0, y0,
                marker="o",
                markersize=8,
                color=gauge_color,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=15,
                label="Gauge point" if i == 0 else None
            )

            circle = Circle(
                (x0, y0),
                radius,
                color=gauge_color,
                fill=True,
                alpha=0.25,
                linestyle="--",
                linewidth=1.5,
                zorder=5
            )
            ax.add_patch(circle)

    # Finalize
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    ax.set_title(title)
    ax.set_xlabel("Local Easting (m)")
    ax.set_ylabel("Local Northing (m)")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()




def plot_dem_ascii(
    dem_path,
    polyline_csv=None,
    points_csv=None,
    title="Digital Elevation Model",
    cmap="terrain",
    figsize=(8, 6),
    show=True
):
    """
    Plot an ESRI ASCII DEM (.asc) with optional polyline and points from CSV files.

    Parameters
    ----------
    dem_path : str
        Path to DEM .asc file
    polyline_csv : str or None
        CSV file containing polyline coordinates (x, y) without header
    points_csv : str or None
        CSV file containing point coordinates (x, y) without header
    title : str
        Plot title
    cmap : str
        Matplotlib colormap
    figsize : tuple
        Figure size
    show : bool
        Whether to call plt.show()
    """

    # ---- Read DEM ----
    with open(dem_path, "r") as f:
        header = [next(f) for _ in range(6)]
        data = np.loadtxt(f)

    # ---- Extract header info ----
    header_dict = {}
    for line in header:
        key, value = line.split()
        header_dict[key.lower()] = float(value)

    ncols = int(header_dict["ncols"])
    nrows = int(header_dict["nrows"])
    xll = header_dict.get("xllcorner", 0.0)
    yll = header_dict.get("yllcorner", 0.0)
    cellsize = header_dict["cellsize"]
    nodata = header_dict.get("nodata_value", -9999)

    # ---- Mask NoData ----
    dem = np.ma.masked_where(data == nodata, data)

    # ---- Spatial extent ----
    extent = [
        xll,
        xll + ncols * cellsize,
        yll,
        yll + nrows * cellsize,
    ]

    # ---- Plot DEM ----
    plt.figure(figsize=figsize)
    im = plt.imshow(
        dem,
        cmap=cmap,
        origin="upper",
        extent=extent
    )
    plt.colorbar(im, label="Elevation (m)")

    # ---- Plot polyline ----
    if polyline_csv is not None:
        poly = np.loadtxt(polyline_csv, delimiter=",")
        plt.plot(
            poly[:, 0],
            poly[:, 1],
            color="red",
            linewidth=2,
            label="Polyline"
        )

    # ---- Plot points ----
    if points_csv is not None:
        pts = np.loadtxt(points_csv, delimiter=",")
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            color="blue",
            s=25,
            edgecolor="black",
            label="Points"
        )

    # ---- Final formatting ----
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(False)

    if show:
        plt.show()

    return dem



# ────────────────────────────────────────────────
#  READ ASC FILE (ESRI ASCII Grid)
# ────────────────────────────────────────────────
def read_asc(file_path):
    with open(file_path, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip()
            key, value = line.split()
            header[key.lower()] = float(value) if '.' in value else int(value)

        data = np.loadtxt(f, dtype=float)

        nodata = header.get('nodata_value', -9999)
        data[data == nodata] = np.nan

    return data, header


# ────────────────────────────────────────────────
#  MAIN PLOT FUNCTION
# ────────────────────────────────────────────────
def plot_dem_plotly(
    dem_path,
    polygon_csv=None,
    points_csv=None,
    title="Flood Plain DEM"
):
    # ---- Load DEM ----
    dem, hdr = read_asc(dem_path)

    nrows = hdr['nrows']
    ncols = hdr['ncols']
    xll = hdr['xllcorner']
    yll = hdr['yllcorner']
    cellsize = hdr['cellsize']

    # ---- Coordinate grids ----
    x = np.linspace(xll, xll + ncols * cellsize, ncols + 1)[:-1] + cellsize / 2
    y = np.linspace(yll + nrows * cellsize, yll, nrows + 1)[1:] - cellsize / 2

    fig = go.Figure()

    # ---- DEM heatmap ----
    fig.add_trace(go.Heatmap(
        x=x,
        y=y,
        z=dem,
        colorscale='Earth',
        zmin=np.nanmin(dem),
        zmax=np.nanmax(dem),
        colorbar=dict(
            title='Elevation (m)',
            thickness=20,
            len=0.7
        ),
        hovertemplate=(
            'Easting: %{x:.1f} m<br>'
            'Northing: %{y:.1f} m<br>'
            'Elevation: %{z:.2f} m'
            '<extra></extra>'
        ),
        name='DEM'
    ))

    # ---- Polygon (optional) ----
    if polygon_csv is not None:
        poly = pd.read_csv(polygon_csv, header=None, names=['x', 'y'])

        polygon_x = np.append(poly['x'].values, poly['x'].values[0])
        polygon_y = np.append(poly['y'].values, poly['y'].values[0])

        fig.add_trace(go.Scatter(
            x=polygon_x,
            y=polygon_y,
            mode='lines',
            line=dict(color='blue', width=3),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            name='Finer mesh region',
            hovertemplate=(
                'Finer mesh region<br>'
                'Easting: %{x:.1f} m<br>'
                'Northing: %{y:.1f} m'
                '<extra></extra>'
            )
        ))

    # ---- Points from CSV (optional) ----
    if points_csv is not None:
        pts = pd.read_csv(points_csv, header=None, names=['x', 'y'])

        fig.add_trace(go.Scatter(
            x=pts['x'],
            y=pts['y'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            name='Points of interest',
            hovertemplate=(
                'Point<br>'
                'Easting: %{x:.1f} m<br>'
                'Northing: %{y:.1f} m'
                '<extra></extra>'
            )
        ))

    # ---- Layout ----
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Easting (m)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Northing (m)',
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        width=840,
        height=700,
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=True
    )

    fig.show()
