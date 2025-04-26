from pyctr.cosserat_rod import *
from pyctr.cosserat_rod import CurvedCosseratRod
from pyctr.robots import ConcentricTubeContinuumRobot

import pyvista as pv
from typing import Sequence, List


def simple_plot_tubes_pyvista(plotter, ps, indices, ds, name, alpha=1.0, color="silver"):
    for i in range(len(indices) - 1, -1, -1):
        line = pv.Spline(ps[: indices[i]], len(ps[: indices[i]]) * 100)
        outer_tube = line.tube(radius=ds[i], n_sides=50)
        plotter.add_mesh(
            outer_tube,
            color=color,
            smooth_shading=True,
            specular=1.0,
            specular_power=100,
            metallic=True,
            opacity=alpha,
            name=f"{name} tube {i}",
        )
    return plotter

def plot_tubes_pyvista(
    plotter: pv.Plotter,
    ps: Sequence,                 # backbone points, shape (N, 3) or list of 3‑tuples
    indices: List[int],           # last point that still belongs to tube‑i
    ds: Sequence[float],          # outer radius of tube‑i  (same order as indices)
    name: str,
    alpha: float = 1.0,
    n_sides: int = 50,
    res_per_seg: int = 20,        # spline interpolation density
    color: str = "silver"
) -> pv.Plotter:
    """
    Draw concentric tubes on an existing PyVista plotter.
    Assumes `indices` are sorted **innermost → outermost**.

    The outermost tube is added first so that the innermost tube remains
    visible even with transparency < 1.
    """
    tube_actors = []
    for i in range(len(indices) - 1, -1, -1):          # outermost → innermost
        end_idx = indices[i]
        pts = ps[: end_idx + 1]                        # include the end‑point!

        # Skip degenerate or empty selections
        if len(pts) < 2:
            continue

        n_interp = max(2, len(pts) * res_per_seg)
        spline = pv.Spline(pts, n_interp)
        tube   = spline.tube(radius=ds[i], n_sides=n_sides)
        tube_actors.append(plotter.add_mesh(
            tube,
            color=color,
            smooth_shading=True,
            specular=1.0,
            specular_power=100,
            metallic=True,
            opacity=alpha,
            name=f"{name} tube {i}",
        ))

    return plotter, tube_actors



def update_tubes_pyvista(
    tube_actors, 
    plotter,
    ps,                      # backbone points
    indices,                 # last point that belongs to tube‑i
    ds,                      # outer radius of tube‑i
    name: str,
    alpha: float = 1.0,
    n_sides: int = 50,
    res_per_seg: int = 20,   # spline interpolation density
):
    """
    Instead of updating existing actors (which might cause kernel crashes),
    we'll remove and recreate them with each update.
    """
    # First, remove all existing actors
    for actor in tube_actors:
        plotter.remove_actor(actor)
    
    # Create all tubes again from scratch
    new_actors = []
    
    # Process tubes from outermost to innermost (for proper visibility)
    for i in range(len(indices) - 1, -1, -1):
        end_idx = indices[i]
        pts = ps[: end_idx + 1]
        
        if len(pts) < 2:
            # Add a placeholder actor to maintain indexing
            dummy = pv.Sphere(radius=0.001, center=(0,0,0))
            actor = plotter.add_mesh(dummy, opacity=0)
            new_actors.append(actor)
            continue
            
        # Create new geometry
        n_interp = max(2, len(pts) * res_per_seg)
        new_spline = pv.Spline(pts, n_interp)
        new_tube = new_spline.tube(radius=ds[i], n_sides=n_sides)
        
        # Add as a new actor
        actor = plotter.add_mesh(
            new_tube,
            opacity=alpha,
            color=f"#{hash(f'tube_{i}') % 0xFFFFFF:06x}"  # Consistent color based on index
        )
        new_actors.append(actor)
    
    # Return the updated list of actors (in reverse order to match original indexing)
    return plotter, list(reversed(new_actors))

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def create_rod(params):
    return CurvedCosseratRod(params=params)


def create_ctr_wrapper(rods):
    return ConcentricTubeContinuumRobot(rods)


def make_adj(R):
    upper = np.hstack([R.reshape(3, 3), np.zeros((3, 3))])
    lower = np.hstack([np.zeros((3, 3)), R.reshape(3, 3)])
    return np.vstack([upper, lower])


def create_lego_tube(points, orientations, max_points, step_len=0.01):
    ez = np.array([[0, 0, 1]]).T
    tip = points[-1:, :]
    missing_parts = max_points - points.shape[0]
    pc2_add = np.zeros((points.shape[0] + missing_parts, points.shape[1]))
    pc2_add[: points.shape[0], :] = points
    dv = step_len * (orientations[-1].reshape(3, 3) @ ez).flatten()
    for i in range(points.shape[0], max_points):
        pc2_add[i, :] = pc2_add[i - 1, :] + dv
    return pc2_add


def split_array_to_columns(arr):
    return arr[:, 0], arr[:, 1], arr[:, 2]


def split_array_to_columns_2d(arr):
    return arr[:, 2], arr[:, 0]


from typing import List, Tuple

def to_index(values, step=0.01):
    return [round(v/step) for v in values]


def get_plot_len_and_size(
    ctr,
    array_len: int,
    beta: List[float],
) -> Tuple[List[int], List[float]]:
    segments = ctr.get_ordered_segments(False)[1:]
    print(segments)
    max_len = ctr.tubes[0][1].params["L"]
    indices = []
    tube_ds = []
    for i, tube in enumerate(ctr.tubes[::-1]): # reverse tube array to start with most outer tube
        params = tube[1].params
        L = params["L"]
        str_L = params["straight_length"]
        tube_ds.append(params["r_outer"])
        if str_L > (1-beta[i])*L:
            normalized_len = segments[2*i]/max_len
            indices.append(int(array_len*normalized_len))
        else:
            normalized_len = segments[2*i+1]/max_len
            indices.append(int(array_len*normalized_len))
        print(normalized_len)
        print(int(array_len * normalized_len))

    return indices, tube_ds

def old_get_plot_len_and_size(
    ctr,
    array_len: int,
    beta: List[float],
) -> Tuple[List[int], List[float]]:
    """
    Compute discrete backbone indices that mark where each tube ends and
    return the corresponding outer diameters.

    Parameters
    ----------
    ctr : CTR
        Concentric‑tube robot instance exposing
        * ``ctr.tubes`` – iterable whose items look like (name, tube_object) and
          whose tube_object carries ``params["L"]`` and ``params["r_outer"]``.
        * ``ctr.get_ordered_segments()`` – list ``[0, s₁, L₁, s₂, L₂, …]`` where
          every pair (sᵢ, Lᵢ) = (end of straight part, tube tip) belongs to tube i,
          starting with the **innermost** tube (first deployed).
          When a tube is fully covered by its neighbours the tip entry may be
          missing, so three values per tube are also valid.
    array_len : int
        Number of points that discretise the robot backbone (length of your FK/IK
        result array).
    beta : list[float]
        Deployment ratio of each tube (0 = completely retracted, 1 = fully
        deployed) – *same order as ctr.tubes*.

    Returns
    -------
    indices : list[int]
        Integer indices, ascending, whose i‑th value is the **last point that
        still belongs to tube i** in the discretised backbone.
    tube_ds : list[float]
        Outer diameters of the tubes in the same order as *indices*.
    """
    # ──────────────────────────────────────────────────────────────────────────
    # Validate / prepare the geometry information
    segments = ctr.get_ordered_segments()
    if not segments or segments[0] != 0.0:
        raise ValueError("ctr.get_ordered_segments() must start with 0.0")

    segments = segments[1:]                         # drop the leading zero
    n_tubes = len(ctr.tubes)
    if len(beta) != n_tubes:
        raise ValueError("len(beta) must equal the number of tubes")

    # Innermost → outermost for plotting convenience
    tube_lengths = [tube[1].params["L"] for tube in ctr.tubes][::-1]
    tube_ds      = [tube[1].params["r_outer"] for tube in ctr.tubes][::-1]

    backbone_total_len = float(segments[-1])
    if backbone_total_len <= 0:
        raise ValueError("Total backbone length must be positive")

    # ──────────────────────────────────────────────────────────────────────────
    # Convert physical segment ends into discrete indices
    indices: List[int] = []
    for i, (L_i, β_i) in enumerate(zip(tube_lengths, beta)):
        # Retrieve the two markers belonging to this tube
        straight = segments[2 * i]                           # always present
        tip      = segments[2 * i + 1] if 2 * i + 1 < len(segments) else straight

        # Desired physical length outside the base
        desired_len = β_i * L_i

        # Take whichever marker is closer to `desired_len`
        seg_end = min((straight, tip), key=lambda s: abs(s - desired_len))

        # Map physical length to array index (clamped to valid range)
        #idx = max(0, min(array_len - 1,
        #                 round(array_len * seg_end / backbone_total_len)))
        indices.append(seg_end)

    indices = to_index(indices)
    indices[-1] = array_len -1 
    
    return indices, tube_ds


def darken_color(color, factor=0.7):
    """Darkens the given color by a specified factor."""
    from matplotlib.colors import to_rgba

    c = to_rgba(color)
    return (c[0] * factor, c[1] * factor, c[2] * factor, c[3])


colors = ["dimgrey", "grey", "darkgrey", "lightgrey"]


def plot_tubes(ax, ps, indices, ds, alpha=1.0, shadow=True, color=None):
    if color is None:
        for i in range(len(indices) - 1, -1, -1):
            x, y, z = split_array_to_columns(ps[: indices[i]])
            if shadow:
                ax.plot(
                    x,
                    y,
                    np.zeros(x.shape),
                    color=darken_color("lightgray", 0.8),
                    linewidth=ds[i] * 1000,
                    alpha=alpha,
                )
            ax.plot(
                *split_array_to_columns(ps[: indices[i]]),
                color=darken_color(colors[i % 4], 0.6),
                linewidth=ds[i] * 1000,
                alpha=alpha,
            )
            ax.plot(
                *split_array_to_columns(ps[: indices[i]]),
                color=darken_color(colors[i % 4], 0.8),
                linewidth=ds[i] * 500,
                alpha=alpha,
            )
            ax.plot(
                *split_array_to_columns(ps[: indices[i]]),
                color=colors[i % 4],
                linewidth=ds[i] * 250,
                alpha=alpha,
            )
    else:
        for i in range(len(indices) - 1, -1, -1):
            x, y, z = split_array_to_columns(ps[: indices[i]])
            if shadow:
                ax.plot(
                    x,
                    y,
                    np.zeros(x.shape),
                    color=darken_color("lightgray", 0.8),
                    linewidth=ds[i] * 1000,
                    alpha=alpha,
                )
            ax.plot(
                *split_array_to_columns(ps[: indices[i]]),
                color=darken_color(colors[color], 0.6),
                linewidth=ds[i] * 1000,
                alpha=alpha,
            )
            ax.plot(
                *split_array_to_columns(ps[: indices[i]]),
                color=darken_color(colors[color], 0.8),
                linewidth=ds[i] * 500,
                alpha=alpha,
            )
            ax.plot(
                *split_array_to_columns(ps[: indices[i]]),
                color=colors[color],
                linewidth=ds[i] * 250,
                alpha=alpha,
            )

    return ax


def plot_tubes_2d(ax, ps, indices, ds, alpha=1.0, shadow=True, color=None):
    for i in range(len(indices) - 1, -1, -1):
        x, y, z = split_array_to_columns(ps[: indices[i]])
        ax.plot(
            *split_array_to_columns_2d(ps[: indices[i]]),
            color=darken_color(colors[i % 4], 0.6),
            linewidth=ds[i] * 5000,
            alpha=alpha,
        )
        ax.plot(
            *split_array_to_columns_2d(ps[: indices[i]]),
            color=darken_color(colors[i % 4], 0.8),
            linewidth=ds[i] * 2500,
            alpha=alpha,
        )
        ax.plot(
            *split_array_to_columns_2d(ps[: indices[i]]),
            color=colors[i % 4],
            linewidth=ds[i] * 1250,
            alpha=alpha,
        )

    return ax
