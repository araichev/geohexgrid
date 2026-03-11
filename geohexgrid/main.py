from __future__ import annotations

from typing import Callable, Literal
import functools
import math
import multiprocessing as mp
import numbers
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg
from pyproj import CRS


#: For multiprocessing
NUM_CPUS = mp.cpu_count()

#: Recurring constants.
SQRT3 = np.sqrt(3)
K = SQRT3 / 2  # cosine of 30° which is about 0.866
#: The inradius (minimal radius) r of a hexagon is K times its circumradius
#: (maximal radius) R, that is, r = K * R.


# -----------------------------
# Validation helpers
# -----------------------------
def _validate_positive_real(value, name: str) -> None:
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive real number")


def _validate_positive_int(value, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_trim_mode(trim_mode) -> None:
    allowed = {None, "intersect", "clip"}
    if trim_mode not in allowed:
        raise ValueError("trim_mode must be one of None, 'intersect', or 'clip'")


def _validate_origin_pair(ox, oy) -> None:
    if (ox is None) != (oy is None):
        raise ValueError("ox and oy must either both be None or both be numeric")
    if ox is not None and (not isinstance(ox, numbers.Real) or isinstance(ox, bool)):
        raise ValueError("ox must be a real number or None")
    if oy is not None and (not isinstance(oy, numbers.Real) or isinstance(oy, bool)):
        raise ValueError("oy must be a real number or None")


def _validate_geodataframe(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not isinstance(g, gpd.GeoDataFrame):
        raise TypeError("g must be a GeoDataFrame")
    if g.empty:
        raise ValueError("g must be a non-empty GeoDataFrame")

    try:
        geometry = g.geometry
    except Exception as e:
        raise ValueError("g must have an active geometry column") from e

    if geometry.isna().all():
        raise ValueError("g must contain at least one non-null geometry")

    g = g.loc[geometry.notna()].copy()
    geometry = g.geometry
    if g.empty:
        raise ValueError("g must contain at least one non-null geometry")

    if geometry.is_empty.all():
        raise ValueError("g must contain at least one non-empty geometry")

    g = g.loc[~geometry.is_empty].copy()
    if g.empty:
        raise ValueError("g must contain at least one non-empty geometry")

    return g


def _warn_if_geographic_crs(crs) -> None:
    if crs is None:
        return

    try:
        parsed = CRS.from_user_input(crs)
    except Exception:
        return

    if parsed.is_geographic:
        warnings.warn(
            "The CRS is geographic, so R and all grid dimensions are interpreted "
            "in angular units (usually degrees), not linear units like metres. "
            "For distance-based hexagon sizes, reproject to a projected CRS first.",
            UserWarning,
            stacklevel=2,
        )


# -----------------------------
# Helper functions
# -----------------------------
#: Hexagon grid terminology taken from https://www.redblobgames.com/grids/hexagons
def axial_round(a: float, b: float) -> tuple[int, int]:
    """
    Given floating-point axial coordinates of a point in a hexagon grid,
    return the axial coordinates of the hexagon containing the point.
    Adapted from https://observablehq.com/@jrus/hexround.
    """
    a_round, b_round = round(a), round(b)
    a, b = a - a_round, b - b_round  # remainders
    if abs(a) >= abs(b):
        result = int(a_round + round(a + 0.5 * b)), int(b_round)
    else:
        result = int(a_round), int(b_round + round(b + 0.5 * a))
    return result


def cartesian_to_axial(x: float, y: float, R: float) -> tuple[int, int]:
    """
    Given a flat-top hexagon grid of circumradius ``R`` centered at the origin and
    given Cartesian coordinates ``(x, y)`` of a point in the plane, return the
    axial coordinates of the hexagon containing the point.

    Formula from https://www.redblobgames.com/grids/hexagons/#pixel-to-hex .
    """
    _validate_positive_real(R, "R")
    return axial_round((2 / 3) * x / R, (-x / 3 + (SQRT3 / 3) * y) / R)


def axial_to_double(a: float, b: float) -> tuple[float, float]:
    """
    Given axial coordinates of a hexagon in a flat-top hexagonal grid,
    return its double coordinates.

    Formula from https://www.redblobgames.com/grids/hexagons/#conversions-doubled .
    """
    return a, a + 2 * b


def cartesian_to_double(x: float, y: float, R: float) -> tuple[float, float]:
    """
    Given a flat-top hexagon grid of circumradius ``R`` centered at the origin and
    given Cartesian coordinates ``(x, y)`` of a point in the plane, return the
    double coordinates of the hexagon containing the point.
    """
    _validate_positive_real(R, "R")
    return axial_to_double(*cartesian_to_axial(x, y, R))


def double_to_cartesian(a: float, b: float, R: float) -> tuple[float, float]:
    """
    Given double coordinates of a hexagon in a flat-top hexagonal grid centered
    at the origin with hexagon circumradius ``R``, return the Cartesian
    coordinates of its center.

    Formula from https://www.redblobgames.com/grids/hexagons/#hex-to-pixel-doubled .
    """
    _validate_positive_real(R, "R")
    return 1.5 * R * a, K * R * b


# -----------------------------
# Main functions
# -----------------------------
def make_grid_points(
    nrows: int, ncols: int, R: float = 1, x0: float = 0, y0: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make the vertices and centers of a flat-top hexagon grid of circumradius ``R``
    with ``nrows`` rows, ``ncols`` columns, and bottom-left hexagon centered at
    ``(x0, y0)``.

    A row is a left-to-right northeast-neighbor-southeast-neighbor zig-zag of
    hexagons and the next row is stacked on top of the previous row.
    """
    _validate_positive_int(nrows, "nrows")
    _validate_positive_int(ncols, "ncols")
    _validate_positive_real(R, "R")

    r = K * R
    x, y = np.meshgrid(
        np.linspace(
            0,
            math.ceil(3 * ncols / 2) * R,
            math.ceil((3 * ncols + 2) / 2),
        ),
        np.linspace(0, (2 * nrows + 1) * r, 2 * nrows + 2),
        sparse=False,
        indexing="xy",
    )
    x[1::2, :] -= R / 2
    return x - R / 2 + x0, y - r + y0


def make_grid(
    nrows: int,
    ncols: int,
    R: float = 1,
    x0: float = 0,
    y0: float = 0,
    a0: int = 0,
    b0: int = 0,
) -> gpd.GeoDataFrame:
    r"""
    Make a flat-top hexagon grid of circumradius ``R`` with ``nrows`` rows,
    ``ncols`` columns, and bottom-left hexagon centered at ``(x0, y0)``.

    A row is a left-to-right northeast-neighbor-southeast-neighbor zig-zag of
    hexagons and the next row is stacked on top of the previous row.

    Label the bottom-left hexagon with cell ID ``f'{a0},{b0}'``, where ``a0``
    and ``b0`` are integers with an even sum. Label the remaining hexagons with
    *double coordinates* recursively as follows. Given a hexagon with ID 'a,b',
    label its northern neighbor with ID 'a,b+2', its northeast neighbor with
    ID 'a+1,b+1', and its southeast neighbor with ID 'a+1,b-1'.

    For example, if a0 = b0 = 0, then first two rows of cell IDs are
    '0,0', '1,1', '2,0', '3,1', '4,0', '5,1',...
    '0,2', '1,3', '2,2', '3,3', '4,2', '5,3',...

    NOTES:
    - The area of each hexagon is :math:`\frac{3 \sqrt(3)}{2} R^2`.
    """
    _validate_positive_int(nrows, "nrows")
    _validate_positive_int(ncols, "ncols")
    _validate_positive_real(R, "R")

    if not isinstance(a0, int) or not isinstance(b0, int) or (a0 + b0) % 2 != 0:
        raise ValueError(
            "a0 and b0 must be integers with an even sum to qualify for the first "
            "cell ID of the grid"
        )

    X, Y = make_grid_points(nrows=nrows, ncols=ncols, x0=x0, y0=y0, R=R)
    y = Y[:, 0]

    cell_id = [
        f"{a0 + j},{b0 + 2 * i + j % 2}" for i in range(nrows) for j in range(ncols)
    ]

    """
    Make hexagons, each of which has vertex order:
        v4    v3

    v5             v2

        v0    v1
    """
    geometry = [
        sg.Polygon(
            [
                [X[j % 2][(3 * j + 1) // 2], y[2 * i + j % 2]],  # v0
                [X[j % 2][math.ceil((3 * j + 2) / 2)], y[2 * i + j % 2]],  # v1
                [X[j % 2 + 1][3 * j // 2 + 2], y[2 * i + j % 2 + 1]],  # v2
                [X[j % 2][math.ceil((3 * j + 2) / 2)], y[2 * i + j % 2 + 2]],  # v3
                [X[j % 2][(3 * j + 1) // 2], y[2 * i + j % 2 + 2]],  # v4
                [X[j % 2 + 1][3 * j // 2], y[2 * i + j % 2 + 1]],  # v5
            ]
        )
        for i in range(nrows)
        for j in range(ncols)
    ]
    return gpd.GeoDataFrame({"cell_id": cell_id, "geometry": geometry})


def make_grid_from_bounds(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    R: float,
    ox: float | None = 0,
    oy: float | None = 0,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Return a flat-top hexagon grid of circumradius ``R`` with m rows and n columns,
    where m and n are minimal such that the grid covers the rectangle whose
    coordinate extrema are ``minx``, ``miny``, ``maxx``, ``maxy``.

    Label the hexagons with double-coordinate cell IDs (see :func:`make_grid`)
    relative to a 0,0 hexagon centered at point ``(ox, oy)``. The grid will lie
    in the plane of the given CRS (which defaults to ``None``) and will use its
    distance units.
    """
    _validate_positive_real(R, "R")
    _validate_origin_pair(ox, oy)

    if minx > maxx:
        raise ValueError("minx must be <= maxx")
    if miny > maxy:
        raise ValueError("miny must be <= maxy")

    _warn_if_geographic_crs(crs)

    if ox is None and oy is None:
        # Cover the rectangle with a grid whose bottom-left cell lies at minx, miny.
        # A column of i such cells has covering height >= (2*i - 1)*r,
        # where r = K*R, so
        nrows = math.ceil((maxy - miny) / (2 * K * R) + 1 / 2)
        # A row of j such cells has covering width >= (3*j - 2)*R/2, so
        ncols = math.ceil(2 * (maxx - minx) / (3 * R) + 2 / 3)
        grid = make_grid(nrows=nrows, ncols=ncols, x0=minx, y0=miny, R=R)
    else:
        assert ox is not None and oy is not None

        # First translate the calculation to a hex grid with origin hexagon at (0, 0).
        minx -= ox
        miny -= oy
        maxx -= ox
        maxy -= oy

        # Then cover the rectangle with a hex grid.
        # To that end, start by getting the double coordinates of the hexagons covering
        # the rectangle's southwest and northeast corners.
        a0, b0 = cartesian_to_double(minx, miny, R)
        a1, b1 = cartesian_to_double(maxx, maxy, R)
        ncols = int(a1 - a0 + 1)
        nrows = int((b1 - b0) // 2 + 1)

        # center of southwest hexagon H0
        x0, y0 = double_to_cartesian(a0, b0, R)
        # center of northwest hexagon H1
        x1, y1 = double_to_cartesian(a1, b1, R)

        # Adjust H0, H1, nrows, ncols as necessary to handle four edge cases
        if minx < x0 - R / 2:
            # Grid not covering left edge of rectangle,
            # so shift H0 to its southwest neighbor, add a column,
            # and update H1 center
            x0 -= 3 * R / 2
            y0 -= K * R
            a0 -= 1
            b0 -= 1
            if ncols % 2 == 0:
                y1 -= K * R
            ncols += 1
        if miny < y0 and ncols > 1:
            # Grid not covering bottom edge of rectangle,
            # so shift H0 to its south neighbor and add a row.
            # H1 center remains unchanged after row addition.
            y0 -= 2 * K * R
            b0 -= 2
            nrows += 1
        if maxx > x1 + R / 2:
            # Grid not covering right edge of rectangle,
            # so add a column and update H1 center
            x1 += 3 * R / 2
            if ncols % 2 == 0:
                y1 -= K * R
            else:
                y1 += K * R
            ncols += 1
        if maxy > y1 and ncols > 1:
            # Grid not covering top edge of rectangle,
            # so add a row.
            nrows += 1

        # Translate grid labels to those relative to an origin hexagon at (ox, oy).
        grid = make_grid(
            nrows=nrows,
            ncols=ncols,
            x0=x0 + ox,
            y0=y0 + oy,
            R=R,
            a0=int(a0),
            b0=int(b0),
        )

    if crs is not None:
        grid = grid.set_crs(crs)
    return grid


def mp_apply(
    my_func: Callable,  # Using 'func' conflicts with Pandas pipe() method
    df: pd.DataFrame | gpd.GeoDataFrame,
    max_batch_size=5_000,
    num_workers=NUM_CPUS,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Use the multiprocessing module to apply the given pickleable function to the given
    (Geo)DataFrame by splitting it into batches of size ``max_batch_size`` and
    operating on them with ``num_workers`` parallel processes.
    Return a concatenation of the results of batches.

    If ``df`` has at most ``max_batch_size`` rows, then apply ``my_func`` directly
    without batching and parallel processing.

    This function is especially useful for spatial operations.

    EXAMPLES::

        >>> import functools
        >>> import geopandas as gpd
        >>> # load `sites` and `zones` GeoDataFrames and spatially join the former to the latter
        >>> func = functools.partial(gpd.sjoin, right_df=zones)
        >>> mp_apply(func, sites)

    """
    _validate_positive_int(max_batch_size, "max_batch_size")
    _validate_positive_int(num_workers, "num_workers")

    n = len(df)
    if n <= max_batch_size:
        return my_func(df)

    # Use pandas slicing, not np.array_split, to preserve DataFrame/GeoDataFrame type
    chunks = [df.iloc[i : i + max_batch_size] for i in range(0, n, max_batch_size)]

    with mp.Pool(num_workers) as pool:
        frames = pool.map(my_func, chunks)

    return pd.concat(frames, ignore_index=True)


def make_grid_from_gdf(
    g: gpd.GeoDataFrame,
    R: float,
    ox: float | None = 0,
    oy: float | None = 0,
    trim_mode: None | Literal["intersect", "clip"] = "intersect",
    max_batch_size: int = 5_000,
) -> gpd.GeoDataFrame:
    """
    Return a flat-top hexagon grid of circumradius ``R`` with m rows and n columns,
    where m and n are minimal such that the grid covers the total bounds of the
    given GeoDataFrame ``g``.

    Label the hexagons with double-coordinate cell IDs (see :func:`make_grid`)
    relative to a 0,0 hexagon centered at point ``(ox, oy)``, which does not
    necessarily appear in the grid. The grid will lie in the plane of ``g``'s CRS
    and will use its distance units.

    If ``trim_mode == 'intersect'``, then return only the hexagons that intersect
    ``g``. This performs a spatial join under the hood.

    If ``trim_mode == 'clip'``, then return the grid clipped to ``g``, which may
    contain fragments of hexagons. This performs a spatial clip under the hood.

    The spatial operations above will be sped up by parallel processing (via
    :func:`mp_apply`) when the grid's number of cells is greater than
    ``max_batch_size``.
    """
    _validate_positive_real(R, "R")
    _validate_positive_int(max_batch_size, "max_batch_size")
    _validate_trim_mode(trim_mode)
    _validate_origin_pair(ox, oy)

    g = _validate_geodataframe(g)
    grid = make_grid_from_bounds(*g.total_bounds, R=R, ox=ox, oy=oy, crs=g.crs)

    if trim_mode == "intersect":
        if len(grid) <= max_batch_size:
            h = grid.sjoin(g)
        else:
            func = functools.partial(gpd.sjoin, right_df=g)
            h = mp_apply(func, grid, max_batch_size=max_batch_size)
        grid = h.drop_duplicates(subset=["cell_id"]).filter(["cell_id", "geometry"])
    elif trim_mode == "clip":
        if len(grid) <= max_batch_size:
            grid = grid.clip(g)
        else:
            func = functools.partial(gpd.clip, mask=g)
            grid = mp_apply(func, grid, max_batch_size=max_batch_size)

    return grid
