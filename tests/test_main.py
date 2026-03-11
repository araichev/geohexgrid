import functools
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg

from .context import DATA_DIR, geohexgrid, pytest
from geohexgrid import main as ghg


ROUND_TRIP_DOUBLE_CASES = [
    (0, 0),
    (1, 1),
    (2, 0),
    (-1, -1),
    (-2, 0),
    (3, 1),
]

AXIAL_DOUBLE_CASES = [
    (0, 0),
    (1, 0),
    (-2, 3),
    (4, -1),
]


def _double_to_cell_id(a, b):
    return f"{a},{b}"


def _assign_double_x(frame):
    return frame.assign(y=frame["x"] * 2)


def test_axial_round():
    # near-center behavior
    assert ghg.axial_round(0.1, -0.1) == (0, 0)

    # positive directions
    assert ghg.axial_round(0.9, 0.1) == (1, 0)
    assert ghg.axial_round(0.1, 0.9) == (0, 1)

    # negative directions
    assert ghg.axial_round(-0.8, -0.2) == (-1, 0)
    assert ghg.axial_round(-0.2, -0.8) == (0, -1)


def test_cartesian_to_axial():
    # known point from existing coverage
    R = 3
    p = (1.7 * R, 0.2 * R)
    assert ghg.cartesian_to_axial(*p, R=R) == (1, 0)

    # round-trip at hex centers
    for a, b in AXIAL_DOUBLE_CASES:
        da, db = ghg.axial_to_double(a, b)
        x, y = ghg.double_to_cartesian(da, db, R=1)
        assert ghg.cartesian_to_axial(x, y, R=1) == (a, b)


def test_axial_to_double():
    # basic conversions
    assert ghg.axial_to_double(0, 0) == (0, 0)
    assert ghg.axial_to_double(1, 0) == (1, 1)
    assert ghg.axial_to_double(0, 1) == (0, 2)
    assert ghg.axial_to_double(-2, 3) == (-2, 4)

    # consistency with known round-trip through Cartesian space
    for a, b in AXIAL_DOUBLE_CASES:
        da, db = ghg.axial_to_double(a, b)
        x, y = ghg.double_to_cartesian(da, db, R=2)
        assert ghg.cartesian_to_axial(x, y, R=2) == (a, b)


def test_cartesian_to_double():
    # known point from existing coverage
    R = 3
    p = (1.7 * R, 0.2 * R)
    assert ghg.cartesian_to_double(*p, R) == (1, 1)

    # round-trip at hex centers
    for a, b in ROUND_TRIP_DOUBLE_CASES:
        x, y = ghg.double_to_cartesian(a, b, R=2.5)
        assert ghg.cartesian_to_double(x, y, R=2.5) == (a, b)


def test_double_to_cartesian():
    # basic coordinates
    assert ghg.double_to_cartesian(0, 0, 1) == (0.0, 0.0)
    assert ghg.double_to_cartesian(1, 1, 1) == (1.5, ghg.K)
    assert ghg.double_to_cartesian(0, 2, 1) == (0.0, 2 * ghg.K)

    # consistency with inverse mapping
    for a, b in ROUND_TRIP_DOUBLE_CASES:
        x, y = ghg.double_to_cartesian(a, b, R=1.75)
        assert ghg.cartesian_to_double(x, y, R=1.75) == (a, b)


def test_make_grid_points():
    # shape and spacing
    nrows = 3
    ncols = 5
    R = 3
    x0 = 2
    y0 = 1
    X, Y = ghg.make_grid_points(nrows, ncols, R=R, x0=x0, y0=y0)

    assert X.shape == (2 * nrows + 2, math.ceil((3 * ncols + 2) / 2))
    assert X[0][1] - X[0][0] == R
    assert Y[1][0] - Y[0][0] == ghg.K * R

    # origin shift
    X0, Y0 = ghg.make_grid_points(nrows, ncols, R=R, x0=0, y0=0)
    assert np.allclose(X - X0, x0)
    assert np.allclose(Y - Y0, y0)


def test_make_grid():
    # happy path
    nrows = 2
    ncols = 4
    R = 1
    a0 = -2
    b0 = -2
    x0, y0 = ghg.double_to_cartesian(a0, b0, R)
    grid = ghg.make_grid(nrows, ncols, R=R, x0=x0, y0=y0, a0=a0, b0=b0)

    assert set(grid["cell_id"]) == {
        "-2,-2",
        "-1,-1",
        "0,-2",
        "1,-1",
        "-2,0",
        "-1,1",
        "0,0",
        "1,1",
    }

    # geometry invariants
    assert np.allclose(grid.area, 3 * np.sqrt(3) * R**2 / 2)
    assert grid.union_all().boundary.is_ring

    centroids = grid.geometry.centroid
    for cell_id, centroid in zip(grid["cell_id"], centroids):
        a, b = map(int, cell_id.split(","))
        expected_x, expected_y = ghg.double_to_cartesian(a, b, R)
        assert math.isclose(centroid.x, expected_x, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(centroid.y, expected_y, rel_tol=0, abs_tol=1e-12)

    # validation
    with pytest.raises(ValueError):
        ghg.make_grid(2, 2, a0=0, b0=1)

    with pytest.raises(ValueError):
        ghg.make_grid(2, 2, a0=0.0, b0=0)


def test_make_grid_from_bounds():
    # known cell IDs for edge-sensitive rectangle
    rect = gpd.GeoDataFrame(
        geometry=[sg.Polygon([(2.1, -1), (4.9, -1), (4.9, 1.9), (2.1, 1.9)])]
    )
    grid = ghg.make_grid_from_bounds(*rect.total_bounds, R=1, ox=0, oy=0)

    assert set(grid["cell_id"]) == {
        "1,-3",
        "2,-2",
        "3,-3",
        "1,-1",
        "2,0",
        "3,-1",
        "1,1",
        "2,2",
        "3,1",
        "1,3",
        "2,4",
        "3,3",
    }
    assert grid.union_all().contains(rect.union_all())

    # identical overlapping cells when origins match
    rect1 = gpd.GeoDataFrame(geometry=[sg.Polygon([(-2, 1), (3, 1), (3, 5), (-2, 5)])])
    rect2 = rect1.translate(-1, 1)
    R = 1
    ox = 0
    oy = 0
    grid1 = ghg.make_grid_from_bounds(*rect1.total_bounds, R=R, ox=ox, oy=oy)
    grid2 = ghg.make_grid_from_bounds(*rect2.total_bounds, R=R, ox=ox, oy=oy)
    cell_ids = set(grid1["cell_id"]) & set(grid2["cell_id"])
    g1 = grid1.loc[lambda x: x["cell_id"].isin(cell_ids)].sort_values(
        "cell_id", ignore_index=True
    )
    g2 = grid2.loc[lambda x: x["cell_id"].isin(cell_ids)].sort_values(
        "cell_id", ignore_index=True
    )
    assert g1.geom_equals_exact(g2, tolerance=1e-14).all()

    # branch where ox or oy is None
    rect = gpd.GeoDataFrame(
        geometry=[sg.Polygon([(-0.5, -0.25), (2.4, -0.25), (2.4, 1.6), (-0.5, 1.6)])]
    )
    grid = ghg.make_grid_from_bounds(*rect.total_bounds, R=0.5, ox=None, oy=None)
    assert grid.union_all().contains(rect.union_all())
    assert len(grid) > 0

    # one hexagon just covers rectangle
    R = 1
    r = ghg.K * R
    rect = gpd.GeoDataFrame(
        geometry=[sg.Polygon([(-R / 2, -r), (R / 2, -r), (R / 2, r), (-R / 2, r)])]
    )
    grid = ghg.make_grid_from_bounds(*rect.total_bounds, R=R)
    assert grid.union_all().contains(rect.union_all())

    # same scenario with shifted origin
    grid2 = ghg.make_grid_from_bounds(*rect.total_bounds, R=R, ox=-3 * R / 2, oy=-r)
    assert grid2.union_all() == grid.union_all()
    assert grid2.cell_id.tolist() == ["1,1"]

    # edges not initially covered
    rect = gpd.GeoDataFrame(
        geometry=[
            sg.Polygon(
                [
                    (-0.6 * R, -0.1),
                    (2.2 * R, -0.1),
                    (2.2 * R, 3.2 * r),
                    (-0.6 * R, 3.2 * r),
                ]
            )
        ]
    )
    grid = ghg.make_grid_from_bounds(*rect.total_bounds, R=R)
    assert grid.union_all().contains(rect.union_all())

    # more edge cases
    rect = gpd.GeoDataFrame(geometry=[sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    grids = [
        ghg.make_grid_from_bounds(*rect.total_bounds, R=0.27, ox=0, oy=0),
        ghg.make_grid_from_bounds(*rect.total_bounds, R=0.2, ox=0, oy=0),
        ghg.make_grid_from_bounds(*rect.total_bounds, R=0.5, ox=0, oy=0.1),
        ghg.make_grid_from_bounds(*rect.total_bounds, R=1, ox=0.6, oy=0),
    ]
    for grid in grids:
        assert grid.union_all().contains(rect.union_all())

    # CRS propagation
    grid = ghg.make_grid_from_bounds(0, 0, 1, 1, R=0.2, crs="EPSG:4326")
    assert str(grid.crs).upper() == "EPSG:4326"


def test_mp_apply():
    # direct path
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = ghg.mp_apply(_assign_double_x, df, max_batch_size=10)
    expected = _assign_double_x(df)
    pd.testing.assert_frame_equal(result, expected)

    # batched path
    df = pd.DataFrame({"x": list(range(10))})
    result = ghg.mp_apply(_assign_double_x, df, max_batch_size=3, num_workers=2)
    expected = _assign_double_x(df)
    pd.testing.assert_frame_equal(
        result.sort_values("x").reset_index(drop=True),
        expected.sort_values("x").reset_index(drop=True),
    )


def test_make_grid_from_gdf():
    # intersect trimming on simple shape
    shape = gpd.GeoDataFrame(geometry=[sg.Polygon([(1, -1), (3, 1), (0, 3)])])
    grid = ghg.make_grid_from_gdf(shape, R=1, ox=0, oy=0, trim_mode="intersect")
    assert set(grid.columns) == {"cell_id", "geometry"}
    assert set(grid["cell_id"]) == {
        "1,-1",
        "0,0",
        "1,1",
        "2,0",
        "0,2",
        "1,3",
        "2,2",
        "0,4",
    }

    # no trimming
    grid_none = ghg.make_grid_from_gdf(shape, R=1, trim_mode=None)
    grid_bounds = ghg.make_grid_from_bounds(*shape.total_bounds, R=1, crs=shape.crs)
    assert set(grid_none["cell_id"]) == set(grid_bounds["cell_id"])
    assert grid_none.geom_equals_exact(grid_bounds, tolerance=1e-14).all()

    # covering linework over a range of radii
    g = gpd.GeoDataFrame({"geometry": [sg.LineString([(0, 0), (1, 0), (1, 1), (0, 1)])]})
    for i in range(20):
        R = (i + 1) / 10
        r = ghg.K * R
        grid = ghg.make_grid_from_gdf(g, R=R, ox=-3 * R / 2 - 0.1, oy=-r - 0.1)
        assert grid.union_all().contains(g.union_all())

    # real data, CRS propagation, multiprocessing, and clipping
    shapes = gpd.read_file(DATA_DIR / "shapes.geojson").to_crs("epsg:2193")
    R = 900

    grid1 = ghg.make_grid_from_gdf(shapes, R=R, trim_mode="intersect")
    assert grid1.crs == shapes.crs
    assert grid1.dissolve().contains(shapes.dissolve()).all()

    grid2 = ghg.make_grid_from_gdf(shapes, R=R, trim_mode="intersect", max_batch_size=1)
    assert grid2.shape[0] <= grid1.shape[0]
    assert shapes.area.sum() <= grid2.area.sum() <= grid1.area.sum()

    grid3 = ghg.make_grid_from_gdf(shapes, R=R, trim_mode="clip", max_batch_size=1)
    assert np.allclose(grid3.area.sum(), shapes.area.sum())
