# -*- coding: utf-8 -*-
"""
多方法 LAS 点云地面滤波对比程序

在现有 CSF 程序基础上扩展 4 种主流/经典路线：
1) CSF  - Cloth Simulation Filtering
2) PMF  - Progressive Morphological Filter
3) SMRF - Simple Morphological Filter
4) PTD  - Progressive TIN Densification（纯 Python 近似实现）

输出：
- 每个方法单独目录：ground / nonground / classified LAS
- 2D surface / dz 图
- 3D points / surface / overlay 图
- 若存在 classification=2 真值，则输出精度评价
- 总体 comparison_summary.csv / json / html

说明：
- 该脚本优先追求“可直接运行 + 与你当前程序输出风格一致”。
- PTD 这里是基于 Axelsson(2000) 思想的纯 Python 版本，不是商业软件 Terrascan 的完全复刻。
- SMRF 这里是依据 Pingel et al.(2013) 核心思想实现的简化版，但流程与参数含义保持一致。
"""

import os
import math
import json
import webbrowser
import warnings

import numpy as np
import pandas as pd
import laspy
import open3d as o3d

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import pearsonr
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# 用户配置区
# =========================================================
INPUT_LAS = r"E:\Homework\激光雷达\Forest.las"
OUTPUT_DIR = r"E:\Homework\激光雷达\GroundFilter_MultiMethod_Output_Forest"

# ---------- 预处理 ----------
DO_OUTLIER_REMOVAL = True
NB_NEIGHBORS = 20
STD_RATIO = 2.0

# ---------- 输出控制 ----------
AUTO_OPEN_HTML = False
PLOTLY_MAX_HEATMAP_DIM = 1800
PLOTLY_SCATTER_MAX_POINTS = 30000
PLOTLY_3D_POINTS_MAX = 50000
PLOTLY_3D_SURFACE_MAX_DIM = 250
PLOTLY_3D_POINTS_SURFACE_POINTS_MAX = 30000

SHOW_FINAL_OPEN3D_WINDOW = True
FINAL_VIS_MAX_POINTS = 200000
FINAL_VIS_POINT_SIZE = 2.0
FINAL_VIS_METHOD = "CSF"   # 可选: CSF / PMF / SMRF / PTD

# ---------- 通用网格 ----------
GRID_RES = 0.5
SURFACE_CLASS_THRESHOLD_DEFAULT = 0.5

# ---------- CSF 参数 ----------
CSF_PARAMS = dict(
    gr=0.5,
    dt=0.35,
    ri=2,
    steep_slope_fit=True,
    class_threshold=0.5,
    hcp=0.3,
    max_iter=1000,
    stop_threshold=0.01,
    initial_height_offset=2.0,
)

# ---------- PMF 参数（Zhang et al., 2003）----------
PMF_PARAMS = dict(
    cell_size=0.5,
    initial_window=1,
    max_window=33,
    slope=0.2,
    initial_distance=0.3,
    max_distance=3.0,
    class_threshold=0.5,
)

# ---------- SMRF 参数（Pingel et al., 2013）----------
SMRF_PARAMS = dict(
    cell_size=0.5,
    slope=0.2,
    scalar=1.25,
    window_max=18,
    elevation_threshold=0.35,
    class_threshold=0.5,
)

# ---------- PTD 参数（Axelsson, 2000 思想实现）----------
PTD_PARAMS = dict(
    seed_cell=8.0,
    max_iterations=8,
    distance_threshold=0.45,
    angle_threshold_deg=12.0,
    max_edge_length=60.0,
    class_threshold=0.5,
)

METHODS_TO_RUN = ["CSF", "PMF", "SMRF", "PTD"]


# =========================================================
# 基础工具
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def maybe_open_html(path):
    if AUTO_OPEN_HTML:
        try:
            webbrowser.open_new_tab("file://" + os.path.abspath(path))
        except Exception as e:
            print(f"[WARN] 自动打开 HTML 失败: {e}", flush=True)


def sample_indices(n, max_points, seed=42):
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def sample_points(xyz, max_points=120000, seed=42):
    if xyz.shape[0] == 0:
        return xyz
    idx = sample_indices(xyz.shape[0], max_points, seed)
    return xyz[idx]


def load_las_xyz(las_path):
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    xyz = np.column_stack((x, y, z))
    return las, xyz


def try_get_reference_ground_mask(las):
    try:
        dims = [d for d in las.point_format.dimension_names]
        if "classification" not in dims:
            return None
        cls = np.asarray(las.classification)
        if cls.size == 0:
            return None
        return cls == 2
    except Exception as e:
        print(f"[WARN] 读取 classification 失败: {e}", flush=True)
        return None


def write_las_subset(src_las, mask, out_path):
    new_las = laspy.create(
        point_format=src_las.header.point_format,
        file_version=src_las.header.version,
    )
    new_las.header.scales = src_las.header.scales
    new_las.header.offsets = src_las.header.offsets

    for dim_name in src_las.point_format.dimension_names:
        try:
            new_las[dim_name] = src_las[dim_name][mask]
        except Exception:
            pass

    new_las.write(out_path)


def write_las_with_classification(src_las, full_classification, out_path):
    out_las = laspy.create(
        point_format=src_las.header.point_format,
        file_version=src_las.header.version,
    )
    out_las.header.scales = src_las.header.scales
    out_las.header.offsets = src_las.header.offsets

    for dim_name in src_las.point_format.dimension_names:
        try:
            out_las[dim_name] = src_las[dim_name]
        except Exception:
            pass

    dims = [d for d in out_las.point_format.dimension_names]
    if "classification" in dims:
        out_las.classification = full_classification.astype(np.uint8)
    else:
        print("[WARN] LAS 不支持 classification 字段。", flush=True)

    out_las.write(out_path)


def open3d_statistical_outlier_removal(xyz, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    ind = np.asarray(ind, dtype=np.int64)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[ind] = True
    return xyz[mask], mask


def fill_nan_with_nearest(arr):
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr.copy()
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def downsample_2d_for_plot(arr, max_dim=1800):
    h, w = arr.shape
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    return arr[::step, ::step], step


def compute_grid_spec(x, y, grid_size):
    xmin, ymin = float(np.min(x)), float(np.min(y))
    xmax, ymax = float(np.max(x)), float(np.max(y))
    cols = int(math.ceil((xmax - xmin) / grid_size)) + 1
    rows = int(math.ceil((ymax - ymin) / grid_size)) + 1
    return xmin, ymin, xmax, ymax, rows, cols


def xyz_to_grid_indices(x, y, xmin, ymin, grid_size, rows, cols):
    col_idx = np.floor((x - xmin) / grid_size).astype(np.int32)
    row_idx = np.floor((y - ymin) / grid_size).astype(np.int32)
    valid = (row_idx >= 0) & (row_idx < rows) & (col_idx >= 0) & (col_idx < cols)
    return row_idx, col_idx, valid


def build_min_surface_grid(x, y, z, grid_size):
    xmin, ymin, _, _, rows, cols = compute_grid_spec(x, y, grid_size)
    rr, cc, valid = xyz_to_grid_indices(x, y, xmin, ymin, grid_size, rows, cols)

    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    flat = rr[valid] * cols + cc[valid]
    order = np.argsort(flat)
    flat_sorted = flat[order]
    z_sorted = z[valid][order]

    start = 0
    n = len(flat_sorted)
    while start < n:
        end = start + 1
        idx0 = flat_sorted[start]
        zmin = z_sorted[start]
        while end < n and flat_sorted[end] == idx0:
            if z_sorted[end] < zmin:
                zmin = z_sorted[end]
            end += 1
        r = idx0 // cols
        c = idx0 % cols
        grid[r, c] = zmin
        start = end

    grid = fill_nan_with_nearest(grid)
    return grid, xmin, ymin, rows, cols


def interpolate_surface_to_points(surface_grid, xmin, ymin, grid_size, x, y):
    rows, cols = surface_grid.shape
    ys = ymin + np.arange(rows) * grid_size
    xs = xmin + np.arange(cols) * grid_size

    interp = RegularGridInterpolator(
        (ys, xs),
        surface_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([y, x])
    z_surface = interp(pts)

    nan_mask = np.isnan(z_surface)
    if np.any(nan_mask):
        interp_nn = RegularGridInterpolator(
            (ys, xs),
            surface_grid,
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )
        z_surface[nan_mask] = interp_nn(pts[nan_mask])

    return z_surface


def surface_result_from_grid(method_name, surface_grid, xmin, ymin, grid_size, xyz, class_threshold=0.5, extra=None):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    z_surface = interpolate_surface_to_points(surface_grid, xmin, ymin, grid_size, x, y)
    dz = z - z_surface
    ground_mask = dz <= class_threshold
    res = {
        "method": method_name,
        "surface_grid": surface_grid,
        "xmin": xmin,
        "ymin": ymin,
        "grid_size": grid_size,
        "z_surface": z_surface,
        "dz": dz,
        "ground_mask": ground_mask,
    }
    if extra:
        res.update(extra)
    return res


# =========================================================
# 评价指标
# =========================================================
def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {"N": int(len(y_true)), "R": np.nan, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}

    r = pearsonr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    return {
        "N": int(len(y_true)),
        "R": float(r),
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias),
    }


def classification_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(np.uint8)
    y_pred = y_pred[mask].astype(np.uint8)
    if len(y_true) == 0:
        return None

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "cm_ground_nonground_order": cm.tolist(),
        "N": int(len(y_true)),
        "OA": float(oa),
        "Kappa": float(kappa),
        "Precision_ground": float(precision),
        "Recall_ground": float(recall),
        "F1_ground": float(f1),
    }


# =========================================================
# Plotly 输出
# =========================================================
def save_plotly_heatmap(arr, out_html, title, colorscale="Earth", zmin=None, zmax=None):
    arr_plot, step = downsample_2d_for_plot(arr, max_dim=PLOTLY_MAX_HEATMAP_DIM)
    fig = go.Figure(data=go.Heatmap(
        z=arr_plot,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="Value"),
    ))
    fig.update_layout(
        title=f"{title} (display step={step})",
        template="plotly_white",
        width=1050,
        height=820,
        xaxis_title="Column",
        yaxis_title="Row",
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_difference_map(arr, out_html, title="Difference Map"):
    valid = np.isfinite(arr)
    if np.any(valid):
        p2 = np.nanpercentile(arr[valid], 2)
        p98 = np.nanpercentile(arr[valid], 98)
        vmax = max(abs(p2), abs(p98))
        vmin = -vmax
    else:
        vmin, vmax = -1, 1
    save_plotly_heatmap(arr, out_html, title=title, colorscale="RdBu", zmin=vmin, zmax=vmax)


def save_plotly_scatter(x, y, out_html, xlabel, ylabel, title):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return
    idx = sample_indices(len(x), PLOTLY_SCATTER_MAX_POINTS, seed=42)
    x = x[idx]
    y = y[idx]
    metrics = regression_metrics(x, y)
    xy_min = float(min(np.nanmin(x), np.nanmin(y)))
    xy_max = float(max(np.nanmax(x), np.nanmax(y)))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="markers",
        marker=dict(size=3, opacity=0.35, color="rgba(30,144,255,0.55)"),
        name="Samples",
    ))
    fig.add_trace(go.Scatter(
        x=[xy_min, xy_max], y=[xy_min, xy_max], mode="lines",
        line=dict(color="red", dash="dash"), name="1:1 line",
    ))
    text_box = (
        f"N={metrics['N']}<br>R={metrics['R']:.4f}<br>R²={metrics['R2']:.4f}<br>"
        f"RMSE={metrics['RMSE']:.4f}<br>MAE={metrics['MAE']:.4f}<br>Bias={metrics['Bias']:.4f}"
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=780,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        annotations=[dict(
            x=0.01, y=0.99, xref="paper", yref="paper", text=text_box,
            showarrow=False, align="left", bgcolor="rgba(255,255,255,0.85)", bordercolor="black"
        )],
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_confusion_matrix(cm, out_html, labels=("Ground", "Non-ground"), title="Confusion Matrix"):
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=list(labels), y=list(labels), colorscale="Blues",
        showscale=True, text=cm, texttemplate="%{text}", textfont={"size": 18}
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=700,
        height=650,
        xaxis_title="Prediction",
        yaxis_title="Reference",
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_3d_points(full_xyz, ground_mask, nonground_mask, out_html, max_points=50000):
    ground_xyz = full_xyz[ground_mask]
    nonground_xyz = full_xyz[nonground_mask]
    ground_vis = sample_points(ground_xyz, max_points // 2, seed=42) if ground_xyz.shape[0] > 0 else np.empty((0, 3))
    nonground_vis = sample_points(nonground_xyz, max_points // 2, seed=43) if nonground_xyz.shape[0] > 0 else np.empty((0, 3))

    fig = go.Figure()
    if ground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=ground_vis[:, 0], y=ground_vis[:, 1], z=ground_vis[:, 2],
            mode="markers", name="Ground",
            marker=dict(size=2, color="rgb(194,148,63)", opacity=0.8)
        ))
    if nonground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=nonground_vis[:, 0], y=nonground_vis[:, 1], z=nonground_vis[:, 2],
            mode="markers", name="Non-ground",
            marker=dict(size=2, color="rgb(51,115,217)", opacity=0.7)
        ))
    fig.update_layout(
        title="3D Point Cloud Classification",
        template="plotly_white",
        width=1100,
        height=850,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_3d_surface(surface_grid, xmin, ymin, grid_size, out_html, max_dim=250):
    surface_plot, step = downsample_2d_for_plot(surface_grid, max_dim=max_dim)
    rows, cols = surface_plot.shape
    xs = xmin + np.arange(cols) * grid_size * step
    ys = ymin + np.arange(rows) * grid_size * step
    fig = go.Figure(data=[go.Surface(
        x=xs, y=ys, z=surface_plot,
        colorscale="Earth", colorbar=dict(title="Elevation")
    )])
    fig.update_layout(
        title=f"3D Surface (display step={step})",
        template="plotly_white",
        width=1100,
        height=850,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_3d_points_surface(full_xyz, ground_mask, nonground_mask,
                                  surface_grid, xmin, ymin, grid_size,
                                  out_html, max_points=30000, max_dim=250):
    ground_xyz = full_xyz[ground_mask]
    nonground_xyz = full_xyz[nonground_mask]
    ground_vis = sample_points(ground_xyz, max_points // 2, seed=42) if ground_xyz.shape[0] > 0 else np.empty((0, 3))
    nonground_vis = sample_points(nonground_xyz, max_points // 2, seed=43) if nonground_xyz.shape[0] > 0 else np.empty((0, 3))

    surface_plot, step = downsample_2d_for_plot(surface_grid, max_dim=max_dim)
    rows, cols = surface_plot.shape
    xs = xmin + np.arange(cols) * grid_size * step
    ys = ymin + np.arange(rows) * grid_size * step

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=surface_plot,
        colorscale="Earth", opacity=0.65, name="Surface",
        showscale=True, colorbar=dict(title="Elevation")
    ))
    if ground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=ground_vis[:, 0], y=ground_vis[:, 1], z=ground_vis[:, 2],
            mode="markers", name="Ground",
            marker=dict(size=2, color="rgb(194,148,63)", opacity=0.85)
        ))
    if nonground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=nonground_vis[:, 0], y=nonground_vis[:, 1], z=nonground_vis[:, 2],
            mode="markers", name="Non-ground",
            marker=dict(size=2, color="rgb(51,115,217)", opacity=0.6)
        ))
    fig.update_layout(
        title="3D Point Cloud + Surface",
        template="plotly_white",
        width=1200,
        height=900,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_comparison_bar(df, out_html):
    if df.empty:
        return
    metrics = [c for c in ["OA", "Kappa", "F1_ground", "Precision_ground", "Recall_ground"] if c in df.columns]
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(x=df["Method"], y=df[m], name=m))
    fig.update_layout(
        title="Ground Filtering Accuracy Comparison",
        template="plotly_white",
        barmode="group",
        width=1100,
        height=700,
        yaxis_title="Metric value",
        xaxis_title="Method",
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


# =========================================================
# Open3D 可视化
# =========================================================
def make_o3d_pcd(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None and xyz.shape[0] > 0:
        colors = np.tile(np.array(color, dtype=np.float64), (xyz.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_final_result(full_xyz, ground_mask, nonground_mask, window_name,
                           max_points=200000, point_size=2.0):
    ground_xyz = full_xyz[ground_mask]
    nonground_xyz = full_xyz[nonground_mask]
    ground_vis = sample_points(ground_xyz, max_points // 2, seed=42) if ground_xyz.shape[0] > 0 else np.empty((0, 3))
    nonground_vis = sample_points(nonground_xyz, max_points // 2, seed=43) if nonground_xyz.shape[0] > 0 else np.empty((0, 3))
    geoms = []
    if ground_vis.shape[0] > 0:
        geoms.append(make_o3d_pcd(ground_vis, color=[0.76, 0.59, 0.25]))
    if nonground_vis.shape[0] > 0:
        geoms.append(make_o3d_pcd(nonground_vis, color=[0.20, 0.45, 0.85]))
    if len(geoms) == 0:
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1600, height=1000, visible=True)
    for g in geoms:
        vis.add_geometry(g)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    render_option.point_size = point_size

    bbox = geoms[0].get_axis_aligned_bounding_box()
    for g in geoms[1:]:
        bbox += g.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.3, -0.3, -0.9])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.7)
    vis.run()
    vis.destroy_window()


# =========================================================
# CSF
# =========================================================
def build_inverted_surface_grid(x, y, z, grid_size):
    zmax = np.max(z)
    z_inv = zmax - z
    xmin, ymin, _, _, rows, cols = compute_grid_spec(x, y, grid_size)
    rr, cc, valid = xyz_to_grid_indices(x, y, xmin, ymin, grid_size, rows, cols)

    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    flat = rr[valid] * cols + cc[valid]
    order = np.argsort(flat)
    flat_sorted = flat[order]
    z_sorted = z_inv[valid][order]

    start = 0
    n = len(flat_sorted)
    while start < n:
        end = start + 1
        idx0 = flat_sorted[start]
        z_top = z_sorted[start]
        while end < n and flat_sorted[end] == idx0:
            if z_sorted[end] < z_top:
                z_top = z_sorted[end]
            end += 1
        r = idx0 // cols
        c = idx0 % cols
        grid[r, c] = z_top
        start = end

    grid = fill_nan_with_nearest(grid)
    return grid, xmin, ymin, rows, cols, zmax


def internal_force_pass(cloth, movable, rigidness):
    rows, cols = cloth.shape
    for _ in range(rigidness):
        delta = np.zeros_like(cloth, dtype=np.float64)
        count = np.zeros_like(cloth, dtype=np.float64)
        for dr, dc in [(0, 1), (1, 0)]:
            r0 = 0
            r1 = rows - dr
            c0 = 0
            c1 = cols - dc

            a = cloth[r0:r1, c0:c1]
            b = cloth[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ma = movable[r0:r1, c0:c1]
            mb = movable[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

            diff = b - a
            half = 0.5 * diff

            both = ma & mb
            delta[r0:r1, c0:c1][both] += half[both]
            delta[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both] -= half[both]
            count[r0:r1, c0:c1][both] += 1
            count[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both] += 1

            only_a = ma & (~mb)
            delta[r0:r1, c0:c1][only_a] += diff[only_a]
            count[r0:r1, c0:c1][only_a] += 1

            only_b = (~ma) & mb
            delta[r0 + dr:r1 + dr, c0 + dc:c1 + dc][only_b] -= diff[only_b]
            count[r0 + dr:r1 + dr, c0 + dc:c1 + dc][only_b] += 1

        valid = (count > 0) & movable
        cloth[valid] += delta[valid] / count[valid]
    return cloth


def csf_postprocess(cloth, movable, ihv, hcp=0.3, max_rounds=50):
    rows, cols = cloth.shape
    for _ in range(max_rounds):
        changed = False
        movable_now = movable.copy()
        to_fix = np.zeros_like(movable, dtype=bool)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r0 = max(0, -dr)
            r1 = min(rows, rows - dr)
            c0 = max(0, -dc)
            c1 = min(cols, cols - dc)
            m_cur = movable_now[r0:r1, c0:c1]
            m_nb = movable_now[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ihv_cur = ihv[r0:r1, c0:c1]
            ihv_nb = ihv[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            cond = m_cur & (~m_nb) & (np.abs(ihv_cur - ihv_nb) <= hcp)
            if np.any(cond):
                to_fix[r0:r1, c0:c1][cond] = True
        if np.any(to_fix):
            cloth[to_fix] = ihv[to_fix]
            movable[to_fix] = False
            changed = True
        if not changed:
            break
    return cloth, movable


def run_csf(xyz, params):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ihv, xmin, ymin, rows, cols, zmax = build_inverted_surface_grid(x, y, z, params["gr"])
    cloth_init = np.nanmax(ihv) + params["initial_height_offset"]
    cloth = np.full((rows, cols), cloth_init, dtype=np.float64)
    movable = np.ones((rows, cols), dtype=bool)
    prev_cloth = cloth.copy()
    g_step = params["dt"] ** 2

    final_iter = 0
    final_max_hv = np.nan
    for it in range(params["max_iter"]):
        cloth[movable] -= g_step
        collide = movable & (cloth <= ihv)
        cloth[collide] = ihv[collide]
        movable[collide] = False
        cloth = internal_force_pass(cloth, movable, params["ri"])
        collide2 = movable & (cloth <= ihv)
        cloth[collide2] = ihv[collide2]
        movable[collide2] = False
        max_hv = np.max(np.abs(cloth - prev_cloth))
        prev_cloth[:] = cloth
        final_iter = it + 1
        final_max_hv = float(max_hv)
        if max_hv < params["stop_threshold"]:
            break

    if params["steep_slope_fit"]:
        cloth, movable = csf_postprocess(cloth, movable, ihv, hcp=params["hcp"], max_rounds=50)

    terrain_grid = zmax - cloth
    return surface_result_from_grid(
        "CSF", terrain_grid, xmin, ymin, params["gr"], xyz,
        class_threshold=params["class_threshold"],
        extra={"iterations": final_iter, "final_max_hv": final_max_hv},
    )


# =========================================================
# PMF
# =========================================================
def progressive_window_sequence(initial_window, max_window):
    wins = []
    w = int(initial_window)
    if w % 2 == 0:
        w += 1
    while w <= max_window:
        wins.append(w)
        w = w * 2 + 1
    if wins[-1] != max_window:
        if max_window % 2 == 0:
            max_window += 1
        if wins[-1] != max_window:
            wins.append(max_window)
    return wins


def pmf_threshold_for_window(win, cell_size, slope, initial_distance, max_distance):
    dh = slope * (win * cell_size) + initial_distance
    return min(dh, max_distance)


def run_pmf(xyz, params):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    grid, xmin, ymin, _, _ = build_min_surface_grid(x, y, z, params["cell_size"])
    ground_grid = grid.copy()

    wins = progressive_window_sequence(params["initial_window"], params["max_window"])
    for win in wins:
        opened = ndimage.grey_opening(ground_grid, size=(win, win))
        diff = ground_grid - opened
        th = pmf_threshold_for_window(
            win, params["cell_size"], params["slope"],
            params["initial_distance"], params["max_distance"]
        )
        mask_ng = diff > th
        ground_grid = ground_grid.copy()
        ground_grid[mask_ng] = opened[mask_ng]
        ground_grid = fill_nan_with_nearest(ground_grid)

    return surface_result_from_grid(
        "PMF", ground_grid, xmin, ymin, params["cell_size"], xyz,
        class_threshold=params["class_threshold"],
        extra={"windows": wins},
    )


# =========================================================
# SMRF
# =========================================================
def run_smrf(xyz, params):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cell_size = params["cell_size"]
    min_grid, xmin, ymin, _, _ = build_min_surface_grid(x, y, z, cell_size)
    elev_grid = min_grid.copy()
    base = fill_nan_with_nearest(elev_grid)

    surface = base.copy()
    windows = list(range(1, params["window_max"] + 1, 2))
    for win in windows:
        opened = ndimage.grey_opening(base, size=(win, win))
        residual = base - opened
        threshold = params["elevation_threshold"] + params["scalar"] * params["slope"] * (win * cell_size)
        threshold = np.maximum(threshold, params["elevation_threshold"])
        mask_obj = residual > threshold
        candidate = base.copy()
        candidate[mask_obj] = opened[mask_obj]
        surface = np.minimum(surface, candidate)

    surface = ndimage.median_filter(surface, size=3)
    surface = fill_nan_with_nearest(surface)

    return surface_result_from_grid(
        "SMRF", surface, xmin, ymin, cell_size, xyz,
        class_threshold=params["class_threshold"],
        extra={"windows": windows},
    )


# =========================================================
# PTD
# =========================================================
def build_seed_points_lowest(xyz, cell_size):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xmin, ymin, _, _, rows, cols = compute_grid_spec(x, y, cell_size)
    rr, cc, valid = xyz_to_grid_indices(x, y, xmin, ymin, cell_size, rows, cols)
    flat = rr[valid] * cols + cc[valid]
    idx_valid = np.where(valid)[0]
    order = np.argsort(flat)
    flat_sorted = flat[order]
    idx_sorted = idx_valid[order]
    z_sorted = z[idx_sorted]

    seed_indices = []
    start = 0
    n = len(flat_sorted)
    while start < n:
        end = start + 1
        idx0 = flat_sorted[start]
        best_idx = idx_sorted[start]
        best_z = z_sorted[start]
        while end < n and flat_sorted[end] == idx0:
            if z_sorted[end] < best_z:
                best_z = z_sorted[end]
                best_idx = idx_sorted[end]
            end += 1
        seed_indices.append(best_idx)
        start = end
    return np.array(seed_indices, dtype=np.int64)


def interpolate_tin_surface(seed_xyz, query_xy):
    if seed_xyz.shape[0] < 3:
        raise ValueError("TIN 种子点不足，至少需要 3 个点。")

    tri = Delaunay(seed_xyz[:, :2])
    simplices = tri.find_simplex(query_xy)
    surface_z = np.full(query_xy.shape[0], np.nan, dtype=np.float64)

    inside = simplices >= 0
    if np.any(inside):
        X = tri.transform[simplices[inside], :2]
        Y = query_xy[inside] - tri.transform[simplices[inside], 2]
        bary = np.einsum('ijk,ik->ij', X, Y)
        bary = np.c_[bary, 1 - bary.sum(axis=1)]
        verts = tri.simplices[simplices[inside]]
        zverts = seed_xyz[verts, 2]
        surface_z[inside] = np.sum(zverts * bary, axis=1)

    outside = ~inside
    if np.any(outside):
        tree = cKDTree(seed_xyz[:, :2])
        _, nn = tree.query(query_xy[outside], k=1)
        surface_z[outside] = seed_xyz[nn, 2]

    return surface_z


def run_ptd(xyz, params):
    n = xyz.shape[0]
    seed_idx = build_seed_points_lowest(xyz, params["seed_cell"])
    ground_mask = np.zeros(n, dtype=bool)
    ground_mask[seed_idx] = True

    max_angle = np.deg2rad(params["angle_threshold_deg"])

    for _ in range(params["max_iterations"]):
        seed_xyz = xyz[ground_mask]
        if seed_xyz.shape[0] < 3:
            break

        q_idx = np.where(~ground_mask)[0]
        if len(q_idx) == 0:
            break

        query_xy = xyz[q_idx, :2]
        z_tin = interpolate_tin_surface(seed_xyz, query_xy)
        dz = xyz[q_idx, 2] - z_tin

        tree = cKDTree(seed_xyz[:, :2])
        dist_xy, _ = tree.query(query_xy, k=1)
        dist_xy = np.maximum(dist_xy, 1e-6)
        angle = np.arctan(np.maximum(dz, 0.0) / dist_xy)

        accept = (
            (dz <= params["distance_threshold"]) &
            (angle <= max_angle) &
            (dist_xy <= params["max_edge_length"])
        )

        if not np.any(accept):
            break

        new_idx = q_idx[accept]
        if np.all(ground_mask[new_idx]):
            break
        ground_mask[new_idx] = True

    seed_xyz = xyz[ground_mask]
    z_surface = interpolate_tin_surface(seed_xyz, xyz[:, :2])
    dz = xyz[:, 2] - z_surface

    xmin, ymin, _, _, rows, cols = compute_grid_spec(xyz[:, 0], xyz[:, 1], GRID_RES)
    rr, cc, valid = xyz_to_grid_indices(xyz[:, 0], xyz[:, 1], xmin, ymin, GRID_RES, rows, cols)
    surface_grid = np.full((rows, cols), np.nan, dtype=np.float64)
    sum_grid = np.zeros((rows, cols), dtype=np.float64)
    cnt_grid = np.zeros((rows, cols), dtype=np.int32)
    np.add.at(sum_grid, (rr[valid], cc[valid]), z_surface[valid])
    np.add.at(cnt_grid, (rr[valid], cc[valid]), 1)
    m = cnt_grid > 0
    surface_grid[m] = sum_grid[m] / cnt_grid[m]
    surface_grid = fill_nan_with_nearest(surface_grid)

    return {
        "method": "PTD",
        "surface_grid": surface_grid,
        "xmin": xmin,
        "ymin": ymin,
        "grid_size": GRID_RES,
        "z_surface": z_surface,
        "dz": dz,
        "ground_mask": dz <= params["class_threshold"],
        "seed_points": int(seed_xyz.shape[0]),
    }


# =========================================================
# 统一导出
# =========================================================
def export_method_outputs(src_las, full_xyz, keep_mask, result_dict, out_dir, method_name):
    method_dir = os.path.join(out_dir, method_name)
    ensure_dir(method_dir)

    filtered_indices = np.where(keep_mask)[0]
    ground_mask_filtered = result_dict["ground_mask"]
    nonground_mask_filtered = ~ground_mask_filtered

    full_ground_mask = np.zeros(full_xyz.shape[0], dtype=bool)
    full_nonground_mask = np.zeros(full_xyz.shape[0], dtype=bool)
    full_ground_mask[filtered_indices[ground_mask_filtered]] = True
    full_nonground_mask[filtered_indices[nonground_mask_filtered]] = True

    full_classification = np.ones(full_xyz.shape[0], dtype=np.uint8)
    full_classification[full_ground_mask] = 2
    full_classification[full_nonground_mask] = 1

    prefix = method_name.lower()
    write_las_with_classification(src_las, full_classification, os.path.join(method_dir, f"{prefix}_classified.las"))
    write_las_subset(src_las, full_ground_mask, os.path.join(method_dir, f"{prefix}_ground.las"))
    write_las_subset(src_las, full_nonground_mask, os.path.join(method_dir, f"{prefix}_nonground.las"))

    save_plotly_heatmap(
        result_dict["surface_grid"],
        os.path.join(method_dir, f"{prefix}_surface.html"),
        title=f"{method_name} Surface",
        colorscale="Earth",
    )

    dz = result_dict["dz"]
    surface_grid = result_dict["surface_grid"]
    dz_grid = np.full_like(surface_grid, np.nan, dtype=np.float64)
    x = full_xyz[keep_mask][:, 0]
    y = full_xyz[keep_mask][:, 1]
    xmin = result_dict["xmin"]
    ymin = result_dict["ymin"]
    grid_size = result_dict["grid_size"]
    rows, cols = dz_grid.shape
    rr, cc, valid = xyz_to_grid_indices(x, y, xmin, ymin, grid_size, rows, cols)

    sum_grid = np.zeros((rows, cols), dtype=np.float64)
    cnt_grid = np.zeros((rows, cols), dtype=np.int32)
    np.add.at(sum_grid, (rr[valid], cc[valid]), dz[valid])
    np.add.at(cnt_grid, (rr[valid], cc[valid]), 1)
    mask = cnt_grid > 0
    dz_grid[mask] = sum_grid[mask] / cnt_grid[mask]

    save_plotly_difference_map(
        dz_grid,
        os.path.join(method_dir, f"{prefix}_dz.html"),
        title=f"Point minus {method_name} Surface (dz)",
    )

    save_plotly_scatter(
        result_dict["z_surface"],
        full_xyz[keep_mask][:, 2],
        os.path.join(method_dir, f"{prefix}_surface_vs_points_scatter.html"),
        xlabel=f"{method_name} Surface Elevation (m)",
        ylabel="Point Elevation (m)",
        title=f"{method_name} Surface vs Original Point Elevation",
    )

    save_plotly_3d_points(
        full_xyz,
        full_ground_mask,
        full_nonground_mask,
        os.path.join(method_dir, f"{prefix}_3d_points.html"),
        max_points=PLOTLY_3D_POINTS_MAX,
    )
    save_plotly_3d_surface(
        result_dict["surface_grid"],
        result_dict["xmin"],
        result_dict["ymin"],
        result_dict["grid_size"],
        os.path.join(method_dir, f"{prefix}_3d_surface.html"),
        max_dim=PLOTLY_3D_SURFACE_MAX_DIM,
    )
    save_plotly_3d_points_surface(
        full_xyz,
        full_ground_mask,
        full_nonground_mask,
        result_dict["surface_grid"],
        result_dict["xmin"],
        result_dict["ymin"],
        result_dict["grid_size"],
        os.path.join(method_dir, f"{prefix}_3d_points_surface.html"),
        max_points=PLOTLY_3D_POINTS_SURFACE_POINTS_MAX,
        max_dim=PLOTLY_3D_SURFACE_MAX_DIM,
    )

    return {
        "method_dir": method_dir,
        "full_ground_mask": full_ground_mask,
        "full_nonground_mask": full_nonground_mask,
        "full_classification": full_classification,
    }


# =========================================================
# 主程序
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    print("=" * 100, flush=True)
    print("多方法 LAS 点云地面滤波对比程序（CSF / PMF / SMRF / PTD）", flush=True)
    print("=" * 100, flush=True)
    print(f"输入文件: {INPUT_LAS}", flush=True)
    print(f"输出目录: {OUTPUT_DIR}", flush=True)

    las, xyz = load_las_xyz(INPUT_LAS)
    print(f"[1/7] 读取完成，点数: {xyz.shape[0]:,}", flush=True)

    if DO_OUTLIER_REMOVAL:
        print("[2/7] 开始统计离群点去噪...", flush=True)
        xyz_filtered, keep_mask = open3d_statistical_outlier_removal(
            xyz, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO
        )
        print(f"       去噪完成，保留: {xyz_filtered.shape[0]:,}，移除: {xyz.shape[0] - xyz_filtered.shape[0]:,}", flush=True)
    else:
        xyz_filtered = xyz.copy()
        keep_mask = np.ones(xyz.shape[0], dtype=bool)
        print("[2/7] 跳过去噪", flush=True)

    ref_ground_mask = try_get_reference_ground_mask(las)
    results_by_method = {}
    exports_by_method = {}
    comparison_rows = []

    runners = {
        "CSF": lambda pts: run_csf(pts, CSF_PARAMS),
        "PMF": lambda pts: run_pmf(pts, PMF_PARAMS),
        "SMRF": lambda pts: run_smrf(pts, SMRF_PARAMS),
        "PTD": lambda pts: run_ptd(pts, PTD_PARAMS),
    }

    print("[3/7] 开始执行多方法滤波...", flush=True)
    for method in METHODS_TO_RUN:
        print(f"\n    >>> 正在执行 {method} ...", flush=True)
        result = runners[method](xyz_filtered)
        export = export_method_outputs(las, xyz, keep_mask, result, OUTPUT_DIR, method)
        results_by_method[method] = result
        exports_by_method[method] = export

        row = {
            "Method": method,
            "GroundPoints": int(np.sum(export["full_ground_mask"])),
            "NonGroundPoints": int(np.sum(export["full_nonground_mask"])),
        }

        reg = regression_metrics(result["z_surface"], xyz_filtered[:, 2])
        row.update({
            "Scatter_R": reg["R"],
            "Scatter_R2": reg["R2"],
            "Scatter_RMSE": reg["RMSE"],
            "Scatter_MAE": reg["MAE"],
            "Scatter_Bias": reg["Bias"],
        })

        if ref_ground_mask is not None and np.any(ref_ground_mask):
            pred = export["full_ground_mask"].astype(np.uint8)
            truth = ref_ground_mask.astype(np.uint8)
            cls = classification_metrics(truth, pred)
            row.update({
                "N": cls["N"],
                "OA": cls["OA"],
                "Kappa": cls["Kappa"],
                "Precision_ground": cls["Precision_ground"],
                "Recall_ground": cls["Recall_ground"],
                "F1_ground": cls["F1_ground"],
            })
            with open(os.path.join(export["method_dir"], f"{method.lower()}_accuracy_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(cls, f, ensure_ascii=False, indent=2)
            pd.DataFrame([{
                "N": cls["N"],
                "OA": cls["OA"],
                "Kappa": cls["Kappa"],
                "Precision_ground": cls["Precision_ground"],
                "Recall_ground": cls["Recall_ground"],
                "F1_ground": cls["F1_ground"],
            }]).to_csv(
                os.path.join(export["method_dir"], f"{method.lower()}_accuracy_metrics.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            save_plotly_confusion_matrix(
                np.array(cls["cm_ground_nonground_order"], dtype=int),
                os.path.join(export["method_dir"], f"{method.lower()}_confusion_matrix.html"),
                labels=("Ground", "Non-ground"),
                title=f"Reference vs {method}",
            )

        comparison_rows.append(row)
        print(f"    >>> {method} 完成", flush=True)

    print("\n[4/7] 汇总精度对比...", flush=True)
    comp_df = pd.DataFrame(comparison_rows)
    if ("OA" in comp_df.columns) and ("F1_ground" in comp_df.columns):
        comp_df = comp_df.sort_values(by=["F1_ground", "OA"], ascending=False)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False, encoding="utf-8-sig")
    comp_df.to_json(os.path.join(OUTPUT_DIR, "comparison_summary.json"), orient="records", force_ascii=False, indent=2)
    save_comparison_bar(comp_df, os.path.join(OUTPUT_DIR, "comparison_accuracy_bar.html"))

    print("[5/7] 输出参数记录...", flush=True)
    run_config = {
        "input_las": INPUT_LAS,
        "output_dir": OUTPUT_DIR,
        "do_outlier_removal": DO_OUTLIER_REMOVAL,
        "nb_neighbors": NB_NEIGHBORS,
        "std_ratio": STD_RATIO,
        "methods_to_run": METHODS_TO_RUN,
        "CSF_PARAMS": CSF_PARAMS,
        "PMF_PARAMS": PMF_PARAMS,
        "SMRF_PARAMS": SMRF_PARAMS,
        "PTD_PARAMS": PTD_PARAMS,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("[6/7] 控制台输出对比结果...", flush=True)
    print("\n==================== 方法对比汇总 ====================", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(comp_df.to_string(index=False), flush=True)

    if SHOW_FINAL_OPEN3D_WINDOW and FINAL_VIS_METHOD in exports_by_method:
        print(f"[7/7] 打开最终 Open3D 窗口: {FINAL_VIS_METHOD}", flush=True)
        exp = exports_by_method[FINAL_VIS_METHOD]
        visualize_final_result(
            full_xyz=xyz,
            ground_mask=exp["full_ground_mask"],
            nonground_mask=exp["full_nonground_mask"],
            window_name=f"{FINAL_VIS_METHOD} Filtering Result - Interactive View",
            max_points=FINAL_VIS_MAX_POINTS,
            point_size=FINAL_VIS_POINT_SIZE,
        )

    print("\n全部处理完成。", flush=True)
    print("每个方法目录下均输出：", flush=True)
    print("- *_ground.las", flush=True)
    print("- *_nonground.las", flush=True)
    print("- *_classified.las", flush=True)
    print("- *_surface.html", flush=True)
    print("- *_dz.html", flush=True)
    print("- *_surface_vs_points_scatter.html", flush=True)
    print("- *_3d_points.html", flush=True)
    print("- *_3d_surface.html", flush=True)
    print("- *_3d_points_surface.html", flush=True)
    print("若存在 classification=2，还会输出：", flush=True)
    print("- *_accuracy_metrics.csv / json", flush=True)
    print("- *_confusion_matrix.html", flush=True)
    print("总目录还会输出：comparison_summary.csv / json / html", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
