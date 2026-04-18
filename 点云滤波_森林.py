# -*- coding: utf-8 -*-
"""
基于论文 CSF（Cloth Simulation Filtering）的 LAS 点云地面滤波完整程序
森林场景版：Forest.las
不使用 matplotlib，改用 Plotly 输出交互式 2D/3D HTML，
并在最后弹出 Open3D 可交互窗口。

论文：
Zhang et al., 2016, An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation

功能：
1. 读取 LAS 点云
2. 可选统计离群点去噪
3. 按论文 CSF 思路执行地面滤波
4. 输出：
   - CSF/csf_ground.las
   - CSF/csf_nonground.las
   - CSF/csf_classified.las
   - csf_surface.html                2D 地表热力图
   - csf_dz.html                     2D 残差图
   - csf_surface_vs_points_scatter.html
   - csf_confusion_matrix.html       若存在真值
   - csf_3d_points.html              3D 可拖拽点云分类图
   - csf_3d_surface.html             3D 可拖拽地表图
   - csf_3d_points_surface.html      3D 可拖拽点云+地表叠加图
5. 最后弹出 Open3D 可交互窗口

说明：
- 这是“纯 CSF 版”，不保留 PMF 对比主流程
- 分类阶段使用“点到 cloth surface 的垂向距离”近似论文中的 cloud-to-cloud 距离分类
- 绘图全部使用 Plotly，输出交互式 HTML
"""

import os
import math
import json
import webbrowser
import numpy as np
import pandas as pd
import laspy
import open3d as o3d

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
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
    r2_score
)

import plotly.graph_objects as go


# =========================================================
# 用户配置区
# =========================================================
INPUT_LAS = r"E:\Homework\激光雷达\Forest.las"
OUTPUT_DIR = r"E:\Homework\激光雷达\CSF_Output_Forest"

# ---------- 预处理 ----------
DO_OUTLIER_REMOVAL = True
NB_NEIGHBORS = 20
STD_RATIO = 2.0

# ---------- Plotly 输出控制 ----------
PLOTLY_MAX_HEATMAP_DIM = 2000
PLOTLY_SCATTER_MAX_POINTS = 30000

# 3D 显示采样控制（防止 HTML 太大）
PLOTLY_3D_POINTS_MAX = 50000
PLOTLY_3D_SURFACE_MAX_DIM = 250
PLOTLY_3D_POINTS_SURFACE_POINTS_MAX = 30000

# 是否自动打开 HTML
AUTO_OPEN_HTML = False

# ---------- Open3D 最终显示 ----------
SHOW_FINAL_OPEN3D_WINDOW = True
FINAL_VIS_MAX_POINTS = 200000
FINAL_VIS_POINT_SIZE = 2.0

# ---------- CSF 参数（按论文，针对森林场景） ----------
# 论文中 dT=0.65、GR=0.5 为通用值；RI/ST 随场景调整。
# 森林通常更接近复杂地形/坡地，先用更软的 cloth + 陡坡后处理。
CSF_GR = 0.5
CSF_DT = 0.65
CSF_RI = 1
CSF_ST = True
CSF_HCC = 0.5
CSF_HCP = 0.3
CSF_MAX_ITER = 500
CSF_STOP_THRESHOLD = 0.003
CSF_INITIAL_HEIGHT_OFFSET = 5.0


# =========================================================
# 工具函数
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


def make_o3d_pcd(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None and xyz.shape[0] > 0:
        colors = np.tile(np.array(color, dtype=np.float64), (xyz.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_final_csf_result(full_xyz, ground_mask, nonground_mask,
                               max_points=200000, point_size=2.0):
    """
    最终弹出一个可交互 Open3D 窗口：
    - Ground: 棕色
    - Non-ground: 蓝色
    """
    print("[Open3D] 正在准备最终可视化窗口...", flush=True)

    ground_xyz = full_xyz[ground_mask]
    nonground_xyz = full_xyz[nonground_mask]

    ground_vis = sample_points(ground_xyz, max_points // 2, seed=42) if ground_xyz.shape[0] > 0 else np.empty((0, 3))
    nonground_vis = sample_points(nonground_xyz, max_points // 2, seed=43) if nonground_xyz.shape[0] > 0 else np.empty((0, 3))

    geoms = []

    if ground_vis.shape[0] > 0:
        pcd_ground = make_o3d_pcd(ground_vis, color=[0.76, 0.59, 0.25])
        geoms.append(pcd_ground)

    if nonground_vis.shape[0] > 0:
        pcd_nonground = make_o3d_pcd(nonground_vis, color=[0.20, 0.45, 0.85])
        geoms.append(pcd_nonground)

    if len(geoms) == 0:
        print("[WARN] 没有可显示的点。", flush=True)
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="CSF Filtering Result - Interactive View",
        width=1600,
        height=1000,
        visible=True
    )

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

    print("[Open3D] 窗口已打开，可手动旋转/缩放/平移查看。关闭窗口后程序结束。", flush=True)
    vis.run()
    vis.destroy_window()


def write_las_subset(src_las, mask, out_path):
    new_las = laspy.create(
        point_format=src_las.header.point_format,
        file_version=src_las.header.version
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
        file_version=src_las.header.version
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
        print("[WARN] 当前 LAS 点格式不支持 classification 字段。", flush=True)

    out_las.write(out_path)


def load_las_xyz(las_path):
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    xyz = np.column_stack((x, y, z))
    return las, xyz


def open3d_statistical_outlier_removal(xyz, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
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


def interpolate_surface_to_points(surface_grid, xmin, ymin, grid_size, x, y):
    rows, cols = surface_grid.shape
    ys = ymin + np.arange(rows) * grid_size
    xs = xmin + np.arange(cols) * grid_size

    interp = RegularGridInterpolator(
        (ys, xs),
        surface_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
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
            fill_value=None
        )
        z_surface[nan_mask] = interp_nn(pts[nan_mask])

    return z_surface


def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {
            "N": int(len(y_true)),
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan
        }

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
        "Bias": float(bias)
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
        "F1_ground": float(f1)
    }


def try_get_reference_ground_mask(las):
    """
    从原始 LAS 中读取 classification 字段，
    若存在 ground class=2，则返回布尔掩膜；
    否则返回 None
    """
    try:
        dims = [d for d in las.point_format.dimension_names]
        if "classification" not in dims:
            return None

        cls = np.asarray(las.classification)
        if cls.size == 0:
            return None

        return (cls == 2)

    except Exception as e:
        print(f"[WARN] 读取 classification 字段失败: {e}", flush=True)
        return None


# =========================================================
# Plotly 绘图函数
# =========================================================
def downsample_2d_for_plot(arr, max_dim=2000):
    h, w = arr.shape
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    if step > 1:
        print(f"[INFO] 绘图数组过大，已降采样显示: original=({h},{w}) -> plot={arr[::step, ::step].shape}", flush=True)
    return arr[::step, ::step], step


def save_plotly_heatmap(arr, out_html, title, colorscale="Earth", zmin=None, zmax=None):
    arr_plot, step = downsample_2d_for_plot(arr, max_dim=PLOTLY_MAX_HEATMAP_DIM)

    fig = go.Figure(
        data=go.Heatmap(
            z=arr_plot,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Value")
        )
    )
    fig.update_layout(
        title=f"{title} (display step={step})",
        template="plotly_white",
        width=1100,
        height=850,
        xaxis_title="Column",
        yaxis_title="Row"
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

    save_plotly_heatmap(
        arr,
        out_html,
        title=title,
        colorscale="RdBu",
        zmin=vmin,
        zmax=vmax
    )


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
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=3, opacity=0.35, color="rgba(30,144,255,0.55)"),
        name="Samples"
    ))

    fig.add_trace(go.Scatter(
        x=[xy_min, xy_max],
        y=[xy_min, xy_max],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="1:1 line"
    ))

    text_box = (
        f"N={metrics['N']}<br>"
        f"R={metrics['R']:.4f}<br>"
        f"R²={metrics['R2']:.4f}<br>"
        f"RMSE={metrics['RMSE']:.4f}<br>"
        f"MAE={metrics['MAE']:.4f}<br>"
        f"Bias={metrics['Bias']:.4f}"
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=800,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        annotations=[
            dict(
                x=0.01, y=0.99,
                xref="paper", yref="paper",
                text=text_box,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="black"
            )
        ]
    )

    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_confusion_matrix(cm, out_html, labels=("Ground", "Non-ground"), title="Confusion Matrix"):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=list(labels),
            y=list(labels),
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18}
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=700,
        height=650,
        xaxis_title="Prediction",
        yaxis_title="Reference"
    )
    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_3d_points(full_xyz, ground_mask, nonground_mask, out_html, max_points=50000):
    """
    可拖拽 3D 点云分类图
    """
    ground_xyz = full_xyz[ground_mask]
    nonground_xyz = full_xyz[nonground_mask]

    ground_vis = sample_points(ground_xyz, max_points // 2, seed=42) if ground_xyz.shape[0] > 0 else np.empty((0, 3))
    nonground_vis = sample_points(nonground_xyz, max_points // 2, seed=43) if nonground_xyz.shape[0] > 0 else np.empty((0, 3))

    fig = go.Figure()

    if ground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=ground_vis[:, 0],
            y=ground_vis[:, 1],
            z=ground_vis[:, 2],
            mode="markers",
            name="Ground",
            marker=dict(size=2, color="rgb(194,148,63)", opacity=0.8)
        ))

    if nonground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=nonground_vis[:, 0],
            y=nonground_vis[:, 1],
            z=nonground_vis[:, 2],
            mode="markers",
            name="Non-ground",
            marker=dict(size=2, color="rgb(51,115,217)", opacity=0.7)
        ))

    fig.update_layout(
        title="3D Point Cloud Classification (drag to rotate)",
        template="plotly_white",
        width=1100,
        height=850,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        legend=dict(x=0.02, y=0.98)
    )

    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_3d_surface(surface_grid, xmin, ymin, grid_size, out_html, max_dim=250):
    """
    可拖拽 3D 地表图
    """
    surface_plot, step = downsample_2d_for_plot(surface_grid, max_dim=max_dim)
    rows, cols = surface_plot.shape

    xs = xmin + np.arange(cols) * grid_size * step
    ys = ymin + np.arange(rows) * grid_size * step

    fig = go.Figure(
        data=[
            go.Surface(
                x=xs,
                y=ys,
                z=surface_plot,
                colorscale="Earth",
                colorbar=dict(title="Elevation")
            )
        ]
    )

    fig.update_layout(
        title=f"3D CSF Surface (display step={step}, drag to rotate)",
        template="plotly_white",
        width=1100,
        height=850,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        )
    )

    fig.write_html(out_html)
    maybe_open_html(out_html)


def save_plotly_3d_points_surface(full_xyz, ground_mask, nonground_mask,
                                  surface_grid, xmin, ymin, grid_size,
                                  out_html,
                                  max_points=30000, max_dim=250):
    """
    可拖拽 3D 点云 + CSF 地表叠加图
    """
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
        x=xs,
        y=ys,
        z=surface_plot,
        colorscale="Earth",
        opacity=0.65,
        name="CSF Surface",
        showscale=True,
        colorbar=dict(title="Elevation")
    ))

    if ground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=ground_vis[:, 0],
            y=ground_vis[:, 1],
            z=ground_vis[:, 2],
            mode="markers",
            name="Ground",
            marker=dict(size=2, color="rgb(194,148,63)", opacity=0.85)
        ))

    if nonground_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=nonground_vis[:, 0],
            y=nonground_vis[:, 1],
            z=nonground_vis[:, 2],
            mode="markers",
            name="Non-ground",
            marker=dict(size=2, color="rgb(51,115,217)", opacity=0.6)
        ))

    fig.update_layout(
        title="3D Point Cloud + CSF Surface (drag to rotate)",
        template="plotly_white",
        width=1200,
        height=900,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        )
    )

    fig.write_html(out_html)
    maybe_open_html(out_html)


# =========================================================
# 论文 CSF 核心
# =========================================================
def build_inverted_surface_grid(x, y, z, grid_size):
    """
    构建倒置点云表面：
    原始 z -> z_inv = zmax - z

    对每个 grid cell，取 z_inv 的最小值，
    对应倒置表面的最上边界，cloth 从上往下落时首先与之碰撞。
    """
    zmax = np.max(z)
    z_inv = zmax - z

    xmin, ymin = x.min(), y.min()
    xmax, ymax = x.max(), y.max()

    cols = int(math.ceil((xmax - xmin) / grid_size)) + 1
    rows = int(math.ceil((ymax - ymin) / grid_size)) + 1

    col_idx = np.floor((x - xmin) / grid_size).astype(np.int32)
    row_idx = np.floor((y - ymin) / grid_size).astype(np.int32)

    grid = np.full((rows, cols), np.nan, dtype=np.float64)

    flat_index = row_idx * cols + col_idx
    order = np.argsort(flat_index)
    flat_sorted = flat_index[order]
    z_sorted = z_inv[order]

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
    """
    按论文思想做内部约束：
    相邻粒子如果高度不同，则向同一水平面靠拢
    - 双方都可动：各移动一半
    - 一方不可动：另一方全移动
    RI 决定重复次数，RI 越大，cloth 越硬
    """
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
    """
    陡坡后处理：
    若 movable 粒子邻域中存在 unmovable 粒子，
    且两者对应 CP 高差 <= hcp，则把当前粒子压到地面并设为 unmovable
    """
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
    """
    纯 CSF 主过程：
    1. 点云翻转
    2. 初始化 cloth
    3. gravity step
    4. collision -> unmovable
    5. internal force pass
    6. optional post-process
    7. 反变换得到 terrain surface
    8. 按 hcc 分类
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    grid_size = params["gr"]
    ihv, xmin, ymin, rows, cols, zmax = build_inverted_surface_grid(x, y, z, grid_size)

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
            print(f"       CSF 在第 {it + 1} 次迭代收敛，max_hv={max_hv:.6f}", flush=True)
            break

    if params["steep_slope_fit"]:
        cloth, movable = csf_postprocess(
            cloth=cloth,
            movable=movable,
            ihv=ihv,
            hcp=params["hcp"],
            max_rounds=50
        )

    terrain_grid = zmax - cloth

    z_surface = interpolate_surface_to_points(terrain_grid, xmin, ymin, grid_size, x, y)
    dz = z - z_surface
    ground_mask = dz <= params["class_threshold"]

    return {
        "surface_grid": terrain_grid,
        "xmin": xmin,
        "ymin": ymin,
        "grid_size": grid_size,
        "z_surface": z_surface,
        "dz": dz,
        "ground_mask": ground_mask,
        "cloth_grid_inverted": cloth,
        "ihv": ihv,
        "iterations": final_iter,
        "final_max_hv": final_max_hv
    }


# =========================================================
# 导出
# =========================================================
def export_csf_outputs(src_las, full_xyz, keep_mask, result_dict, out_dir):
    method_dir = os.path.join(out_dir, "CSF")
    ensure_dir(method_dir)

    print("       [CSF] 开始整理分类掩膜...", flush=True)

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

    classified_path = os.path.join(method_dir, "csf_classified.las")
    ground_path = os.path.join(method_dir, "csf_ground.las")
    nonground_path = os.path.join(method_dir, "csf_nonground.las")

    print("       [CSF] 写 csf_classified.las ...", flush=True)
    write_las_with_classification(src_las, full_classification, classified_path)

    print("       [CSF] 写 csf_ground.las ...", flush=True)
    write_las_subset(src_las, full_ground_mask, ground_path)

    print("       [CSF] 写 csf_nonground.las ...", flush=True)
    write_las_subset(src_las, full_nonground_mask, nonground_path)

    print("       [CSF] 保存 surface 图（Plotly HTML）...", flush=True)
    print(f"       [CSF] surface_grid shape = {result_dict['surface_grid'].shape}", flush=True)
    save_plotly_heatmap(
        result_dict["surface_grid"],
        os.path.join(method_dir, "csf_surface.html"),
        title="CSF Surface",
        colorscale="Earth"
    )
    print("       [CSF] surface 图保存完成", flush=True)

    print("       [CSF] 保存 dz 图（Plotly HTML）...", flush=True)
    dz = result_dict["dz"]
    dz_grid = np.full_like(result_dict["surface_grid"], np.nan, dtype=np.float64)

    x = full_xyz[keep_mask][:, 0]
    y = full_xyz[keep_mask][:, 1]
    xmin = result_dict["xmin"]
    ymin = result_dict["ymin"]
    grid_size = result_dict["grid_size"]
    rows, cols = dz_grid.shape

    col_idx = np.floor((x - xmin) / grid_size).astype(np.int32)
    row_idx = np.floor((y - ymin) / grid_size).astype(np.int32)
    valid = (row_idx >= 0) & (row_idx < rows) & (col_idx >= 0) & (col_idx < cols)

    sum_grid = np.zeros((rows, cols), dtype=np.float64)
    cnt_grid = np.zeros((rows, cols), dtype=np.int32)

    rr = row_idx[valid]
    cc = col_idx[valid]
    vv = dz[valid]

    np.add.at(sum_grid, (rr, cc), vv)
    np.add.at(cnt_grid, (rr, cc), 1)

    mask = cnt_grid > 0
    dz_grid[mask] = sum_grid[mask] / cnt_grid[mask]

    save_plotly_difference_map(
        dz_grid,
        os.path.join(method_dir, "csf_dz.html"),
        title="Point minus CSF Surface (dz)"
    )
    print("       [CSF] dz 图保存完成", flush=True)

    print("       [CSF] 保存三维点云分类图（Plotly HTML）...", flush=True)
    save_plotly_3d_points(
        full_xyz,
        full_ground_mask,
        full_nonground_mask,
        os.path.join(method_dir, "csf_3d_points.html"),
        max_points=PLOTLY_3D_POINTS_MAX
    )

    print("       [CSF] 保存三维地表图（Plotly HTML）...", flush=True)
    save_plotly_3d_surface(
        result_dict["surface_grid"],
        result_dict["xmin"],
        result_dict["ymin"],
        result_dict["grid_size"],
        os.path.join(method_dir, "csf_3d_surface.html"),
        max_dim=PLOTLY_3D_SURFACE_MAX_DIM
    )

    print("       [CSF] 保存三维点云+地表叠加图（Plotly HTML）...", flush=True)
    save_plotly_3d_points_surface(
        full_xyz,
        full_ground_mask,
        full_nonground_mask,
        result_dict["surface_grid"],
        result_dict["xmin"],
        result_dict["ymin"],
        result_dict["grid_size"],
        os.path.join(method_dir, "csf_3d_points_surface.html"),
        max_points=PLOTLY_3D_POINTS_SURFACE_POINTS_MAX,
        max_dim=PLOTLY_3D_SURFACE_MAX_DIM
    )

    print("       [CSF] 所有图形导出完成", flush=True)

    return {
        "method_dir": method_dir,
        "full_ground_mask": full_ground_mask,
        "full_nonground_mask": full_nonground_mask,
        "full_classification": full_classification
    }


# =========================================================
# 主程序
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 90, flush=True)
    print("基于论文 CSF 的 LAS 点云地面滤波程序（Forest + Plotly 2D/3D + Open3D）", flush=True)
    print("=" * 90, flush=True)
    print(f"输入文件: {INPUT_LAS}", flush=True)
    print(f"输出目录: {OUTPUT_DIR}", flush=True)
    print("森林场景参数策略：按论文复杂地形/坡地思路，使用软 cloth + 陡坡后处理", flush=True)
    print(f"CSF 参数: GR={CSF_GR}, dT={CSF_DT}, RI={CSF_RI}, ST={CSF_ST}, hcc={CSF_HCC}, hcp={CSF_HCP}", flush=True)

    # 1 读取
    las, xyz = load_las_xyz(INPUT_LAS)
    print(f"[1/6] 读取完成，点数: {xyz.shape[0]:,}", flush=True)

    # 2 去噪
    if DO_OUTLIER_REMOVAL:
        print("[2/6] 开始统计离群点去噪...", flush=True)
        xyz_filtered, keep_mask = open3d_statistical_outlier_removal(
            xyz, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO
        )
        print(
            f"       去噪完成，保留点数: {xyz_filtered.shape[0]:,}，移除点数: {xyz.shape[0] - xyz_filtered.shape[0]:,}",
            flush=True
        )
    else:
        xyz_filtered = xyz.copy()
        keep_mask = np.ones(xyz.shape[0], dtype=bool)
        print("[2/6] 跳过去噪", flush=True)

    # 3 CSF
    print("[3/6] 执行论文 CSF 滤波...", flush=True)
    csf_result = run_csf(
        xyz_filtered,
        params={
            "gr": CSF_GR,
            "dt": CSF_DT,
            "ri": CSF_RI,
            "steep_slope_fit": CSF_ST,
            "class_threshold": CSF_HCC,
            "hcp": CSF_HCP,
            "max_iter": CSF_MAX_ITER,
            "stop_threshold": CSF_STOP_THRESHOLD,
            "initial_height_offset": CSF_INITIAL_HEIGHT_OFFSET
        }
    )
    print("       CSF 完成", flush=True)
    print(f"       迭代次数: {csf_result['iterations']}", flush=True)
    print(f"       最终 max_hv: {csf_result['final_max_hv']:.6f}", flush=True)

    # 4 导出
    print("[4/6] 导出 CSF 结果...", flush=True)
    csf_export = export_csf_outputs(las, xyz, keep_mask, csf_result, OUTPUT_DIR)

    # 5 真值评价 + 散点图
    print("[5/6] 检查是否存在真值分类字段...", flush=True)
    ref_ground_mask = try_get_reference_ground_mask(las)

    summary = {
        "input_las": INPUT_LAS,
        "n_total_points": int(xyz.shape[0]),
        "n_filtered_points": int(xyz_filtered.shape[0]),
        "csf_params": {
            "CSF_GR": CSF_GR,
            "CSF_DT": CSF_DT,
            "CSF_RI": CSF_RI,
            "CSF_ST": CSF_ST,
            "CSF_HCC": CSF_HCC,
            "CSF_HCP": CSF_HCP,
            "CSF_MAX_ITER": CSF_MAX_ITER,
            "CSF_STOP_THRESHOLD": CSF_STOP_THRESHOLD,
            "CSF_INITIAL_HEIGHT_OFFSET": CSF_INITIAL_HEIGHT_OFFSET
        },
        "csf_iterations": int(csf_result["iterations"]),
        "csf_final_max_hv": float(csf_result["final_max_hv"]),
        "csf_ground_points": int(np.sum(csf_export["full_ground_mask"])),
        "csf_nonground_points": int(np.sum(csf_export["full_nonground_mask"]))
    }

    z_surface = csf_result["z_surface"]
    z_points = xyz_filtered[:, 2]
    save_plotly_scatter(
        z_surface,
        z_points,
        os.path.join(OUTPUT_DIR, "csf_surface_vs_points_scatter.html"),
        xlabel="CSF Surface Elevation (m)",
        ylabel="Point Elevation (m)",
        title="CSF Surface vs Original Point Elevation"
    )

    if ref_ground_mask is not None and np.any(ref_ground_mask):
        print("       检测到 classification=2，开始精度评价...", flush=True)

        csf_cls = csf_export["full_ground_mask"].astype(np.uint8)
        ref_cls = ref_ground_mask.astype(np.uint8)

        metrics = classification_metrics(ref_cls, csf_cls)
        summary["truth_metrics"] = metrics

        with open(os.path.join(OUTPUT_DIR, "csf_accuracy_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        pd.DataFrame([{
            "N": metrics["N"],
            "OA": metrics["OA"],
            "Kappa": metrics["Kappa"],
            "Precision_ground": metrics["Precision_ground"],
            "Recall_ground": metrics["Recall_ground"],
            "F1_ground": metrics["F1_ground"]
        }]).to_csv(
            os.path.join(OUTPUT_DIR, "csf_accuracy_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        save_plotly_confusion_matrix(
            np.array(metrics["cm_ground_nonground_order"], dtype=int),
            os.path.join(OUTPUT_DIR, "csf_confusion_matrix.html"),
            labels=("Ground", "Non-ground"),
            title="Reference vs CSF"
        )
    else:
        print("       未检测到可用真值 classification 字段，跳过真值评价。", flush=True)

    with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, "summary_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 6 最终 Open3D 窗口
    if SHOW_FINAL_OPEN3D_WINDOW:
        print("[6/6] 弹出最终 Open3D 结果窗口...", flush=True)
        visualize_final_csf_result(
            full_xyz=xyz,
            ground_mask=csf_export["full_ground_mask"],
            nonground_mask=csf_export["full_nonground_mask"],
            max_points=FINAL_VIS_MAX_POINTS,
            point_size=FINAL_VIS_POINT_SIZE
        )

    print("=" * 90, flush=True)
    print("全部处理完成。", flush=True)
    print("输出内容包括：", flush=True)
    print("1) CSF/csf_ground.las", flush=True)
    print("2) CSF/csf_nonground.las", flush=True)
    print("3) CSF/csf_classified.las", flush=True)
    print("4) csf_surface.html / csf_dz.html", flush=True)
    print("5) csf_surface_vs_points_scatter.html", flush=True)
    print("6) 若有真值字段，则输出精度评价 CSV / JSON / HTML", flush=True)
    print("7) csf_3d_points.html             三维点云分类图（可拖拽）", flush=True)
    print("8) csf_3d_surface.html            三维地表图（可拖拽）", flush=True)
    print("9) csf_3d_points_surface.html     三维点云+地表叠加图（可拖拽）", flush=True)
    print("10) 最后弹出 Open3D 可交互窗口", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()