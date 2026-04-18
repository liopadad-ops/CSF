# -*- coding: utf-8 -*-
import os
import numpy as np
import laspy
import open3d as o3d


# =========================================================
# 路径配置
# =========================================================
CITY_LAS = r"E:\Homework\激光雷达\City.las"
FOREST_LAS = r"E:\Homework\激光雷达\Forest.las"

# 每个点云最多显示多少点，太大会卡
MAX_POINTS_VIS = 300000

# Open3D 点大小
POINT_SIZE = 1.5

# 是否把两个点云在同一个窗口中同时显示
SHOW_BOTH_IN_ONE_WINDOW = False

# 同一窗口显示时，两个点云是否自动左右错开，避免重叠
SEPARATE_WHEN_SHOW_BOTH = True


# =========================================================
# 工具函数
# =========================================================
def check_file_exists(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")


def read_las_xyz(las_path):
    """
    读取 LAS 文件，返回 Nx3 的 xyz 点坐标
    """
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    xyz = np.column_stack((x, y, z))
    return xyz


def random_sample_points(xyz, max_points=300000, seed=42):
    """
    随机采样，防止点数过多导致显示卡顿
    """
    n = xyz.shape[0]
    if n <= max_points:
        return xyz

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx]


def normalize_z_to_color(z):
    """
    按高程着色：
    低处偏蓝 -> 中间偏绿 -> 高处偏黄/红
    这里使用 Open3D 可直接接受的 RGB [0,1]
    """
    z = np.asarray(z, dtype=np.float64)
    z_min = np.min(z)
    z_max = np.max(z)

    if z_max - z_min < 1e-12:
        t = np.zeros_like(z)
    else:
        t = (z - z_min) / (z_max - z_min)

    # 一个简单好看的渐变
    r = np.clip(1.5 * t, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(t - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - t), 0, 1)

    colors = np.column_stack((r, g, b))
    return colors


def make_o3d_point_cloud(xyz, colors=None):
    """
    构建 Open3D 点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def preprocess_for_visualization(xyz, max_points=300000, seed=42):
    """
    显示前预处理：
    1. 随机采样
    2. 按高程着色
    3. 转成 Open3D 点云
    """
    xyz_vis = random_sample_points(xyz, max_points=max_points, seed=seed)
    colors = normalize_z_to_color(xyz_vis[:, 2])
    pcd = make_o3d_point_cloud(xyz_vis, colors)
    return xyz_vis, pcd


def print_basic_info(name, xyz):
    """
    打印点云基本信息
    """
    xmin, ymin, zmin = np.min(xyz, axis=0)
    xmax, ymax, zmax = np.max(xyz, axis=0)

    print("=" * 70)
    print(f"{name} 点云信息")
    print("=" * 70)
    print(f"点数: {xyz.shape[0]:,}")
    print(f"X范围: [{xmin:.3f}, {xmax:.3f}]")
    print(f"Y范围: [{ymin:.3f}, {ymax:.3f}]")
    print(f"Z范围: [{zmin:.3f}, {zmax:.3f}]")
    print("=" * 70)


def show_one_cloud(pcd, window_name="Point Cloud", point_size=1.5):
    """
    显示单个点云
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900, visible=True)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0], dtype=np.float64)
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


def translate_cloud_for_side_by_side(xyz_left, xyz_right):
    """
    为了同时显示两个点云时避免重叠，
    把右边点云沿 X 方向平移一段距离
    """
    left_min = np.min(xyz_left, axis=0)
    left_max = np.max(xyz_left, axis=0)
    right_min = np.min(xyz_right, axis=0)

    left_width = left_max[0] - left_min[0]
    gap = left_width * 0.2 + 20.0  # 留一点间隔

    xyz_right_shifted = xyz_right.copy()
    xyz_right_shifted[:, 0] = xyz_right_shifted[:, 0] - right_min[0] + left_max[0] + gap

    return xyz_left.copy(), xyz_right_shifted


def show_two_clouds(city_xyz, forest_xyz, point_size=1.5, separate=True):
    """
    在同一个窗口中同时显示两个点云
    City：按高程色带
    Forest：按高程色带
    """
    city_vis = random_sample_points(city_xyz, MAX_POINTS_VIS, seed=42)
    forest_vis = random_sample_points(forest_xyz, MAX_POINTS_VIS, seed=43)

    if separate:
        city_vis, forest_vis = translate_cloud_for_side_by_side(city_vis, forest_vis)

    city_colors = normalize_z_to_color(city_vis[:, 2])
    forest_colors = normalize_z_to_color(forest_vis[:, 2])

    city_pcd = make_o3d_point_cloud(city_vis, city_colors)
    forest_pcd = make_o3d_point_cloud(forest_vis, forest_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="City + Forest Point Cloud", width=1600, height=950, visible=True)

    vis.add_geometry(city_pcd)
    vis.add_geometry(forest_pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0], dtype=np.float64)
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


# =========================================================
# 主程序
# =========================================================
def main():
    print("开始检查文件...")
    check_file_exists(CITY_LAS)
    check_file_exists(FOREST_LAS)

    print("读取 City 点云...")
    city_xyz = read_las_xyz(CITY_LAS)
    print_basic_info("City", city_xyz)

    print("读取 Forest 点云...")
    forest_xyz = read_las_xyz(FOREST_LAS)
    print_basic_info("Forest", forest_xyz)

    print("预处理 City 显示数据...")
    _, city_pcd = preprocess_for_visualization(city_xyz, max_points=MAX_POINTS_VIS, seed=42)

    print("预处理 Forest 显示数据...")
    _, forest_pcd = preprocess_for_visualization(forest_xyz, max_points=MAX_POINTS_VIS, seed=43)

    # 先分别显示
    print("打开 City 可视化窗口...")
    show_one_cloud(city_pcd, window_name="City Point Cloud", point_size=POINT_SIZE)

    print("打开 Forest 可视化窗口...")
    show_one_cloud(forest_pcd, window_name="Forest Point Cloud", point_size=POINT_SIZE)

    # 再同时显示
    if SHOW_BOTH_IN_ONE_WINDOW:
        print("打开 City + Forest 同时显示窗口...")
        show_two_clouds(
            city_xyz,
            forest_xyz,
            point_size=POINT_SIZE,
            separate=SEPARATE_WHEN_SHOW_BOTH
        )

    print("显示完成。")


if __name__ == "__main__":
    main()