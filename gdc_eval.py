import os
import sys
import json
import torch
import cv2
import trimesh
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from argparse import ArgumentParser


# ==================== 1. 核心数学度量函数（完全复制原版） ====================
def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc, distances


def completion(gt_points, rec_points):
    rec_points_kd_tree = KDTree(rec_points)
    distances, _ = rec_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp, distances


def nn_correspondance(verts1, verts2):
    if len(verts1) == 0 or len(verts2) == 0:
        return [], []
    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    return distances.reshape(-1), indices


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


# ==================== 2. C++ 级别的向量化全自动配准 ====================
def run_open3d_alignment(mesh_rec_trimesh, mesh_gt_trimesh):
    """
    由于没有官方的 align_params.npz，我们用 Open3D 的 C++ 向量化算法
    全自动解算出原论文需要的 align_scale 和 align_transform
    """
    print("    -> [Auto-Align] 正在利用 C++ 算子提取几何特征并进行全局刚体配准...")

    # 转换为 Open3D 点云以供特征计算
    pcd_rec = o3d.geometry.PointCloud()
    pcd_rec.points = o3d.utility.Vector3dVector(mesh_rec_trimesh.vertices)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(mesh_gt_trimesh.vertices)

    # 对巨量真值做 2cm 下采样，仅用于快速求解变换矩阵，不破坏后续的 Trimesh 表面采样！
    pcd_gt_down = pcd_gt.voxel_down_sample(voxel_size=0.02)
    pcd_rec_down = pcd_rec.voxel_down_sample(voxel_size=0.02)

    voxel_size = 0.1
    pcd_rec_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh_rec = o3d.pipelines.registration.compute_fpfh_feature(pcd_rec_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 5, max_nn=100))

    pcd_gt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(pcd_gt_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 5, max_nn=100))

    # RANSAC 粗配准
    ransac_res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_rec_down, pcd_gt_down, fpfh_rec, fpfh_gt, True, voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999)
    )

    # ICP 锁死精度
    icp_res = o3d.pipelines.registration.registration_icp(
        pcd_rec_down, pcd_gt_down, max_correspondence_distance=0.2, init=ransac_res.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    )

    final_matrix = icp_res.transformation
    # 巧妙解耦出平移旋转矩阵和纯尺度因子，完美投喂给原论文后面的算法
    align_scale = float(np.mean([np.linalg.norm(final_matrix[:3, i]) for i in range(3)]))
    align_transform = final_matrix.copy()
    align_transform[:3, :3] /= align_scale

    return align_transform, align_scale


# ==================== 3. 满血复刻版评测主流程 ====================
def evaluate_recon_metric_perfect(source_path, model_path):
    print(">>> Step 1: Loading Meshes")
    rec_meshfile = os.path.join(model_path, "mesh", "tsdf_fusion_post.ply")
    if not os.path.exists(rec_meshfile):
        rec_meshfile = os.path.join(model_path, "tsdf_fusion_post.ply")

    # 比赛官方给的真值网格文件
    gt_meshfile = os.path.join(source_path, "gt", "gt_pd.ply")

    if not os.path.exists(rec_meshfile):
        sys.exit(f"Error: Reconstructed mesh not found at {rec_meshfile}")
    if not os.path.exists(gt_meshfile):
        sys.exit(f"Error: GT mesh not found at {gt_meshfile}")

    output_dir = os.path.join(model_path, "gdc_perfect_eval")
    print(f"Output Directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    mesh_rec = trimesh.load(rec_meshfile)
    mesh_gt = trimesh.load(gt_meshfile)

    print(">>> Step 2: Alignment (Advanced Decoupled Solver)")
    # 💥 核心改进：用 Open3D 自动解算出原论文原本要从 npz 中读取的 align_transform 和 align_scale
    align_transform, align_scale = run_open3d_alignment(mesh_rec, mesh_gt)
    print(f"    -> 自动解算成功！Scale: {align_scale:.4f}")

    # 严格遵循原论文的变换先后次序，保证三维坐标绝无拓扑形变
    mesh_rec.apply_scale(align_scale)
    mesh_rec = mesh_rec.apply_transform(align_transform)

    print(">>> Step 3 & 4: Aligning Bounding Boxes (AABB-OBB Transform)")
    # 规避原论文不可用的视角裁剪，直接进入原论文标准的边界框（OBB转AABB）坐标规范化
    gt_obb = mesh_gt.bounding_box_oriented
    obb2aabb_transform = np.linalg.inv(gt_obb.transform)
    mesh_gt.apply_transform(obb2aabb_transform)
    mesh_rec.apply_transform(obb2aabb_transform)

    # 导出规范化后的网格模型
    mesh_rec.export(os.path.join(output_dir, f"mask_mesh.ply"))
    mesh_gt.export(os.path.join(output_dir, f"gt_mesh.ply"))

    print(">>> Step 5: Calculating Metrics via Surface Uniform Sampling")
    # 统一设定采样面片数点（原论文默认值 300,000）
    MESH_SAMPLE = 300000
    FSCORE_THRESH = 0.05  # 论文原版的 5cm 精细几何阈值

    # 严格按照论文逻辑：在对齐规范化后的三维网格表面进行均匀随机抽样
    rec_pc = trimesh.sample.sample_surface(mesh_rec, MESH_SAMPLE)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, MESH_SAMPLE)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    # 💥 利用 Scipy 的 C 优化 KDTree 进行高效率测距（由于采样点限制在30万，此时绝不会卡死！）
    accuracy_rec, dist_d2s = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec, dist_s2d = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)

    # 严格复刻原论文的 F-Score 召回率/准确率互查机制
    precision_ratio_rec = np.mean((dist_d2s < FSCORE_THRESH).astype(float))
    completion_ratio_rec = np.mean((dist_s2d < FSCORE_THRESH).astype(float))
    fscore = 2 * precision_ratio_rec * completion_ratio_rec / (completion_ratio_rec + precision_ratio_rec) if (
                                                                                                                          completion_ratio_rec + precision_ratio_rec) > 0 else 0

    # 【额外赠送】完美满足竞赛要求的 20cm 和 10cm 内点比例指标
    inliers_20cm = np.mean((dist_d2s < 0.20).astype(float))
    inliers_10cm = np.mean((dist_d2s < 0.10).astype(float))

    # 💥 满血恢复：原论文最引以为傲的“法线表面一致性 (Normal Consistency)”度量算法
    print("    -> 正在计算网格表面法线一致性张量...")
    pointcloud_pred, idx = mesh_rec.sample(MESH_SAMPLE, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normal_pred = mesh_rec.face_normals[idx]

    pointcloud_trgt, idx = mesh_gt.sample(MESH_SAMPLE, return_index=True)
    pointcloud_trgt = pointcloud_trgt.astype(np.float32)
    normal_trgt = mesh_gt.face_normals[idx]

    _, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
    _, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

    normal_acc = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
    normal_comp = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
    normal_avg = (normal_acc + normal_comp) * 0.5

    # 统一转换指标单位（距离 -> cm，比率 -> %）
    accuracy_rec *= 100
    completion_rec *= 100
    completion_ratio_rec *= 100
    precision_ratio_rec *= 100
    fscore *= 100
    normal_avg *= 100  # 保持与原论文一致的百分制格式

    metrics = {
        "Mean Accuracy (cm)": float(accuracy_rec),
        "Median Error (cm)": float(np.median(dist_d2s) * 100),
        "RMSE (cm)": float(np.sqrt(np.mean(dist_d2s ** 2)) * 100),
        "90% Percentile (cm)": float(np.percentile(dist_d2s, 90) * 100),
        "95% Percentile (cm)": float(np.percentile(dist_d2s, 95) * 100),
        "★ 20cm Inliers (%)": float(inliers_20cm * 100),
        "★ 10cm Inliers (加分项 %)": float(inliers_10cm * 100),
        "Chamfer Distance (cm)": float((accuracy_rec + completion_rec) / 2),
        "★ F-Score (5cm %)": float(fscore),
        "★ Normal Consistency (%)": float(normal_avg)
    }

    # 打印国赛终极战报
    print("\n" + "█" * 20 + " GDC2026 论文级全量精度报告 " + "█" * 20)
    print(f"  ● 平均几何误差 (Mean Accuracy):   {metrics['Mean Accuracy (cm)']:.3f} cm")
    print(f"  ● 中位数距离误差 (Median Error):  {metrics['Median Error (cm)']:.3f} cm")
    print(f"  ● 均方根误差 (RMSE):             {metrics['RMSE (cm)']:.3f} cm")
    print(
        f"  ● 90% / 95% 分位误差:           {metrics['90% Percentile (cm)']:.2f} / {metrics['95% Percentile (cm)']:.2f} cm")
    print(f"  ---------------------------------------------------------")
    print(f"  ★ 20 cm 内点比例 (国赛线):       {metrics['★ 20cm Inliers (%)']:.2f} %")
    print(f"  ★ 10 cm 内点比例 (绝对加分):     {metrics['★ 10cm Inliers (加分项 %)']:.2f} %")
    print(f"  ● Chamfer Distance (倒角距离):   {metrics['Chamfer Distance (cm)']:.3f} cm")
    print(f"  ★ F-Score (5cm 论文核心指标):    {metrics['★ F-Score (5cm %)']:.2f} %")
    print(f"  ★ Normal Consistency (表面法线):  {metrics['★ Normal Consistency (%)']:.2f} %")
    print("█" * 68 + "\n")

    with open(os.path.join(output_dir, "output_perfect.json"), 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    print(">>> Step 6: Visualizing Error (Perfect Mesh Heatmap)")
    # 完全采用原论文的 cv2.applyColorMap 结合高阶剪切逻辑渲染热力图
    VIS_DIST = 0.10  # 10cm 可视化误差最大上限
    stl_alpha = (dist_s2d.clip(max=VIS_DIST) / VIS_DIST).reshape(-1, 1)
    im_gray = (stl_alpha * 255).astype(np.uint8)
    stl_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)[:, 0, [2, 0, 1]] / 255.

    # 将染色后的均匀点云保存，供你在 MeshLab/CloudCompare 中直接查看彩色误差
    write_vis_pcd(os.path.join(output_dir, 'mesh_error_heatmap.ply'), gt_pc_tri.vertices, stl_color)
    print(f">> [大获全胜] 完美适配版数据与热力图已生成至: {output_dir}/")


if __name__ == '__main__':
    parser = ArgumentParser(description="GDC2026 完美学术+竞赛双满足评测程序")
    parser.add_argument('--source_path', '-s', required=True)
    parser.add_argument('--model_path', '-m', required=True)
    args = parser.parse_args()

    evaluate_recon_metric_perfect(args.source_path, args.model_path)
