import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

ALIGN_PARAMS_NAME = "race_align_params.npz"
REPORT_NAME = "race_eval_report.json"
ALIGNED_PLY_NAME = "aligned_our_model.ply"


def load_paths(source_path, model_path):
    our_mesh_path = os.path.join(model_path, "mesh", "tsdf_fusion_post.ply")
    gt_cloud_path = os.path.join(source_path, "gt", "gt_pd.ply")
    align_params_path = os.path.join(model_path, ALIGN_PARAMS_NAME)

    if not os.path.exists(our_mesh_path):
        raise FileNotFoundError(
            f"未找到重建网格 '{our_mesh_path}'，请先运行 render.py。"
        )
    if not os.path.exists(gt_cloud_path):
        raise FileNotFoundError(
            f"未找到真值点云 '{gt_cloud_path}'，请检查竞赛数据集路径。"
        )

    return our_mesh_path, gt_cloud_path, align_params_path


def load_gt_points(gt_cloud_path):
    gt_pcd = o3d.io.read_point_cloud(gt_cloud_path)
    gt_points = np.asarray(gt_pcd.points, dtype=np.float64)
    if gt_points.size == 0:
        raise ValueError(f"真值点云为空: {gt_cloud_path}")
    return gt_points


def load_reconstruction_mesh(our_mesh_path):
    mesh = trimesh.load(our_mesh_path, process=False)
    if mesh.is_empty:
        raise ValueError(f"重建网格为空: {our_mesh_path}")
    return mesh


def mesh_to_o3d_pointcloud(mesh, num_samples, seed):
    points, _ = trimesh.sample.sample_surface(mesh, num_samples, seed=seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def voxel_downsample_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(
        pcd.voxel_down_sample(voxel_size).points,
        dtype=np.float64,
    )


def extract_similarity_scale(transform):
    return float(np.linalg.norm(transform[:3, 0]))


def run_multistage_icp(source_pcd, target_pcd, thresholds):
    transform = np.eye(4)
    last_result = None

    for threshold in thresholds:
        last_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=True
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
        )
        transform = last_result.transformation

    return transform, last_result


def compute_alignment(mesh, gt_points, align_num_samples, align_seed, voxel_size, icp_thresholds):
    print(">> [对齐] 从重建 mesh 采样（固定 seed，仅用于一次性对齐）...")
    align_pcd = mesh_to_o3d_pointcloud(mesh, align_num_samples, align_seed)

    print(f">> [对齐] 体素降采样 (voxel_size={voxel_size:.3f} m)...")
    source_points = voxel_downsample_points(np.asarray(align_pcd.points), voxel_size)
    target_points = voxel_downsample_points(gt_points, voxel_size)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    print(f">> [对齐] 多阶段 ICP (thresholds={icp_thresholds})...")
    transformation, icp_result = run_multistage_icp(source_pcd, target_pcd, icp_thresholds)
    scale = extract_similarity_scale(transformation)

    print(">> [对齐] 完成。变换矩阵:\n", transformation)
    print(f">> [对齐] 估计尺度 scale={scale:.6f}, fitness={icp_result.fitness:.4f}, "
          f"inlier_rmse={icp_result.inlier_rmse:.4f} m")

    return {
        "transformation": transformation,
        "scale": scale,
        "fitness": float(icp_result.fitness),
        "inlier_rmse_m": float(icp_result.inlier_rmse),
        "align_num_samples": align_num_samples,
        "align_seed": align_seed,
        "voxel_size_m": voxel_size,
        "icp_thresholds_m": np.asarray(icp_thresholds, dtype=np.float64),
        "aligned_at": datetime.now(timezone.utc).isoformat(),
        "method": "multistage_icp_with_scaling",
    }


def save_alignment(align_params_path, align_dict):
    np.savez(
        align_params_path,
        transformation=align_dict["transformation"],
        scale=align_dict["scale"],
        fitness=align_dict["fitness"],
        inlier_rmse_m=align_dict["inlier_rmse_m"],
        align_num_samples=align_dict["align_num_samples"],
        align_seed=align_dict["align_seed"],
        voxel_size_m=align_dict["voxel_size_m"],
        icp_thresholds_m=align_dict["icp_thresholds_m"],
        aligned_at=align_dict["aligned_at"],
        method=align_dict["method"],
    )
    print(f">> [对齐] 参数已保存至: {align_params_path}")


def load_alignment(align_params_path):
    if not os.path.exists(align_params_path):
        raise FileNotFoundError(
            f"未找到固定对齐文件 '{align_params_path}'。"
            f"请先运行: python race_eval.py --mode align -s <data_path> -m <output_path>"
        )

    align_data = np.load(align_params_path)
    return {
        "transformation": align_data["transformation"],
        "scale": float(align_data["scale"]),
        "fitness": float(align_data["fitness"]),
        "inlier_rmse_m": float(align_data["inlier_rmse_m"]),
        "align_num_samples": int(align_data["align_num_samples"]),
        "align_seed": int(align_data["align_seed"]),
        "voxel_size_m": float(align_data["voxel_size_m"]),
        "icp_thresholds_m": align_data["icp_thresholds_m"].tolist(),
        "aligned_at": str(align_data["aligned_at"]),
        "method": str(align_data["method"]),
    }


def apply_transform_to_mesh(mesh, transformation):
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(transformation)
    return aligned_mesh


def sample_mesh_points(mesh, num_samples, seed):
    points, _ = trimesh.sample.sample_surface(mesh, num_samples, seed=seed)
    return np.asarray(points, dtype=np.float64)


def compute_point_to_gt_distances(rec_points, gt_points):
    gt_tree = cKDTree(gt_points)
    distances, _ = gt_tree.query(rec_points, k=1, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def summarize_distances(distances):
    return {
        "mean_error_m": float(np.mean(distances)),
        "mean_error_cm": float(np.mean(distances) * 100),
        "median_error_m": float(np.median(distances)),
        "median_error_cm": float(np.median(distances) * 100),
        "rmse_m": float(np.sqrt(np.mean(distances ** 2))),
        "rmse_cm": float(np.sqrt(np.mean(distances ** 2)) * 100),
        "percentile_90_m": float(np.percentile(distances, 90)),
        "percentile_90_cm": float(np.percentile(distances, 90) * 100),
        "percentile_95_m": float(np.percentile(distances, 95)),
        "percentile_95_cm": float(np.percentile(distances, 95) * 100),
        "inliers_20cm_percent": float(np.mean(distances < 0.20) * 100),
        "inliers_10cm_percent": float(np.mean(distances < 0.10) * 100),
    }


def print_metrics(sequence_name, metrics):
    print("\n" + "=" * 20 + f" {sequence_name} 3D 几何精度报告 " + "=" * 20)
    print(f"  平均距离误差 (Mean Error):     {metrics['mean_error_cm']:.2f} cm")
    print(f"  中位数距离误差 (Median Error): {metrics['median_error_cm']:.2f} cm")
    print(f"  均方根误差 (RMSE):            {metrics['rmse_cm']:.2f} cm")
    print(f"  90% 分位误差:                {metrics['percentile_90_cm']:.2f} cm")
    print(f"  95% 分位误差:                {metrics['percentile_95_cm']:.2f} cm")
    print(f"  ★ 20 cm 内点比例 (Inliers):   {metrics['inliers_20cm_percent']:.2f} %")
    print(f"  ★ 10 cm 内点比例 (加分项):     {metrics['inliers_10cm_percent']:.2f} %")
    print("=" * 67)


def run_align(args, our_mesh_path, gt_cloud_path, align_params_path):
    mesh = load_reconstruction_mesh(our_mesh_path)
    gt_points = load_gt_points(gt_cloud_path)
    thresholds = [float(x) for x in args.icp_thresholds.split(",")]

    align_dict = compute_alignment(
        mesh,
        gt_points,
        align_num_samples=args.align_num_samples,
        align_seed=args.align_seed,
        voxel_size=args.voxel_size,
        icp_thresholds=thresholds,
    )
    save_alignment(align_params_path, align_dict)
    return align_dict


def run_eval(args, source_path, model_path, our_mesh_path, gt_cloud_path, align_params_path):
    mesh = load_reconstruction_mesh(our_mesh_path)
    gt_points = load_gt_points(gt_cloud_path)
    align_dict = load_alignment(align_params_path)

    print(">> [评估] 加载固定对齐参数（评估阶段不做 ICP / scale 优化）...")
    print(f"    method={align_dict['method']}, aligned_at={align_dict['aligned_at']}")
    print(f"    scale={align_dict['scale']:.6f}, fitness={align_dict['fitness']:.4f}")

    aligned_mesh = apply_transform_to_mesh(mesh, align_dict["transformation"])

    print(
        f">> [评估] 固定 seed 采样 mesh 表面 "
        f"(num_samples={args.eval_num_samples}, seed={args.eval_seed})..."
    )
    rec_points = sample_mesh_points(aligned_mesh, args.eval_num_samples, args.eval_seed)

    print(">> [评估] 计算重建点到 GT 的最近邻距离（纯误差统计）...")
    distances = compute_point_to_gt_distances(rec_points, gt_points)
    metrics = summarize_distances(distances)

    sequence_name = os.path.basename(os.path.normpath(source_path))
    print_metrics(sequence_name, metrics)

    report = {
        "sequence": sequence_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": os.path.abspath(source_path),
        "model_path": os.path.abspath(model_path),
        "reconstructed_mesh": os.path.abspath(our_mesh_path),
        "ground_truth_ply": os.path.abspath(gt_cloud_path),
        "num_gt_points": int(gt_points.shape[0]),
        "num_eval_samples": int(rec_points.shape[0]),
        "eval_num_samples": args.eval_num_samples,
        "eval_seed": args.eval_seed,
        "alignment": {
            "params_path": os.path.abspath(align_params_path),
            "method": align_dict["method"],
            "aligned_at": align_dict["aligned_at"],
            "scale": align_dict["scale"],
            "fitness": align_dict["fitness"],
            "inlier_rmse_m": align_dict["inlier_rmse_m"],
            "transformation": align_dict["transformation"].tolist(),
        },
        "metrics": metrics,
    }

    report_path = os.path.join(model_path, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f">> [评估] 指标已保存至: {report_path}")

    aligned_ply_path = os.path.join(model_path, ALIGNED_PLY_NAME)
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(rec_points)
    o3d.io.write_point_cloud(aligned_ply_path, aligned_pcd)
    print(f">> [评估] 对齐后采样点云已导出至: {aligned_ply_path}\n")

    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="GDC2026 室内三维重建竞赛评估：对齐与误差计算分离"
    )
    parser.add_argument("--source_path", "-s", required=True, help="竞赛序列路径")
    parser.add_argument("--model_path", "-m", required=True, help="PlanarGS 输出路径")
    parser.add_argument(
        "--mode",
        choices=["align", "eval", "all"],
        default="all",
        help="align=仅预计算对齐; eval=固定对齐后纯评估; all=先对齐再评估",
    )
    parser.add_argument(
        "--force_align",
        action="store_true",
        help="all/eval 模式下若已有对齐文件，仍重新计算对齐",
    )
    parser.add_argument(
        "--align_num_samples",
        type=int,
        default=200000,
        help="对齐阶段 mesh 表面采样点数（固定 seed）",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=500000,
        help="评估阶段 mesh 表面采样点数（固定 seed）",
    )
    parser.add_argument("--align_seed", type=int, default=42, help="对齐采样随机种子")
    parser.add_argument("--eval_seed", type=int, default=42, help="评估采样随机种子")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="对齐前体素降采样大小（米）",
    )
    parser.add_argument(
        "--icp_thresholds",
        type=str,
        default="0.25,0.12,0.06,0.03",
        help="多阶段 ICP 阈值（米），逗号分隔，从粗到细",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    our_mesh_path, gt_cloud_path, align_params_path = load_paths(
        args.source_path, args.model_path
    )

    need_align = args.mode in ("align", "all")
    need_eval = args.mode in ("eval", "all")

    if need_align or (args.force_align and need_eval):
        run_align(args, our_mesh_path, gt_cloud_path, align_params_path)
    elif need_eval and not os.path.exists(align_params_path):
        raise FileNotFoundError(
            f"评估需要固定对齐文件 '{align_params_path}'。"
            "请先运行 --mode align，或使用 --mode all。"
        )

    if need_eval:
        run_eval(
            args,
            args.source_path,
            args.model_path,
            our_mesh_path,
            gt_cloud_path,
            align_params_path,
        )


if __name__ == "__main__":
    main()
