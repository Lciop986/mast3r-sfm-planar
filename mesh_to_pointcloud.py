#!/usr/bin/env python3
"""
mesh_to_pointcloud.py — 使用 Open3D 将三角网格（Mesh）采样为点云（Point Cloud）

优先使用 Poisson Disk Sampling（蓝噪声均匀分布），不可用时回退到 Uniform Sampling。

单文件用法:
    python mesh_to_pointcloud.py \\
        --input output/Sequence_01/mesh/tsdf_fusion_post.ply \\
        --output output/Sequence_01/sampled_point_cloud.ply \\
        --num_points 2500000

按序列名自动解析路径（共 5 个场景）:
    python mesh_to_pointcloud.py --sequence Sequence_01
    python mesh_to_pointcloud.py --all-sequences

路径约定:
    输入 mesh : output/<sequence>/mesh/tsdf_fusion_post.ply
    参考 GT   : indoor_dataset_phone/data/<sequence>/gt/gt_pd.ply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# 与 GT 点云数量对齐的默认采样点数
DEFAULT_NUM_POINTS = 2_500_000
DEFAULT_INIT_FACTOR = 5

OUTPUT_BASE = Path("output")
DATA_BASE = Path("indoor_dataset_phone/data")
MESH_NAME = "tsdf_fusion_post.ply"
SEQUENCES = [
    "Sequence_01",
    "Sequence_02",
    "Sequence_03",
    "Sequence_04",
    "Sequence_05",
]


class MeshToPointCloudError(Exception):
    """网格转点云过程中的可预期错误。"""


def resolve_input_path(input_path: str) -> Path:
    """解析并校验输入路径，支持省略 .ply 后缀。"""
    path = Path(input_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".ply")

    if not path.exists():
        raise MeshToPointCloudError(f"输入文件不存在: {path}")
    if not path.is_file():
        raise MeshToPointCloudError(f"输入路径不是文件: {path}")
    return path.resolve()


def resolve_output_path(output_path: str) -> Path:
    """解析输出路径，自动创建父目录。"""
    path = Path(output_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".ply")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def sequence_paths(sequence: str, output_base: Path) -> tuple[Path, Path]:
    """根据序列名生成默认输入/输出路径。"""
    if sequence not in SEQUENCES:
        raise MeshToPointCloudError(
            f"未知序列: {sequence}，可选: {', '.join(SEQUENCES)}"
        )
    input_path = output_base / sequence / "mesh" / MESH_NAME
    output_path = output_base / sequence / "sampled_point_cloud.ply"
    return input_path, output_path


def load_triangle_mesh(input_path: Path) -> o3d.geometry.TriangleMesh:
    """读取 PLY 三角网格并做基础有效性检查。"""
    mesh = o3d.io.read_triangle_mesh(str(input_path), print_progress=True)
    if mesh.is_empty():
        raise MeshToPointCloudError(f"Mesh 为空或无法解析: {input_path}")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0:
        raise MeshToPointCloudError(f"Mesh 不含顶点: {input_path}")
    if triangles.size == 0:
        raise MeshToPointCloudError(f"Mesh 不含三角面片: {input_path}")

    # 去除未引用的顶点，避免采样异常
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise MeshToPointCloudError(f"Mesh 清理后为空: {input_path}")

    return mesh


def sample_poisson_disk(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    init_factor: float,
) -> o3d.geometry.PointCloud:
    """
    Poisson Disk Sampling（蓝噪声表面采样）。

    兼容 Open3D 新旧 API：
    - 新版: mesh.sample_points_poisson_disk(...)
    - 旧版: o3d.geometry.sample_points_poisson_disk(mesh, ...)
    """
    if hasattr(mesh, "sample_points_poisson_disk"):
        return mesh.sample_points_poisson_disk(
            number_of_points=num_points,
            init_factor=init_factor,
        )

    if hasattr(o3d.geometry, "sample_points_poisson_disk"):
        return o3d.geometry.sample_points_poisson_disk(
            mesh,
            number_of_points=num_points,
            init_factor=init_factor,
        )

    raise AttributeError("当前 Open3D 版本不支持 Poisson Disk Sampling")


def sample_uniformly(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
) -> o3d.geometry.PointCloud:
    """Uniform Sampling（按三角面面积均匀采样）。"""
    if hasattr(mesh, "sample_points_uniformly"):
        return mesh.sample_points_uniformly(number_of_points=num_points)

    if hasattr(o3d.geometry, "sample_points_uniformly"):
        return o3d.geometry.sample_points_uniformly(mesh, number_of_points=num_points)

    raise AttributeError("当前 Open3D 版本不支持 Uniform Sampling")


def sample_mesh_surface(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    init_factor: float,
) -> tuple[o3d.geometry.PointCloud, str]:
    """
    在 Mesh 表面采样点云。

    Returns:
        (point_cloud, method_name)
    """
    if num_points <= 0:
        raise MeshToPointCloudError(f"采样点数必须为正整数，当前: {num_points}")

    # Poisson Disk 依赖顶点法线；缺失时自动计算
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    try:
        pcd = sample_poisson_disk(mesh, num_points, init_factor)
        method = "poisson_disk"
    except Exception as exc:
        print(
            f"[警告] Poisson Disk Sampling 失败 ({exc})，"
            "回退到 Uniform Sampling。",
            file=sys.stderr,
        )
        pcd = sample_uniformly(mesh, num_points)
        method = "uniform"

    if len(pcd.points) == 0:
        raise MeshToPointCloudError("采样结果为空点云")

    return pcd, method


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: Path) -> None:
    """保存点云为 PLY 文件。"""
    ok = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
        write_ascii=False,
        compressed=False,
        print_progress=True,
    )
    if not ok:
        raise MeshToPointCloudError(f"点云写入失败: {output_path}")


def count_gt_points(sequence: str, data_base: Path) -> int | None:
    """从 GT PLY 头部读取点数，用于日志对比（可选）。"""
    gt_path = data_base / sequence / "gt" / "gt_pd.ply"
    if not gt_path.exists():
        return None
    try:
        with gt_path.open("rb") as f:
            header = f.read(4096).decode("ascii", errors="ignore")
        for line in header.splitlines():
            if line.startswith("element vertex"):
                return int(line.split()[-1])
    except (OSError, ValueError):
        return None
    return None


def convert_mesh_to_pointcloud(
    input_path: Path,
    output_path: Path,
    num_points: int,
    init_factor: float,
    sequence: str | None = None,
    data_base: Path = DATA_BASE,
) -> None:
    """执行单次 mesh -> point cloud 转换。"""
    print(f"[读取] {input_path}")
    mesh = load_triangle_mesh(input_path)
    print(
        f"       顶点: {len(mesh.vertices):,}  |  "
        f"三角面: {len(mesh.triangles):,}"
    )

    print(f"[采样] 目标点数: {num_points:,}  (init_factor={init_factor})")
    pcd, method = sample_mesh_surface(mesh, num_points, init_factor)
    actual = len(pcd.points)
    print(f"       方法: {method}  |  实际点数: {actual:,}")

    if sequence is not None:
        gt_n = count_gt_points(sequence, data_base)
        if gt_n is not None:
            print(f"       参考 GT ({sequence}/gt/gt_pd.ply): {gt_n:,} 点")

    print(f"[保存] {output_path}")
    save_point_cloud(pcd, output_path)
    print("[完成]")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Open3D 将 PLY 三角网格采样为点云。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "示例:\n"
            "  python mesh_to_pointcloud.py \\\n"
            "      --input output/Sequence_01/mesh/tsdf_fusion_post.ply \\\n"
            "      --output output/Sequence_01/sampled_point_cloud.ply \\\n"
            "      --num_points 2500000\n"
            "\n"
            "  python mesh_to_pointcloud.py --sequence Sequence_01\n"
            "  python mesh_to_pointcloud.py --all-sequences"
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sequence",
        choices=SEQUENCES,
        help="按序列名自动使用 output/<seq>/mesh/tsdf_fusion_post.ply",
    )
    mode.add_argument(
        "--all-sequences",
        action="store_true",
        help="批量处理全部 5 个序列",
    )

    parser.add_argument(
        "--input",
        "-i",
        help="输入 PLY 三角网格路径（与 --sequence/--all-sequences 互斥时必填）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出 PLY 点云路径（--sequence 模式下默认 output/<seq>/sampled_point_cloud.ply）",
    )
    parser.add_argument(
        "--num_points",
        "-n",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="表面采样点数",
    )
    parser.add_argument(
        "--init_factor",
        type=float,
        default=DEFAULT_INIT_FACTOR,
        help="Poisson Disk 初始均匀采样倍数（init_factor × num_points）",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=OUTPUT_BASE,
        help="重建结果根目录（含 Sequence_XX/mesh/）",
    )
    parser.add_argument(
        "--data_base",
        type=Path,
        default=DATA_BASE,
        help="数据集根目录（含 Sequence_XX/gt/）",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    jobs: list[tuple[Path, Path, str | None]] = []

    if args.all_sequences:
        for seq in SEQUENCES:
            inp, default_out = sequence_paths(seq, args.output_base)
            out = default_out if args.output is None else resolve_output_path(args.output)
            jobs.append((inp, out, seq))
    elif args.sequence:
        inp, default_out = sequence_paths(args.sequence, args.output_base)
        out = default_out if args.output is None else resolve_output_path(args.output)
        jobs.append((inp, out, args.sequence))
    else:
        if not args.input:
            parser.error("请指定 --input，或使用 --sequence / --all-sequences")
        output = resolve_output_path(args.output or "sampled_point_cloud.ply")
        jobs.append((Path(args.input).expanduser(), output, None))

    print(f"Open3D 版本: {o3d.__version__}")
    failed = 0

    for input_path, output_path, sequence in jobs:
        print("\n" + "=" * 72)
        if sequence:
            print(f"序列: {sequence}")
        try:
            if sequence is None:
                input_path = resolve_input_path(str(input_path))
            elif not input_path.exists():
                raise MeshToPointCloudError(f"输入文件不存在: {input_path}")
            convert_mesh_to_pointcloud(
                input_path=input_path,
                output_path=output_path,
                num_points=args.num_points,
                init_factor=args.init_factor,
                sequence=sequence,
                data_base=args.data_base,
            )
        except MeshToPointCloudError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            failed += 1
        except Exception as exc:
            print(f"[错误] 未预期的异常: {exc}", file=sys.stderr)
            failed += 1
        except KeyboardInterrupt:
            print("\n[中断] 用户取消。", file=sys.stderr)
            return 130

    if len(jobs) > 1:
        print("\n" + "=" * 72)
        print(f"批量完成: 成功 {len(jobs) - failed}/{len(jobs)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
