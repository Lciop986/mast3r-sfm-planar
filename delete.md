# PlanarGS 项目 Python 文件使用分析报告

> 生成日期: 2026-06-29
> 分析范围: seq1 ~ seq5 训练/渲染流程中所有项目自有 Python 文件

---

## 🟢 核心流程文件 (已使用 - 请保留)

这些文件是训练/渲染流程的直接或间接依赖，**不能删除**。

### 主入口脚本
| 文件 | 用途 | 被谁调用 |
|------|------|----------|
| `train.py` | 训练主入口 | 直接运行 |
| `render.py` | 渲染 + TSDF 网格提取 | 直接运行 |
| `run_lp3.py` | LP3 平面先验生成 | 直接运行 |
| `run_geomprior.py` | 几何先验生成 (DUSt3R) | 直接运行 |

### arguments/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `arguments/__init__.py` | 参数解析 (ModelParams, PipelineParams, OptimizationParams, PriorParams) | train.py, render.py, run_lp3.py, run_geomprior.py, eval_recon.py 等 |

### scene/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `scene/__init__.py` | Scene 类 | train.py, render.py |
| `scene/cameras.py` | Camera 类, LoadGeomprior | scene/__init__.py, run_lp3.py, common_utils/camera_utils.py |
| `scene/colmap_loader.py` | COLMAP 二进制/文本读取, Camera dataclass | scene/dataset_readers.py, eval_preprocess.py, lp3/box_migrate.py |
| `scene/dataset_readers.py` | readColmapSceneInfo 等 | scene/__init__.py, run_lp3.py, run_geomprior.py, planar/cull_mesh.py |
| `scene/ply_loader.py` | SceneInfo, CameraInfo, BasicPointCloud | scene/__init__.py, scene/dataset_readers.py, geomprior/regreader_utils.py |
| `scene/gaussian_model.py` | GaussianModel 类 | scene/__init__.py, gaussian_renderer/__init__.py |

### gaussian_renderer/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `gaussian_renderer/__init__.py` | render(), GaussianModel | train.py, render.py |

### common_utils/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `common_utils/__init__.py` | (空文件, package marker) | — |
| `common_utils/camera_utils.py` | cameraList_from_camInfos, loadCam | scene/__init__.py |
| `common_utils/general_utils.py` | safe_state, PILtoTorch 等 | train.py, render.py, scene/cameras.py |
| `common_utils/graphics_utils.py` | get_k, Depth2Pointscam, fov2focal 等 | scene/cameras.py, scene/dataset_readers.py, planar/co_planar.py, lp3/box_migrate.py, lp3/color_cluster.py, geomprior/regreader_utils.py 等 |
| `common_utils/loss_utils.py` | l1_loss, l2_loss, ssim, psnr | train.py, metrics.py, planar/training_report.py, geomprior/align_opt.py |
| `common_utils/sh_utils.py` | eval_sh, RGB2SH | gaussian_renderer/__init__.py, scene/gaussian_model.py |

### planar/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `planar/__init__.py` | (空文件, package marker) | — |
| `planar/visualize.py` | visualDepth, visualNorm, visualSegmask 等 | render.py, run_lp3.py, planar/training_report.py, geomprior/regreader_utils.py, lp3/box_migrate.py, lp3/mask_refine.py |
| `planar/co_planar.py` | co_planar (共面约束) | train.py |
| `planar/training_report.py` | prepare_output_and_logger, training_report | train.py |
| `planar/densify_points.py` | InitialPlaneSeg, SegPoints, PlaneMaskGS 等 | planar/co_planar.py |
| `planar/cull_mesh.py` | cull_mesh, mask_mesh | eval_recon.py, eval_preprocess.py |

### geomprior/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `geomprior/run_dust3r.py` | DUSt3R (深度估计) | run_geomprior.py |
| `geomprior/dataloader.py` | GroupAlign, SaveDepthInfo | run_geomprior.py |
| `geomprior/align_opt.py` | OptimizeGroupDepth | geomprior/regreader_utils.py |
| `geomprior/regreader_utils.py` | DepthInfo, AlignGroupDepth, LoadGroupDepth | geomprior/dataloader.py |

### lp3/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `lp3/run_groundedsam.py` | GroundingDINO, SAM | run_lp3.py |
| `lp3/box_migrate.py` | LP3Cam, AddPreviosBox, FilterMask | run_lp3.py |
| `lp3/mask_refine.py` | BoxSmaller, NormalSplit | run_lp3.py |
| `lp3/color_cluster.py` | MaskDistance, SplitPic, kmeans_torch 等 | run_lp3.py, lp3/mask_refine.py |

### lpipsPyTorch/
| 文件 | 用途 | 被谁导入 |
|------|------|----------|
| `lpipsPyTorch/__init__.py` | lpips 指标 | metrics.py |

---

## 🟡 工具/评估脚本 (非核心流程但可能有用)

这些脚本**不在训练/渲染主流程中**，作为独立工具使用。根据实际需要决定是否保留。

| 文件 | 用途 | 备注 |
|------|------|------|
| `metrics.py` | 计算 PSNR/SSIM/LPIPS (2D指标) | 用于测试集渲染图评估 |
| `viewer.py` | Open3D 重建结果查看器 | 可视化网格/点云 |

---

## 🔴 可删除文件 (确认未使用)

### 1. 已废弃的 COLMAP 脚本
| 文件 | 原因 |
|------|------|
| `2_run_colmap_intrinsic_only.py` | COLMAP 纯内参运行脚本。现在使用 MASt3R-SFM 替代 COLMAP，此脚本**不再需要**。依赖 `utils/database` 模块 (如果该模块也存在可一并清理)。 |

### 2. 未被任何代码导入的模块
| 文件 | 原因 |
|------|------|
| `gaussian_renderer/network_gui.py` | 网络 GUI 模块。搜索全项目，**没有任何文件导入它**。是 3DGS 原始项目的遗留代码。 |

### 3. 冗余评估脚本 (功能重复，建议保留一个)

以下 3 个脚本功能高度重叠——都是对 TSDF 重建网格做 3D 精度评估 (vs `gt_pd.ply`)，仅对齐算法不同：

| 文件 | 对齐策略 | 建议 |
|------|----------|------|
| `race_eval.py` | 多阶段带尺度 ICP | **保留此文件** (最新、最完善) |
| `hyl_eval.py` | PCA 粗配准 + 多阶段带尺度 ICP | 可删除 (与 race_eval.py 功能重复) |
| `gdc_eval.py` | FPFH+RANSAC 粗配准 + ICP | 可删除 (与 race_eval.py 功能重复) |

### 4. 原始数据格式评估脚本 (不再适用)
| 文件 | 原因 |
|------|------|
| `eval_recon.py` | 原始 PlanarGS 3D 评估脚本。依赖 `mesh.ply` 作为真值 + `align_params.npz` 预计算对齐参数。当前竞赛数据集使用 `gt/gt_pd.ply` 格式，**不再适用**。 |
| `eval_preprocess.py` | 评估预处理脚本，为 `eval_recon.py` 生成 `align_params.npz`。同样依赖旧的 `mesh.ply` 格式，**不再适用**。 |

### 5. 独立工具 (按需决定)
| 文件 | 原因 |
|------|------|
| `mesh_to_pointcloud.py` | 网格→点云采样工具。独立运行，非流程必需。如不需要将 TSDF 网格转为点云格式可删除。 |
| `report-xyj/parse_cc_report.py` | CloudCompare 报告解析工具。独立运行，仅在生成技术报告时使用，非训练/渲染流程必需。 |

---

## 📊 汇总

| 类别 | 数量 | 文件 |
|------|------|------|
| ✅ 核心依赖 | ~32 | 上方 🟢 所有文件 |
| 🟡 独立工具 | 2 | metrics.py, viewer.py |
| 🔴 建议删除 | 9 | `2_run_colmap_intrinsic_only.py`, `gaussian_renderer/network_gui.py`, `hyl_eval.py`, `gdc_eval.py`, `eval_recon.py`, `eval_preprocess.py`, `mesh_to_pointcloud.py` (可选), `report-xyj/parse_cc_report.py` (可选) |
| 🔴 冗余3选1 | 2/3 | `race_eval.py`(保留), `hyl_eval.py`(删), `gdc_eval.py`(删) |

**最终建议删除 8 个文件** (保留 `race_eval.py`、`mesh_to_pointcloud.py` 和 `report-xyj/parse_cc_report.py`)：
1. `2_run_colmap_intrinsic_only.py`
2. `gaussian_renderer/network_gui.py`
3. `hyl_eval.py`
4. `gdc_eval.py`
5. `eval_recon.py`
6. `eval_preprocess.py`
7. (可选) `mesh_to_pointcloud.py`
8. (可选) `report-xyj/parse_cc_report.py`
