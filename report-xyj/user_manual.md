# 程序使用说明

本项目中的数据处理操作不完全由脚本执行（中间数据使用了CloudCompare软件来处理，以剔除ground truth要求范围以外的重建点对其的干扰）。

按要求“演示程序及使用说明：评委可使用该程序对测试数据集进行重建或结果复现。”，故不在此处提供“一键生成”式的演示程序，仅复述技术报告中所有脚本的执行顺序以及数据处理操作流程。

## 一、运行方式

### 1.安装必要依赖库：

```bash
conda create -n planargs python=3.10
conda activate planargs
pip install cmake==3.20.*

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118  #replace your cuda version

pip install -r requirements.txt 

pip install -e submodules/simple-knn --no-build-isolation 
pip install -e submodules/pytorch3d --no-build-isolation   
pip install submodules/diff-plane-rasterization --no-build-isolation
```

安装GroundedSAM：

```bash
cd submodules 
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git 
mv Grounded-Segment-Anything groundedsam

cd groundedsam && pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO 
&& cd ../..
```

### 2.数据预处理与重建流程：

1. 进入数据目录，运行 COLMAP 流程生成 sparse 与可选 dense 结果
（对于sequence_04，我们使用mast3r-sfm做这一步）;
2. 使用 run_geomprior.py 生成几何先验；
3. 使用 run_lp3.py 生成平面先验；
4. 运行 train.py 完成重建；
5. 运行 render.py 生成网格与渲染结果。

对应命令如下：

```bash
# COLMAP 预处理
cd < your_data_directory >
bash run_colmap.sh
# 对于 Sequence_04 使用 mast3r-sfm
# 在根目录下，运行run_mast3r_seq04.sh
# bash run_mast3r_seq04.sh 

# 生成几何先验
python run_geomprior.py -s <data_path> --group_size 40

# 生成平面先验
python run_lp3.py -s <data_path> -t "wall. floor. door. screen. window. ceiling. table"

# 训练重建
python train.py -s <data_path> -m <output_path> --eval

# 渲染并生成 mesh
python render.py -m <output_path> --voxel_size 0.02 --max_depth 100.0 --eval
```
若未预先下载必要的模型，并且下载失败，可添加如下环境变量：
```text
HF_ENDPOINT=https://hf-mirror.com
```

若生成几何先验时显存不足，可以适当减小 group_size

最终可得到以下结果：

- 重建点云：point_cloud.ply；
- 渲染结果：颜色图、深度图、法向图（见<output_path>/train，<output_path>/train）；
- 三角网格：mesh/tsdf_fusion_post.ply；

### 3.精度评估流程：
1. 在 CloudCompare 中人工剔除重建网格中超出参考真值点云有效空间范围的区域，例如窗户外侧、墙外悬空部分，以减少无关外部几何对误差统计的干扰；
2. 将处理后的重建网格采样为点云（使用 mesh_to_pointcloud.py），并使采样点数量与参考真值点云 gt/gt_pd.ply 的点数保持一致，然后将二者导入 CloudCompare；
3. 完成粗对齐与精对齐，使重建模型与真值点云处于同一坐标系与尺度，随后使用 Cloud-to-Cloud Distance 功能计算重建点云到参考点云的距离；
4. 将计算结果导出为文本文件，文件中包含距离列（列名为 C2C_absolute_distances）；
5. 运行脚本 parse_cc_report.py：
```bash
python parse_cc_report.py -i <your_directory_of_"cloudcompare_export.txt">
```

其中，输入文件为 CloudCompare 成功导出的距离结果文本文件，表头中包含 C2C_absolute_distances 列。

使用以下指标描述重建质量：

- 平均距离误差（Mean Error）；
- 中位数距离误差（Median Error）；
- 均方根误差（RMSE）；
- 90% / 95% 分位误差；
- 20 cm 内点比例；
- 10 cm 内点比例。
