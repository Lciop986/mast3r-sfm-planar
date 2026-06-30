#!/bin/bash
# ============================================================
# hyl_train_all.sh
# GDC 2026 Sequence_02/04/05 完整重训练脚本
# 
# 流程: COLMAP(PINHOLE+固定内参) → geomprior → planarprior → train → render → eval
# 环境: conda activate liumohan_planargs
# ============================================================
set -e  # 任何步骤失败即停止

# ==================== 配置 ====================
CONDA_ENV="liumohan_planargs"
PROJ_DIR="/home/srt_2025/liumohan/PlanarGS/PlanarGS"
DATA_DIR="$PROJ_DIR/indoor_dataset_phone/data"
OUTPUT_DIR="$PROJ_DIR/output"
GPU_ID=0
SEQS=("Sequence_02" "Sequence_04" "Sequence_05")

# PINHOLE 相机固定内参（同款手机，所有序列共用）
CAM_PARAMS="809.3032856727216,807.4371289882238,368.0097263362558,503.0891797364038"

# DUSt3R 几何先验
GROUP_SIZE=25

# GroundedSAM 平面先验
PROMPTS="wall. floor. door. screen. window. ceiling. table"

# 训练参数
TRAIN_ITER=30000

# 渲染参数
VOXEL_SIZE=0.03
MAX_DEPTH=10.0

# ==================== 初始化 ====================
source /home/srt_2025/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$PROJ_DIR"

echo "============================================================"
echo "  hyl_train_all — 批量重训练 Sequence_02/04/05"
echo "  环境: $CONDA_ENV  |  GPU: $GPU_ID  |  相机: PINHOLE"
echo "  内参: fx=${CAM_PARAMS%%,*} fy=$(echo $CAM_PARAMS | cut -d, -f2)"
echo "============================================================"

# ==================== 逐序列处理 ====================
for SEQ in "${SEQS[@]}"; do
    echo ""
    echo "################################################################"
    echo "###  开始处理: $SEQ"
    echo "################################################################"

    SEQ_DATA="$DATA_DIR/$SEQ"
    SEQ_OUT="$OUTPUT_DIR/$SEQ"
    IMG_DIR="$SEQ_DATA/images"
    SP_DIR="$SEQ_DATA/sparse"

    # -------------------- 清理旧数据 --------------------
    echo "[$SEQ] 1/7 清理旧输出..."
    rm -rf "$SEQ_OUT"
    rm -rf "$SEQ_DATA/geomprior"
    rm -rf "$SEQ_DATA/planarprior"
    rm -rf "$SP_DIR"
    rm -f "$SEQ_DATA/database.db"
    mkdir -p "$SP_DIR"

    # -------------------- COLMAP --------------------
    echo "[$SEQ] 2/7 COLMAP: 特征提取 (PINHOLE + 固定内参)..."
    colmap feature_extractor \
        --database_path "$SEQ_DATA/database.db" \
        --image_path "$IMG_DIR" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.camera_params "$CAM_PARAMS" \
        2>&1 | grep -E "Elapsed|Focal|ERROR" | tail -2

    echo "[$SEQ] 2/7 COLMAP: 穷举特征匹配..."
    colmap exhaustive_matcher \
        --database_path "$SEQ_DATA/database.db" \
        2>&1 | grep -E "Elapsed|ERROR" | tail -2

    echo "[$SEQ] 2/7 COLMAP: 稀疏重建 (固定内参不优化)..."
    mkdir -p "$SP_DIR/0"
    colmap mapper \
        --database_path "$SEQ_DATA/database.db" \
        --image_path "$IMG_DIR" \
        --output_path "$SP_DIR/0" \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.ba_refine_extra_params 0 \
        2>&1 | grep -E "initial|Register|Images|points|Elapsed|ERROR"

    if [ ! -f "$SP_DIR/0/cameras.bin" ]; then
        echo "❌ [$SEQ] COLMAP 稀疏重建失败！跳过"
        rm -f "$SEQ_DATA/database.db"
        continue
    fi
    mv "$SP_DIR/0/"* "$SP_DIR/"
    rm -rf "$SP_DIR/0"
    rm -f "$SEQ_DATA/database.db"
    echo "✅ [$SEQ] COLMAP 完成: $(ls $SP_DIR | tr '\n' ' ')"

    # -------------------- 几何先验 (DUSt3R) --------------------
    echo "[$SEQ] 3/7 DUSt3R 几何先验 (depth估计, group_size=$GROUP_SIZE)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_geomprior.py \
        -s "$SEQ_DATA" --group_size $GROUP_SIZE 2>&1 | tail -3
    echo "✅ [$SEQ] geomprior 完成"

    # -------------------- 平面先验 (GroundedSAM) --------------------
    echo "[$SEQ] 4/7 GroundedSAM 平面先验..."
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU_ID \
        python run_lp3.py -s "$SEQ_DATA" -t "$PROMPTS" 2>&1 | tail -3
    echo "✅ [$SEQ] planarprior 完成"

    # -------------------- 3DGS 训练 --------------------
    echo "[$SEQ] 5/7 3DGS 训练 ($TRAIN_ITER iter)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        -s "$SEQ_DATA" -m "$SEQ_OUT" --eval 2>&1 | tail -5
    echo "✅ [$SEQ] train 完成"

    # -------------------- TSDF 渲染 --------------------
    echo "[$SEQ] 6/7 渲染 + TSDF 网格 (voxel=$VOXEL_SIZE, max_depth=$MAX_DEPTH)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
        -m "$SEQ_OUT" --voxel_size $VOXEL_SIZE --max_depth $MAX_DEPTH --eval 2>&1 | tail -3
    echo "✅ [$SEQ] render 完成"

    # -------------------- 评测 --------------------
    echo "[$SEQ] 7/7 几何精度评测..."
    python hyl_eval.py --seq "${SEQ##*_}" 2>&1 | grep -E "序列|平均|${SEQ}"
    echo "✅ [$SEQ] eval 完成"

    echo "🎉 [$SEQ] 全流程完成！"
done

echo ""
echo "============================================================"
echo "  全部序列处理完毕！"
echo "  汇总报告: $OUTPUT_DIR/hyl_eval_summary.json"
echo "============================================================"
