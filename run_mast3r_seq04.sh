#!/bin/bash
# ============================================================
# Sequence_04 专用 MASt3R-SFM 脚本
# 固定相机内参: 738x994, fx=809.30, fy=807.44, cx=368.01, cy=503.09
# ============================================================
set -e

# --- 路径配置 ---
SEQ_DIR="/home/srt_2025/liumohan/PlanarGS/PlanarGS/indoor_dataset_phone/data/Sequence_04"
IMAGE_DIR="${SEQ_DIR}/images"
MAST3R_DIR="/home/srt_2025/liumohan/mast3r-sfm/mast3r-sfm"
MODEL_DIR="${MAST3R_DIR}/checkpoints"
OUTPUT_DIR="${SEQ_DIR}/mast3r_output"

# --- 检查图片 ---
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 图片目录不存在: $IMAGE_DIR"
    exit 1
fi
echo "📸 图片: ${IMAGE_DIR} ($(ls "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l) 张)"

# --- 下载模型 (如未下载) ---
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "⬇️  模型未下载，正在从 HuggingFace 下载..."
    echo "    (约 2.5GB, 可能需要几分钟)"

    source /home/srt_2025/miniconda3/etc/profile.d/conda.sh
    conda activate mast3r

    hf download naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
        --local-dir "$MODEL_DIR" 2>&1

    if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
        echo "❌ 模型下载失败! 请手动下载:"
        echo "   hf download naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --local-dir ${MODEL_DIR}"
        exit 1
    fi
    echo "✅ 模型下载完成"
else
    echo "✅ 模型已存在: ${MODEL_DIR}"
fi

# --- 激活环境 ---
source /home/srt_2025/miniconda3/etc/profile.d/conda.sh
conda activate mast3r
export PYTHONPATH="${MAST3R_DIR}:$PYTHONPATH"

# --- 清理旧输出 ---
rm -rf "$OUTPUT_DIR"

# --- 备份旧 COLMAP ---
OLD_SPARSE="${SEQ_DIR}/sparse"
if [ -d "$OLD_SPARSE" ]; then
    BACKUP="${SEQ_DIR}/sparse_colmap_backup_$(date +%Y%m%d_%H%M%S)"
    echo "💾 备份旧 COLMAP → ${BACKUP}"
    mv "$OLD_SPARSE" "$BACKUP"
fi

# --- 运行 MASt3R-SFM (固定内参) ---
echo "🚀 启动 MASt3R-SFM (shared_intrinsics, 图片尺寸 512)..."
cd "$MAST3R_DIR"

python colmap_from_mast3r_ds.py \
    --image_dir "$IMAGE_DIR" \
    --save_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_DIR" \
    --device cuda \
    --image_size 512 \
    --shared_intrinsics \
    --matching_conf_thr 5.0 \
    --lr1 0.07 --niter1 500 \
    --lr2 0.014 --niter2 200 \
    --min_conf_thr 1.5

# --- 将结果复制到 COLMAP 标准路径 ---
echo "📁 部署 MASt3R 结果到 COLMAP 标准路径..."
SPARSE_OUT="${SEQ_DIR}/sparse"
mkdir -p "$SPARSE_OUT"

# MASt3R 输出的是 TXT 格式, 需要转换为 BIN 格式供 PlanarGS 使用
# 先检查输出路径
RECON_DIR="${OUTPUT_DIR}/sparse/0"
if [ -d "$RECON_DIR" ]; then
    # 直接用 TXT 格式 (PlanarGS 支持 TXT)
    cp "$RECON_DIR"/*.txt "$SPARSE_OUT/" 2>/dev/null || true
    cp "$RECON_DIR"/*.ply "$SPARSE_OUT/" 2>/dev/null || true
    echo "✅ COLMAP TXT 文件已复制到: ${SPARSE_OUT}"
    ls -la "$SPARSE_OUT"
else
    echo "❌ 未找到重建结果: ${RECON_DIR}"
    exit 1
fi

echo ""
echo "============================================"
echo "✅ Sequence_04 MASt3R-SFM 完成!"
echo "   COLMAP 文件: ${SPARSE_OUT}"
echo ""
echo "下一步 (PlanarGS 训练):"
echo "  conda activate liumohan_planargs"
echo "  python run_geomprior.py -s ${SEQ_DIR} --group_size 24"
echo "  python run_lp3.py --source_path ${SEQ_DIR}"
echo "  CUDA_VISIBLE_DEVICES=X python train.py -s ${SEQ_DIR} -m output/Sequence_04 -r 2"
echo "============================================"
