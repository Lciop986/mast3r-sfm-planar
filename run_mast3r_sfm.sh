#!/bin/bash
# ============================================================
# MASt3R-SFM 替代 COLMAP 的完整流程
# 
# 用法: bash run_mast3r_sfm.sh <sequence_name>
# 示例: bash run_mast3r_sfm.sh Sequence_04
# ============================================================

set -e

SEQ=${1:?"Usage: $0 <Sequence_XX>"}
DATA_DIR="/home/srt_2025/liumohan/PlanarGS/PlanarGS/indoor_dataset_phone/data"
MAST3R_DIR="/home/srt_2025/liumohan/mast3r-sfm/mast3r-sfm"

# ============================================================
# 模型下载说明:
#   由于服务器无法访问 HuggingFace, 需要在有网络的机器上下载模型:
#
#   pip install huggingface_hub
#   huggingface-cli download naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
#       --local-dir ./mast3r_checkpoint/
#
#   然后将 ./mast3r_checkpoint/ 文件夹传输到:
#   /home/srt_2025/liumohan/mast3r-sfm/mast3r-sfm/checkpoints/
#
#   下载后的文件应包含:
#   - model.safetensors (约 2.5GB)
#   - config.json
#   - 其他配置文件
# ============================================================

MODEL_DIR="${MAST3R_DIR}/checkpoints"
MODEL_PATH="${MODEL_DIR}/model.safetensors"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ 找到本地模型: ${MODEL_PATH}"
    MODEL_ARG="$MODEL_DIR"
elif [ -f "${MODEL_DIR}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.safetensors" ]; then
    MODEL_PATH="${MODEL_DIR}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.safetensors"
    echo "✅ 找到本地模型: ${MODEL_PATH}"
    MODEL_ARG="$MODEL_DIR"
else
    echo "❌ 未找到 MASt3R 模型!"
    echo ""
    echo "请先在有网络的机器上下载模型:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \\"
    echo "      --local-dir ./mast3r_checkpoint/"
    echo ""
    echo "然后将文件夹传输到: ${MODEL_DIR}"
    echo ""
    echo "或者尝试直接从 HuggingFace 加载 (需要网络):"
    echo "  修改本脚本中的 MODEL_ARG 变量"
    exit 1
fi

IMAGE_DIR="${DATA_DIR}/${SEQ}/images"
OUTPUT_DIR="${DATA_DIR}/${SEQ}/mast3r_output"

# 检查镜像目录
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 图像目录不存在: $IMAGE_DIR"
    exit 1
fi

IMAGE_COUNT=$(ls "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)
echo "📸 找到 ${IMAGE_COUNT} 张图片"
echo "📂 输出目录: ${OUTPUT_DIR}"

# 激活环境
source /home/srt_2025/miniconda3/etc/profile.d/conda.sh
conda activate mast3r
export PYTHONPATH="${MAST3R_DIR}:$PYTHONPATH"

# 清理旧输出
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 清理旧输出..."
    rm -rf "$OUTPUT_DIR"
fi

# 备份旧的 COLMAP sparse (如果存在)
OLD_SPARSE="${DATA_DIR}/${SEQ}/sparse"
if [ -d "$OLD_SPARSE" ]; then
    BACKUP_DIR="${DATA_DIR}/${SEQ}/sparse_colmap_backup"
    echo "💾 备份旧 COLMAP 结果到 ${BACKUP_DIR}..."
    rm -rf "$BACKUP_DIR"
    mv "$OLD_SPARSE" "$BACKUP_DIR"
fi

# 运行 MASt3R-SFM
echo "🚀 启动 MASt3R-SFM 重建..."
echo "   图片: ${IMAGE_DIR}"
echo "   输出: ${OUTPUT_DIR}"

cd "$MAST3R_DIR"

python colmap_from_mast3r_ds.py \
    --image_dir "$IMAGE_DIR" \
    --save_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_ARG" \
    --device cuda \
    --image_size 512 \
    --shared_intrinsics \
    --matching_conf_thr 5.0 \
    --lr1 0.07 --niter1 500 \
    --lr2 0.014 --niter2 200

# 将 MASt3R 输出链接/复制到 COLMAP 标准位置
echo "📁 链接 MASt3R 结果到 COLMAP 标准路径..."
mkdir -p "${DATA_DIR}/${SEQ}/sparse"
cp -r "${OUTPUT_DIR}/sparse/0/"* "${DATA_DIR}/${SEQ}/sparse/"

echo "✅ 完成! COLMAP 兼容的稀疏模型已保存到 ${DATA_DIR}/${SEQ}/sparse/"
echo ""
echo "下一步: 运行 PlanarGS 几何先验提取"
echo "  cd /home/srt_2025/liumohan/PlanarGS/PlanarGS"
echo "  conda activate liumohan_planargs"
echo "  CUDA_VISIBLE_DEVICES=0 python run_geomprior.py -s ${DATA_DIR}/${SEQ} --group_size 25"
