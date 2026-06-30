#!/bin/bash

# ==================== 【用户配置中心】 ====================
# 🎯 在这里指定你本次想要处理的场景，用空格隔开。
# 示例 1（只想单跑4）：TARGET_SEQS=("Sequence_04")
# 示例 2（想同时重试2和5）：TARGET_SEQS=("Sequence_02" "Sequence_05")
# 示例 3（全量重跑三个）：TARGET_SEQS=("Sequence_04" "Sequence_02" "Sequence_05")
TARGET_SEQS=("Sequence_04""Sequence_02" "Sequence_05")

# ==========================================================

for seq in "${TARGET_SEQS[@]}"; do
    if [ -d "$seq" ]; then
        echo "=================================================="
        echo "🚀 [全能重跑引擎] 正在处理指定场景: $seq"
        echo "=================================================="

        # 定义核心路径
        IMAGE_PATH="$seq/images"
        SPARSE_PATH="$seq/sparse"
        DB_PATH="$seq/database.db"
        DENSE_PATH="$seq/dense"  # 稠密深度工作区

        # 彻底清除历史失败残留，保证全新纯净环境
        echo "🧹 正在清理场景 $seq 的所有历史残留数据(含稀疏/稠密)..."
        rm -rf "$SPARSE_PATH"
        rm -rf "$DENSE_PATH"
        rm -f "$DB_PATH"

        # 重新创建基础目录
        mkdir -p "$SPARSE_PATH"

        # ------------------------------------------------
        # 📸 步骤 1: 特征提取 (根据场景智能选择内参策略)
        # ------------------------------------------------
        # Sequence_02/04/05 来自同一手机同一镜头，共享完全相同的标定内参
        if [ "$seq" == "Sequence_02" ] || [ "$seq" == "Sequence_04" ] || [ "$seq" == "Sequence_05" ]; then
            echo "💡 检测到 $seq：正在手动注入标定的精确 PINHOLE 内参（共享同款手机镜头参数）..."
            colmap feature_extractor \
                --database_path "$DB_PATH" \
                --image_path "$IMAGE_PATH" \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model PINHOLE \
                --ImageReader.camera_params "809.3032856727216,807.4371289882238,368.0097263362558,503.0891797364038"
        else
            echo "💡 检测到 $seq：使用标准的共享 PINHOLE 自适应模式..."
            colmap feature_extractor \
                --database_path "$DB_PATH" \
                --image_path "$IMAGE_PATH" \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model PINHOLE
        fi

        # ------------------------------------------------
        # 🔗 步骤 2: 特征匹配
        # ------------------------------------------------
        echo "🔗 步骤 2: 进行穷举特征匹配..."
        colmap exhaustive_matcher \
            --database_path "$DB_PATH"

        # ------------------------------------------------
        # 🏗️ 步骤 3: 稀疏重建
        # ------------------------------------------------
        echo "🏗️ 步骤 3: 启动增量式 SfM 稀疏重建..."
        mkdir -p "$SPARSE_PATH/0"
        colmap mapper \
            --database_path "$DB_PATH" \
            --image_path "$IMAGE_PATH" \
            --output_path "$SPARSE_PATH/0"

        # 整理稀疏文件使之符合规范
        if [ -f "$SPARSE_PATH/0/cameras.bin" ]; then
            mv "$SPARSE_PATH/0/"* "$SPARSE_PATH/"
            rm -rf "$SPARSE_PATH/0"
            echo "🎉 $seq 稀疏重建完成！"

            # ------------------------------------------------
            # 🔥 核心升级：针对 Sequence_02 和 Sequence_05 注入深度约束提取
            # ------------------------------------------------
            if [ "$seq" == "Sequence_02" ] || [ "$seq" == "Sequence_05" ]; then
                echo "⚡ [深度约束激活] 开始为 $seq 构建稠密深度图映射..."
                mkdir -p "$DENSE_PATH"

                echo "    -> [MVS 1/2] 正在进行图像畸变纠正与立体准备 (Image Undistorter)..."
                colmap image_undistorter \
                    --image_path "$IMAGE_PATH" \
                    --input_path "$SPARSE_PATH" \
                    --output_path "$DENSE_PATH" \
                    --output_type COLMAP

                echo "    -> [MVS 2/2] 启动 GPU Patch-Match 稠密光度匹配，正在疯狂榨干显卡..."
                echo "    ⚠️ 提示: 这一步正在密集计算深度估计，请耐心等待..."
                colmap patch_match_stereo \
                    --workspace_path "$DENSE_PATH" \
                    --StereoRASTER.max_image_size 2000

                echo "🎉 $seq 稠密深度约束数据收集完毕！已保存至 $DENSE_PATH/stereo/depth_maps"
            fi

            echo "✅ $seq 流程整体顺利收工！"
        else
            echo "❌ $seq 稀疏重建阶段不幸失败，请检查 images 图像重叠度。"
        fi

        # 清理临时无用 db 文件
        rm -f "$DB_PATH"
    fi
done
