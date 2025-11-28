DATA_ROOT="/root/autodl-tmp/CCPG_Processed"
OUTPUT="../CCPG_RGB_JSON/meta_ccpg_static.json"

# ================= 运行 =================
echo "🚀 Start generate meat for CCPG..."


python 02_generate_ccpg_meta.py \
    --data_root "$DATA_ROOT" \
    --output "$OUTPUT" \

echo "🎉 All Done!"