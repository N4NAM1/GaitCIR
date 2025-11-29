META_PATH="../CCPG_RGB_JSON/meta_ccpg_static.json"
TEMPLATE_PATH="../CCPG_RGB_JSON/templates_instruction.json"
OUTPUT="../CCPG_RGB_JSON/CCPG/ccpg_cir_final.json"

# ================= 运行 =================
echo "🚀 Start generate CIR_dataset for CCPG..."


python 03_build_triplets.py \
    --meta_path "$META_PATH" \
    --template_path "$TEMPLATE_PATH" \
    --output "$OUTPUT" \

echo "🎉 All Done!"