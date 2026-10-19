BASE_ROOT=.

# 手动修改数据集路径
IMAGE_ROOT=F:\MAR-REID\reference\Baseline\data\CUHK-PEDES\imgs
JSON_ROOT=F:\MAR-REID\reference\Baseline\data\CUHK-PEDES\reid_raw.json
OUT_ROOT=$BASE_ROOT/cuhkpedes/processed_data


echo "Process CUHK-PEDES dataset and save it as pickle form"

python ${BASE_ROOT}/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3
