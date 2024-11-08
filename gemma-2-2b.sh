export CUDA_VISIBLE_DEVICES=1 && \
python finetune.py \
    --base_model 'google/gemma-2-2b' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'