
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=11198 \
    driver.py \
    --do_predict true \
    --use_lora \
    --peft_model_id ./peft_lora_adapter \
    --data_path ./alpaca_data_rand1k.json \
    --per_device_eval_batch_size 2 \
    --model_max_length 512 \
    --logging_steps 1 \
    --fp16 \
    --output_dir ./tmp \
    --overwrite_output_dir
