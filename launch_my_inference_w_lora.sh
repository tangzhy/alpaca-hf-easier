
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=11198 \
    driver.py \
    --do_predict true \
    --use_lora \
    --peft_model_id ./peft_lora_adapter_official \
    --data_path ./showcases.json \
    --per_device_eval_batch_size 1 \
    --model_max_length 512 \
    --max_new_tokens 128 \
    --logging_steps 1 \
    --fp16 \
    --output_dir ./tmp/peft_lora_adapter_official \
    --overwrite_output_dir
