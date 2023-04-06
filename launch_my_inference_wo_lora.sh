
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=11198 \
    driver.py \
    --do_predict true \
    --model_name_or_path /home/zhengyang/workspace2/LLaMA/llama-7b-hf \
    --data_path ./alpaca_data_rand1k.json \
    --per_device_eval_batch_size 2 \
    --model_max_length 512 \
    --max_new_tokens 100 \
    --logging_steps 1 \
    --fp16 \
    --output_dir ./tmp/inference_wo_lora \
    --overwrite_output_dir
