
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=11198 \
    driver.py \
    --do_predict true \
    --model_name_or_path /home/zhengyang/workspace2/LLaMA/llama-7b-hf \
    --data_path ./showcases.json \
    --model_max_length 512 \
    --max_new_tokens 128 \
    --logging_steps 1 \
    --fp16 \
    --output_dir ./tmp/llama-7b-hf \
    --overwrite_output_dir
