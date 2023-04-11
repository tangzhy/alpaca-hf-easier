
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=3468 \
    driver4msmarco.py \
    --do_train true \
    --model_name_or_path /home/zhengyang/workspace2/LLaMA/llama-7b-hf \
    --data_path ./train_v2.1.converted.wfansweronly.wprompt.rand500.json \
    --num_train_epochs 100000000000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --model_max_length 2048 \
    --optim adamw_torch \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --logging_steps 1 \
    --fp16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --output_dir ./peft_lora_adapter_msmarco \
    --deepspeed ./ds_config_zero1.json \
    --overwrite_output_dir
