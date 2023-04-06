# 🦙🌲🤏 Alpaca-Hf-Easier

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using both native finetuning and [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

It is developed purely using Huggingface Transformers Trainer api, which allows deepspeed, apex or fsdp under distributed setting, thus making our life easier:)

### Local Setup

1. Install dependencies

   ```bash
    conda create -n torch200 python=3.10
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    git submodule add https://github.com/huggingface/transformers ./transformers; cd ./transformers; git checkout 68d640f7c368bcaaaecfc678f11908ebbd3d6176; pip install -e . 
    git submodule add https://github.com/huggingface/peft.git ./peft; cd ./peft; pip install -e .
    pip install sentencepiece
    conda install datasets
    pip install openai
    pip install wandb 
	conda activate torch200
   ```

### Training & Generation (`driver.py`)

Both training and generation are encapsulated in `driver.py` which you may launch directly via torchrunn or deepspeed or whatever hf supports in both single-server or distributed-servers setting.

BTW, it support native fine-tuning or LoRA for your flexible purpose.


Example usage for LoRA Tuning:

```bash
torchrun --nproc_per_node=2 --master_port=11198 \
    driver.py \
    --do_train true \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./alpaca_data.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --model_max_length 512 \
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
    --output_dir ./peft_lora_adapter_official \
    --deepspeed ./ds_config_zero1.json \
    --overwrite_output_dir
```

Example usage for LoRA Generation:

```bash
torchrun --nproc_per_node=1 --master_port=11198 \
    driver.py \
    --do_predict true \
    --use_lora \
    --peft_model_id ./peft_lora_adapter_official \
    --data_path ./alpaca_data_rand1k.json \
    --per_device_eval_batch_size 2 \
    --model_max_length 512 \
    --logging_steps 1 \
    --fp16 \
    --output_dir ./tmp \
    --overwrite_output_dir
```

Example usage for native Finetuning Generation:

```bash
torchrun --nproc_per_node=1 --master_port=11198 \
    driver.py \
    --do_predict true \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./alpaca_data_rand1k.json \
    --per_device_eval_batch_size 2 \
    --model_max_length 512 \
    --max_new_tokens 100 \
    --logging_steps 1 \
    --fp16 \
    --output_dir ./tmp/inference_wo_lora \
    --overwrite_output_dir
```
