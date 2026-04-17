export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$1}

model_preset="${MODEL_PRESET:-qwen3_8b}"
rl_token_model_path="${RL_TOKEN_MODEL_PATH:-}"
rl_checkpoint_path="${RL_CHECKPOINT_PATH:-}"
data_dir="${DATA_DIR:-./eagle/data}"
dataset_train="${DATASET_TRAIN:-humaneval}"
save_path="${SAVE_PATH:-./checkpoints/${model_preset}/size}"
total_timesteps="${TOTAL_TIMESTEPS:-100000}"
batch_size="${BATCH_SIZE:-64}"
n_steps="${N_STEPS:-128}"
lr="${LR:-3e-4}"

python3 -m rl.rl_total \
    --model_preset ${model_preset} \
    --rl_token_model_path "${rl_token_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir ${data_dir} \
    --dataset_train ${dataset_train} \
    --save_path ${save_path} \
    --total_timesteps ${total_timesteps} \
    --batch_size ${batch_size} \
    --n_steps ${n_steps} \
    --lr ${lr} \
    --pi_arch 1024 256 \
    --vf_arch 1024 256
