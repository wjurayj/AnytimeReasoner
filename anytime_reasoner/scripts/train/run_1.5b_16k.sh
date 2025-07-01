#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# export VLLM_ATTENTION_BACKEND=XFORMERS

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
fi

echo "${@:1}"

# Train over a single node, 8 A100-80GB GPUs.
RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/train.parquet \
    data.val_files=[$HOME/deepscaler/data/aime.parquet,$HOME/deepscaler/data/amc.parquet] \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=18176 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.attn_implementation=flex_attention \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_summary=1 \
    actor_rollout_ref.rollout.max_gen_len=16000 \
    actor_rollout_ref.rollout.n_budget_support=1 \
    actor_rollout_ref.rollout.budget_probs="base" \
    actor_rollout_ref.rollout.variance_reduction="brpo" \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='anytime-reasoning' \
    trainer.experiment_name='16k-train' \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.val_generations_to_log_to_wandb=10 \
    trainer.total_epochs=1 "${@:1}"
