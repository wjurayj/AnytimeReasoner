# GRPO baseline
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=1 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=v2only trainer.experiment_name=GRPO


################################## Main results ##################################
## AnytimeReasoner-linear
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=linear actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=AR-linear
## AnytimeReasoner-uniform
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=uniform actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=AR-uniform
## AnytimeReasoner-base
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=AR-base


################################## Ablations ##################################
## GRPO+linear
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=linear actor_rollout_ref.rollout.variance_reduction=v2only trainer.experiment_name=GRPO+linear
## GRPO+decouple
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=v2only trainer.experiment_name=GRPO+decouple
## GRPO+vr
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=GRPO+vr
## GRPO+vr+decouple
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=GRPO+vr+decouple



# 16k
## GRPO
./anytime_reasoner/scripts/train/run_1.5b_16k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=1 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=v2only trainer.experiment_name=16k-GRPO

## AR-uniform
./anytime_reasoner/scripts/train/run_1.5b_16k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=2 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=8 actor_rollout_ref.rollout.budget_probs=uniform actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=16k-AR-uniform
## AR-base
./anytime_reasoner/scripts/train/run_1.5b_16k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=2 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=8 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo trainer.experiment_name=16k-AR-base
