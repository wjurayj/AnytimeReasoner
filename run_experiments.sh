# GRPO baseline
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=1 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=v2only


################################## Main results ##################################
## AnytimeReasoner-linear
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=linear actor_rollout_ref.rollout.variance_reduction=brpo
## AnytimeReasoner-uniform
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=uniform actor_rollout_ref.rollout.variance_reduction=brpo
## AnytimeReasoner-base
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo


################################## Ablations ##################################
## GRPO+linear
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=linear actor_rollout_ref.rollout.variance_reduction=v2only
## GRPO+decouple
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=v2only
## GRPO+vr
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=1 actor_rollout_ref.rollout.summary_method=grpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo
## GRPO+vr+decouple
./anytime_reasoner/scripts/train/run_1.5b_8k.sh --model $MODEL_PATH actor_rollout_ref.rollout.n_summary=4 actor_rollout_ref.rollout.summary_method=brpo actor_rollout_ref.rollout.n_budget_support=4 actor_rollout_ref.rollout.budget_probs=base actor_rollout_ref.rollout.variance_reduction=brpo
