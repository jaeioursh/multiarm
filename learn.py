
#from sim import armsim as sim

import torch
print(torch.cuda.is_available())

import gymnasium as gym

from rlwrapper import multiarm as sim


from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.utils.test_utils import (
	add_rllib_example_script_args,
	run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from pettingzoo.sisl import waterworld_v4


def trial1():
	parser = add_rllib_example_script_args(
		default_iters=200,
		default_timesteps=1000000,
		default_reward=0.0,
	)
	args = parser.parse_args()
	args.num_agents=3
	args.enable_new_api_stack=True
	args.no_tune=True
	args.num_env_runners=0
	args.algo == "PPO"
	assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
	assert (
		args.enable_new_api_stack
	), "Must set --enable-new-api-stack when running this script!"

	# Here, we use the "Agent Environment Cycle" (AEC) PettingZoo environment type.
	# For a "Parallel" environment example, see the rock paper scissors examples
	# in this same repository folder.
	if 0:
		register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))

		# Policies are called just like the agents (exact 1:1 mapping).
		policies = {f"pursuer_{i}" for i in range(args.num_agents)}

	else:
		register_env("env", lambda _: sim())

		# Policies are called just like the agents (exact 1:1 mapping).
		policies = {f"arm_{i}" for i in range(args.num_agents)}

	base_config = (
		get_trainable_cls(args.algo)
		.get_default_config()
		.environment("env")
		.env_runners(
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
		.multi_agent(
			policies=policies,
			# Exact 1:1 mapping from AgentID to ModuleID.
			policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
		)
		.training(
			vf_loss_coeff=0.005,
		)
		.rl_module(
			rl_module_spec=MultiRLModuleSpec(
				rl_module_specs={p: RLModuleSpec() for p in policies},
			),
			model_config=DefaultModelConfig(vf_share_layers=False),
		)
	)

	run_rllib_example_script_experiment(base_config, args)

def trial2():
	register_env("env", lambda _: sim())

		# Policies are called just like the agents (exact 1:1 mapping).
	obs_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(41,))
	act_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(10,))
	policies = {f"arm_{i}":(None, obs_space, act_space, {}) for i in range(3)}
	new_version=False
	config = (
		PPOConfig()
        .api_stack(
            enable_env_runner_and_connector_v2=new_version,
            enable_rl_module_and_learner=new_version,
        )
		.framework("torch")
		.environment("env")
		#.env_runners(
        #    env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        #)
		.learners(
            # How many Learner workers do we need? If you have more than 1 GPU,
            # set this parameter to the number of GPUs available.
            num_learners=1,
            # How many GPUs does each Learner need? If you have more than 1 GPU or only
            # one Learner, you should set this to 1, otherwise, set this to some
            # fraction.
            num_gpus_per_learner=1,
        )
		.resources(num_gpus=0)
		.multi_agent(
			policies=policies,
			# Exact 1:1 mapping from AgentID to ModuleID.
			policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
		)
		.training(
			vf_loss_coeff=0.005,
		)
	)
	algorithm = config.build()
	for i in range(100):
		result = algorithm.train()
		print(i)
	print(result)

if __name__ == "__main__":
	trial2()