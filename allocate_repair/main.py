"""Entry point: ``python -m allocate_repair.main [hydra overrides]``.

Mirrors ``verl.trainer.main_ppo_v0.TaskRunner`` but registers the allocate-repair actor worker and trainer.
"""

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@ray.remote
class AllocateRepairTaskRunner(BaseTaskRunner):
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from allocate_repair.workers import AllocateRepairActorRolloutRefWorker

        role = Role.ActorRolloutRef if need_reference_policy(config) else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(AllocateRepairActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        return AllocateRepairActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        wandb_cfg = config.allocate_repair.wandb
        os.environ["WANDB_MODE"] = str(wandb_cfg.mode)
        if wandb_cfg.dir:
            os.makedirs(wandb_cfg.dir, exist_ok=True)
            os.environ["WANDB_DIR"] = str(wandb_cfg.dir)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        _, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_reward_model_resource_pool(config)
        validate_config(
            config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config)
        )

        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.config import HFModelConfig

        from allocate_repair.trainer import AllocateRepairTrainer

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_config.tokenizer, model_config.processor
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        trainer = AllocateRepairTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="allocate_repair_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=AllocateRepairTaskRunner)


if __name__ == "__main__":
    main()
