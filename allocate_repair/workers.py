"""verl's ActorRolloutRefWorker with the phase-weighted policy loss installed after model init."""

from functools import partial

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

from allocate_repair.losses import phase_weighted_ppo_loss


class AllocateRepairActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self.actor is None:
            return
        assert not self.distillation_enabled, "allocate-repair does not support on-policy distillation"
        # Reuse the exact ActorConfig verl built for its own ppo_loss.
        actor_config = self.loss_fn.keywords["config"]
        self.loss_fn = partial(phase_weighted_ppo_loss, config=actor_config)
        self.actor.set_loss_fn(self.loss_fn)
