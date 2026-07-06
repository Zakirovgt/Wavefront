from wavefront.training.deeponet import (
    apply_net,
    apply_net_task,
    apply_net_tasks,
    rel_l2_batch_loss,
    step,
)
from wavefront.training.fno import (
    apply_fno,
    rel_l2_field_loss,
    step_fno,
)
from wavefront.training.deeponet_losses import (
    loss_fn,
    loss_ics,
    loss_res,
)
from wavefront.training.fno_losses import (
    fno_rel_l2,
    loss_fno,
    loss_res_fno,
)
from wavefront.training.precision import set_mixed_precision
from wavefront.training.deeponet_trainer import main_routine_deeponet
from wavefront.training.fno_trainer import main_routine_fno
from wavefront.training.runner import main_routine