from wavefront.training.deeponet_trainer import main_routine_deeponet
from wavefront.training.fno_trainer import main_routine_fno


def main_routine(
        args,
        grad_sensor_data=None,
        wavefront_true_data=None,
        grad_grid_data=None,
):
    """Train the selected operator using the supplied data."""
    if args.operator_type == "deeponet":
        return main_routine_deeponet(
            args=args,
            grad_sensor_data=grad_sensor_data,
            wavefront_true_data=wavefront_true_data,
        )

    if args.operator_type == "fno":
        return main_routine_fno(
            args=args,
            grad_sensor_data=grad_sensor_data,
            wavefront_true_data=wavefront_true_data,
            grad_grid_data=grad_grid_data,
        )

    raise ValueError(f"Unknown operator_type={args.operator_type}")
