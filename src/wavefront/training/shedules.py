from __future__ import annotations

import optax


def make_lr_schedule(
        lr: float,
        steps: int,
        scheduler: str = "constant",
        warmup_steps: int | None = None,
        warmup_frac: float = 0.05,
        end_lr_factor: float = 0.01,
):
    """
    Create a learning-rate schedule for Optax optimizers.

    Args:
        lr: Peak learning rate. For a constant schedule, this is the learning
            rate used at every optimization step.
        steps: Total number of planned optimization steps.
        scheduler: Schedule type. Supported values are:
            - "constant": Use a fixed learning rate equal to lr.
            - "warmup_cosine": Linearly warm up from zero to lr, then decay
              smoothly following a cosine schedule.
        warmup_steps: Number of warmup steps for the "warmup_cosine" schedule.
            When None, the value is computed as warmup_frac * steps, with a
            minimum of one step.
        warmup_frac: Fraction of total training steps allocated to linear
            warmup when warmup_steps is None.
        end_lr_factor: Final learning-rate multiplier for the
            "warmup_cosine" schedule. The final value is:

                lr * end_lr_factor

    Returns:
        An Optax learning-rate schedule callable.

    Raises:
        ValueError: If scheduler is not "constant" or "warmup_cosine".

    Notes:
        For "warmup_cosine", the learning rate follows this pattern:

            0 -> lr during warmup
            lr -> lr * end_lr_factor during cosine decay

        The decay duration is always at least one step longer than the warmup
        period, preventing an invalid schedule when warmup is close to or
        exceeds the requested total number of steps.
    """
    lr = float(lr)
    steps = int(steps)

    if scheduler == "constant":
        # Keep the learning rate fixed throughout training.
        return optax.constant_schedule(lr)

    elif scheduler == "warmup_cosine":
        # Derive warmup duration from the requested fraction when an explicit
        # number of warmup steps is not provided.
        if warmup_steps is None:
            warmup_steps = max(
                1,
                int(warmup_frac * steps),
            )

        warmup_steps = int(warmup_steps)

        # Ensure the cosine-decay schedule extends beyond the warmup phase.
        decay_steps = max(
            steps,
            warmup_steps + 1,
        )

        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=lr * float(end_lr_factor),
        )

    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler!r}. "
            "Use 'constant' or 'warmup_cosine'."
        )
