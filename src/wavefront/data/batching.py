from functools import partial

import jax


class DataGenerator:
    """
    Pointwise batch generator for DeepONet-style training.

    Stored arrays:
        u: Input function representations with shape
            (N_tasks, branch_dim).

        y: Query coordinates with shape
            (N_tasks, P, 2), where P is the number of spatial points per task.

        s: Target values with shape
            (N_tasks, P, out_dim).

    Each generated batch samples tasks and spatial points independently and
    returns:

        u_batch:
            Shape (B, branch_dim).

        y_batch:
            Shape (B, 2).

        s_batch:
            Shape (B, out_dim).

    Here, B is batch_size.
    """

    def __init__(
            self,
            u,
            y,
            s,
            batch_size,
            gen_key,
    ):
        """
        Initialize the pointwise data generator.

        Args:
            u: Branch-network inputs with shape (N_tasks, branch_dim).
            y: Coordinate queries with shape (N_tasks, P, 2).
            s: Target values with shape (N_tasks, P, out_dim).
            batch_size: Number of independently sampled task-point pairs per
                generated batch.
            gen_key: JAX PRNG key used for random sampling.
        """
        self.u = u
        self.y = y
        self.s = s

        self.N_tasks = u.shape[0]
        self.P = y.shape[1]

        self.batch_size = batch_size
        self.key = gen_key

    def __iter__(self):
        """
        Return the generator itself so it can be used in iteration contexts.
        """
        return self

    def __next__(self):
        """
        Generate one randomly sampled pointwise batch.

        Returns:
            A tuple:

                inputs:
                    (u_batch, y_batch)

                targets:
                    s_batch
        """
        # Split the persistent random key before sampling a new batch.
        self.key, subkey = jax.random.split(self.key)

        return self.__data_generation(subkey)

    @partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, key_i):
        """
        Sample random task indices and point indices, then gather the matching
        branch inputs, coordinates, and targets.

        Args:
            key_i: JAX PRNG key for this batch-generation call.

        Returns:
            A tuple:

                (u, y):
                    u has shape (B, branch_dim).
                    y has shape (B, 2).

                s:
                    Target tensor with shape (B, out_dim).

        Notes:
            Task and point indices are sampled independently with replacement.
            This means that the same task or task-point pair may appear more
            than once in a batch.
        """
        # Use separate random streams for task and spatial-point selection.
        key_task, key_point = jax.random.split(key_i)

        # Sample task indices from [0, N_tasks).
        task_idx = jax.random.randint(
            key_task,
            (self.batch_size,),
            0,
            self.N_tasks,
        )

        # Sample point indices from [0, P).
        point_idx = jax.random.randint(
            key_point,
            (self.batch_size,),
            0,
            self.P,
        )

        # Gather the function representation associated with each sampled task.
        u = self.u[task_idx]

        # Gather the selected coordinate from each sampled task.
        y = self.y[task_idx, point_idx]

        # Gather the corresponding target field value.
        s = self.s[task_idx, point_idx]

        return (u, y), s


class GridDataGenerator:
    """
    Random mini-batch generator for grid-based models such as FNO.

    Stored arrays:
        X: Model inputs with shape (N, nx, ny, C).
        Y: Target fields with shape (N, nx, ny), or more generally
            (N, nx, ny, out_channels).

    Each generated batch samples complete grid examples and returns:

        X_batch:
            Shape (B, nx, ny, C).

        Y_batch:
            Shape (B, nx, ny), or (B, nx, ny, out_channels).
    """

    def __init__(
            self,
            X,
            Y,
            batch_size,
            key,
    ):
        """
        Initialize the grid-based batch generator.

        Args:
            X: Input tensor with shape (N, nx, ny, C).
            Y: Target tensor with shape (N, nx, ny), or optionally
                (N, nx, ny, out_channels).
            batch_size: Number of complete grid samples per batch.
            key: JAX PRNG key used for random sample selection.
        """
        self.X = X
        self.Y = Y

        self.batch_size = batch_size
        self.N = X.shape[0]

        self.key = key

    def __iter__(self):
        """
        Return the generator itself so it can be used in iteration contexts.
        """
        return self

    def __next__(self):
        """
        Generate one randomly sampled grid batch.

        Returns:
            A tuple:

                X_batch:
                    Input tensor with shape (B, nx, ny, C).

                Y_batch:
                    Target tensor with shape (B, nx, ny), or
                    (B, nx, ny, out_channels).

        Notes:
            Examples are sampled independently with replacement, so a training
            example may appear multiple times in the same batch.
        """
        # Split the persistent random key before sampling batch indices.
        self.key, subkey = jax.random.split(self.key)

        # Sample full grid examples from the dataset.
        idx = jax.random.randint(
            subkey,
            shape=(self.batch_size,),
            minval=0,
            maxval=self.N,
        )

        return self.X[idx], self.Y[idx]
