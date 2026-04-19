from typing import List, Sequence

import numpy as np


class LengthStrategy:
    """Base class for per-batch sequence-length selection strategies.

    Subclasses implement `next_length(rng)` which is called once per batch by
    `DynamicLengthBatchSampler` to decide the sequence length for that batch.
    """

    def __init__(self, lengths: Sequence[int]):
        if len(lengths) == 0:
            raise ValueError("lengths must be non-empty")
        self.lengths: List[int] = [int(length) for length in lengths]

    def next_length(self, rng: np.random.Generator) -> int:
        raise NotImplementedError


class UniformRandomLengthStrategy(LengthStrategy):
    """Pick a length uniformly at random from the configured list per batch."""

    def next_length(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.lengths))
