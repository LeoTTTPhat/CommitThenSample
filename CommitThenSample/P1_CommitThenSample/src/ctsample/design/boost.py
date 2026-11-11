"""Follow-the-evidence boost sets (DESIGN §3.5 Alg. 1 line 8, §3.10: kappa_b = 5, W_b = 20).

A device with a retained hit in epoch e' is boosted for epochs e'+2 ... e'+1+W_b: its utility is
multiplied by kappa_b. (e'+1 is skipped because boost set_{e+2..} is derived from epoch e's retained
frames after its selection.) Hence device d is boosted at epoch e iff its latest retained-hit epoch
that is <= e-2 is >= e-1-W_b. The boost makes pi depend on past Z (adaptive design, assumption (P)).

``BoostState`` is vectorised over a leading reps shape.
"""
from __future__ import annotations

import numpy as np

__all__ = ["BoostState", "boost_set_from_hits"]

_NEVER = np.iinfo(np.int64).min // 4


class BoostState:
    def __init__(self, n_devices: int, shape=(), kappa_b: float = 5.0, W_b: int = 20):
        self.n_devices = int(n_devices)
        self.shape = tuple(shape)
        self.kappa_b = float(kappa_b)
        self.W_b = int(W_b)
        self.last = np.full(self.shape + (self.n_devices,), _NEVER, dtype=np.int64)   # latest hit <= e-1
        self.lag = np.full(self.shape + (self.n_devices,), _NEVER, dtype=np.int64)    # latest hit <= e-2
        self.epoch = None

    def begin_epoch(self, e: int) -> None:
        """Advance to epoch e. Call once per epoch, in order, before multiplier(); every epoch must
        be closed with end_epoch()/record_units() (possibly with no hits) so that the lag shifts."""
        if self.epoch is not None and e != self.epoch + 1:
            raise ValueError("BoostState epochs must be consecutive")
        self.epoch = e

    def boosted(self):
        """Boolean (shape + (n_devices,)) boost set of the current epoch."""
        e = self.epoch
        return self.lag >= e - 1 - self.W_b

    def multiplier(self, device_idx):
        """kappa_b where the unit's device is boosted, else 1; device_idx: (n,) -> shape + (n,)."""
        b = self.boosted()[..., np.asarray(device_idx)]
        return np.where(b, self.kappa_b, 1.0)

    def end_epoch(self, hit_dev_mask) -> None:
        """Record the current epoch's retained hits. hit_dev_mask: bool shape + (n_devices,)."""
        e = self.epoch
        self.lag = self.last.copy()                      # latest hit <= e-1 (valid for epoch e+1)
        self.last = np.where(np.asarray(hit_dev_mask, dtype=bool), e, self.last)

    def record_units(self, Z_hits, trigger, hit_device) -> None:
        """Convenience: Z_hits (shape + (n_hits,)) retained flags of the epoch's hit units,
        trigger (n_hits,) whether a retained unit triggers the detectors, hit_device (n_hits,)."""
        Zt = np.asarray(Z_hits, dtype=bool) & np.asarray(trigger, dtype=bool)
        mask = np.zeros(self.shape + (self.n_devices,), dtype=bool)
        if Zt.size:
            idx = np.nonzero(Zt)
            dev = np.asarray(hit_device)[idx[-1]]
            mask[idx[:-1] + (dev,)] = True
        self.end_epoch(mask)


def boost_set_from_hits(hit_epochs_by_device: dict, e: int, W_b: int = 20) -> set:
    """Reference (slow) definition for tests: devices with a retained hit e' with e'+2 <= e <= e'+1+W_b."""
    return {d for d, eps in hit_epochs_by_device.items() if any(ep + 2 <= e <= ep + 1 + W_b for ep in eps)}
