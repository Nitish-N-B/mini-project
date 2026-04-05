"""
adversarial_channel.py
======================
Simulates an adversarial communication channel between agents.

Specs (from PPT slides 12 & 14):
  - p_drop  : Bernoulli packet drop probability {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
  - p_spoof : Probability of coordinate spoofing (additive Gaussian noise)
              p_spoof = 0.1 during evaluation, 0.0 during standard training
  - p_drop  = 0.2 during adversarial/trust-aware training (dropout regularization)
  - On drop : receiver gets zeros (no message) — handled by TrustModule decay
  - On spoof: additive noise ~ N(0, spoof_std) added to [x, y, vx, vy]
"""

import numpy as np
from typing import Optional, Tuple


class AdversarialChannel:
    """
    Processes outgoing messages and returns (possibly corrupted) received messages.

    Each call to `transmit` handles ONE sender → ALL receivers broadcast.
    The channel is applied independently per message per step.

    Parameters
    ----------
    p_drop    : float  — probability a message is completely dropped [0, 1]
    p_spoof   : float  — probability a received message is spoofed [0, 1]
    spoof_std : float  — std dev of Gaussian spoofing noise (in grid units)
    seed      : int    — for reproducibility
    """

    MAX_SPOOF_ERROR = 5.0   # used in TrustModule to normalize position error

    def __init__(
        self,
        p_drop: float = 0.0,
        p_spoof: float = 0.0,
        spoof_std: float = 2.0,
        seed: Optional[int] = None,
    ):
        assert 0.0 <= p_drop  <= 1.0, "p_drop must be in [0, 1]"
        assert 0.0 <= p_spoof <= 1.0, "p_spoof must be in [0, 1]"

        self.p_drop    = p_drop
        self.p_spoof   = p_spoof
        self.spoof_std = spoof_std
        self.rng       = np.random.default_rng(seed)

        # stats for monitoring
        self._total_messages = 0
        self._total_drops    = 0
        self._total_spoofs   = 0

    def transmit(
        self, messages: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pass messages through the adversarial channel.

        Parameters
        ----------
        messages : np.ndarray, shape (N, msg_dim)
            N agents each broadcasting [x, y, z, vx, vy, vz] of the target.

        Returns
        -------
        received : np.ndarray, shape (N, msg_dim)
            Messages after drop/spoof. Dropped messages → zero vector.
        dropped_mask : np.ndarray, shape (N,), dtype bool
            True where message was dropped (for TrustModule decay).
        """
        N = messages.shape[0]
        received     = messages.copy().astype(np.float64)
        dropped_mask = np.zeros(N, dtype=bool)

        for i in range(N):
            self._total_messages += 1

            # ── packet drop (Bernoulli) ──────────────────────────────────
            if self.rng.random() < self.p_drop:
                received[i]     = 0.0           # zero vector = no info
                dropped_mask[i] = True
                self._total_drops += 1
                continue                         # skip spoofing if dropped

            # ── coordinate spoofing (additive Gaussian noise) ────────────
            if self.rng.random() < self.p_spoof:
                noise = self.rng.normal(0.0, self.spoof_std, size=received[i].shape)
                received[i] += noise
                self._total_spoofs += 1

        return received.astype(np.float32), dropped_mask

    def set_drop_rate(self, p_drop: float):
        """Dynamically change drop rate (used when sweeping eval conditions)."""
        assert 0.0 <= p_drop <= 1.0
        self.p_drop = p_drop

    def set_spoof_rate(self, p_spoof: float):
        """Dynamically change spoof rate."""
        assert 0.0 <= p_spoof <= 1.0
        self.p_spoof = p_spoof

    def get_stats(self) -> dict:
        """Return channel-level statistics."""
        total = max(self._total_messages, 1)
        return {
            "total_messages":  self._total_messages,
            "total_drops":     self._total_drops,
            "total_spoofs":    self._total_spoofs,
            "empirical_drop_rate":  self._total_drops  / total,
            "empirical_spoof_rate": self._total_spoofs / total,
        }

    def reset_stats(self):
        """Reset counters (call between episodes if needed)."""
        self._total_messages = 0
        self._total_drops    = 0
        self._total_spoofs   = 0
