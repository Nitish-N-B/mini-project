"""
trust_module.py
===============
EMA-based per-sender trust scoring.

Specs (from PPT slide 14):
  error    = || recv_pos - true_pos ||
  accuracy = max(0, 1 - error / max_error)
  tau      = 0.1 * accuracy + 0.9 * tau      (on receive)
  tau      = 0.95 * tau                        (on drop)
  tau      ∈ [0, 1]                            (bounded always)

  max_error = AdversarialChannel.MAX_SPOOF_ERROR = 5.0

Convergence target: EMA scores for adversarial senders fall below 0.3
within 200 steps (PPT Expected Outcomes).
"""

import numpy as np
from typing import Optional


class TrustModule:
    """
    Maintains a per-sender EMA trust score for one agent.

    Each agent in the system has its OWN TrustModule tracking
    the trustworthiness of messages received from all other agents.

    Parameters
    ----------
    n_senders   : int   — number of other agents sending messages
    max_error   : float — normalisation constant for position error
    alpha       : float — EMA learning rate for accuracy signal (0.1 per PPT)
    decay_on_drop: float — multiplicative decay when message is dropped (0.95 per PPT)
    init_trust  : float — starting trust score for all senders (1.0 = full trust)
    """

    EMA_ALPHA      = 0.1    # PPT: tau = 0.1 * accuracy + 0.9 * tau
    DECAY_ON_DROP  = 0.95   # PPT: tau = 0.95 * tau on drop
    MAX_ERROR      = 5.0    # from AdversarialChannel.MAX_SPOOF_ERROR

    def __init__(
        self,
        n_senders: int,
        max_error: float = MAX_ERROR,
        alpha: float = EMA_ALPHA,
        decay_on_drop: float = DECAY_ON_DROP,
        init_trust: float = 1.0,
    ):
        assert 0 < alpha < 1,        "alpha must be in (0, 1)"
        assert 0 < decay_on_drop < 1,"decay_on_drop must be in (0, 1)"

        self.n_senders     = n_senders
        self.max_error     = max_error
        self.alpha         = alpha
        self.decay_on_drop = decay_on_drop

        # tau[j] = trust score for sender j ∈ [0, 1]
        self.tau = np.full(n_senders, init_trust, dtype=np.float64)

        # diagnostics
        self._update_count = 0

    # ── public API ───────────────────────────────────────────────────────────

    def update(
        self,
        received_pos: np.ndarray,   # shape (n_senders, 3) — x,y,z from channel
        true_pos: np.ndarray,       # shape (3,)            — ground truth
        dropped_mask: np.ndarray,   # shape (n_senders,)    — bool, True = dropped
    ):
        """
        Update trust scores for all senders after one communication round.

        Parameters
        ----------
        received_pos  : x,y,z portion of received messages (first 3 dims)
        true_pos      : ground-truth target position from env (3D)
        dropped_mask  : True where channel dropped the packet
        """
       # NEW
        assert received_pos.ndim == 2 and received_pos.shape[0] == self.n_senders, \
        f"received_pos shape mismatch: {received_pos.shape}"
        assert true_pos.ndim == 1, f"true_pos shape mismatch: {true_pos.shape}"
        assert dropped_mask.shape == (self.n_senders,), "dropped_mask shape mismatch"

        for j in range(self.n_senders):
            if dropped_mask[j]:
                # ── packet dropped: decay trust ──────────────────────────
                self.tau[j] *= self.decay_on_drop
            else:
                # ── received: compute accuracy and EMA update ────────────
                error    = float(np.linalg.norm(received_pos[j] - true_pos))
                accuracy = max(0.0, 1.0 - error / self.max_error)
                # EMA: tau = alpha * accuracy + (1-alpha) * tau
                self.tau[j] = self.alpha * accuracy + (1.0 - self.alpha) * self.tau[j]

            # Clamp to [0, 1]
            self.tau[j] = float(np.clip(self.tau[j], 0.0, 1.0))

        self._update_count += 1

    def get_trust_scores(self) -> np.ndarray:
        """Return current trust scores, shape (n_senders,)."""
        return self.tau.copy()

    def reset(self, init_trust: float = 1.0):
        """Reset trust scores (call on env.reset())."""
        self.tau[:] = init_trust
        self._update_count = 0

    def get_stats(self) -> dict:
        return {
            "trust_scores":  self.tau.tolist(),
            "mean_trust":    float(np.mean(self.tau)),
            "min_trust":     float(np.min(self.tau)),
            "update_count":  self._update_count,
        }
