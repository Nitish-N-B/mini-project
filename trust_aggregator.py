"""
trust_aggregator.py
===================
Trust-weighted aggregation of teammate messages.

Specs (from PPT slide 14):
  agg = sum(tau_j * msg_j) / sum(tau_j)

  - Reduces to simple mean when all trust scores are equal
  - Zero overhead in benign (zero-drop) conditions
  - Falls back to zero vector if all senders have tau=0
"""

import numpy as np


class TrustAggregator:
    """
    Computes trust-weighted average of incoming messages for one agent.

    The aggregated message populates dims [6:12] of each drone's
    20-dim observation: [own_pos(3), own_vel(3), agg_pos(3), agg_vel(3), ...]

    Parameters
    ----------
    n_senders  : int — how many other agents send messages to this agent
    msg_dim    : int — dimensionality of each message (6: x,y,z,vx,vy,vz)
    """

    def __init__(self, n_senders: int, msg_dim: int = 6):
        self.n_senders = n_senders
        self.msg_dim   = msg_dim

    def aggregate(
        self,
        messages: np.ndarray,       # shape (n_senders, msg_dim)
        trust_scores: np.ndarray,   # shape (n_senders,)
        dropped_mask: np.ndarray,   # shape (n_senders,) — dropped msgs excluded
    ) -> np.ndarray:
        """
        Compute trust-weighted aggregation.

        Dropped messages (zero vectors) are excluded from the weighted sum
        so they don't pull the aggregated estimate toward zero.

        Parameters
        ----------
        messages     : received (potentially spoofed) messages after channel
        trust_scores : current EMA trust scores from TrustModule
        dropped_mask : True where message was dropped → exclude from aggregation

        Returns
        -------
        agg : np.ndarray, shape (msg_dim,)
            Trust-weighted average of non-dropped messages.
            Returns zeros if ALL messages were dropped.
        """
        assert messages.shape     == (self.n_senders, self.msg_dim)
        assert trust_scores.shape == (self.n_senders,)
        assert dropped_mask.shape == (self.n_senders,)

        # Mask out dropped messages (their content is meaningless)
        valid = ~dropped_mask                           # shape (n_senders,)
        valid_messages = messages[valid]               # shape (k, msg_dim)
        valid_scores   = trust_scores[valid]           # shape (k,)

        if valid_scores.sum() < 1e-9:
            # All dropped or all trust=0 → return zero vector (no information)
            return np.zeros(self.msg_dim, dtype=np.float32)

        # Weighted sum: sum(tau_j * msg_j) / sum(tau_j)
        weights = valid_scores / valid_scores.sum()    # normalised, shape (k,)
        agg     = (weights[:, np.newaxis] * valid_messages).sum(axis=0)

        return agg.astype(np.float32)

    def aggregate_all_agents(
        self,
        messages: np.ndarray,       # shape (N, msg_dim) — all agents' messages
        trust_modules: list,        # list of N TrustModule instances
        dropped_masks: np.ndarray,  # shape (N, N) — dropped_masks[i,j] = True if i didn't get j's msg
    ) -> np.ndarray:
        """
        Compute aggregated messages for ALL agents in one call.

        For agent i, aggregate messages from all j ≠ i senders.

        Parameters
        ----------
        messages      : honest messages before channel (env.get_hunter_messages())
                        — channel is applied externally per agent pair
        trust_modules : list of TrustModule, one per agent
        dropped_masks : shape (N, N); dropped_masks[receiver, sender]

        Returns
        -------
        agg_all : np.ndarray, shape (N, msg_dim)
            Aggregated message for each agent.
        """
        N   = len(trust_modules)
        agg = np.zeros((N, self.msg_dim), dtype=np.float32)

        for i in range(N):
            # indices of senders (everyone except self)
            sender_idx = [j for j in range(N) if j != i]
            msgs_recv  = messages[sender_idx]              # shape (N-1, msg_dim)
            scores     = trust_modules[i].get_trust_scores()  # shape (N-1,)
            dropped    = dropped_masks[i][sender_idx]      # shape (N-1,)
            agg[i]     = self.aggregate(msgs_recv, scores, dropped)

        return agg  # shape (N, msg_dim)
