#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Sequential modeling
===================

Sequence alignment
------------------
.. autosummary::
    :toctree: generated/

    dtw
    rqa

Viterbi decoding
----------------
.. autosummary::
    :toctree: generated/

    viterbi
    viterbi_discriminative
    viterbi_binary

Transition matrices
-------------------
.. autosummary::
    :toctree: generated/

    transition_uniform
    transition_loop
    transition_cycle
    transition_local
"""

import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from librosa.util import pad_center, fill_off_diagonal, tiny, expand_to
from librosa.util.exceptions import ParameterError
from librosa.filters import get_window
from librosa.util.decorators import deprecate_positional_args

__all__ = [
    "viterbi",
    "transition_loop",
    "transition_local",
]

# @jit(nopython=True, cache=True)
def _viterbi(log_prob, log_trans, log_p_init):  # pragma: no cover
    """Core Viterbi algorithm.

    This is intended for internal use only.

    Parameters
    ----------
    log_prob : np.ndarray [shape=(T, m)]
        ``log_prob[t, s]`` is the conditional log-likelihood
        ``log P[X = X(t) | State(t) = s]``
    log_trans : np.ndarray [shape=(m, m)]
        The log transition matrix
        ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
    log_p_init : np.ndarray [shape=(m,)]
        log of the initial state distribution

    Returns
    -------
    None
        All computations are performed in-place on ``state, value, ptr``.
    """
    n_steps, n_states = log_prob.shape

    state = np.zeros(n_steps, dtype=np.uint16, order='C')
    value = np.zeros((n_steps, n_states), dtype=np.float64, order='C')
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16, order='C')

    # factor in initial state distribution
    value[0] = log_prob[0] + log_p_init

    log_trans_t = log_trans.T
    log_trans_t = np.ascontiguousarray(log_trans_t, dtype=log_trans.dtype)
    log_trans_t = np.reshape(log_trans_t, log_trans_t.shape, order='C')

    for t in range(1, n_steps):
        # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
        #    assume at time t-1 we were in state k
        #    transition k -> j

        # Broadcast over rows:
        #    Tout[k, j] = V[t-1, k] * A[k, j]
        #    then take the max over columns
        # We'll do this in log-space for stability
        trans_out = value[t - 1, :] + log_trans_t
        
        # Unroll the max/argmax loop to enable numba support
        '''
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]
        '''
        ptr[t, :] = np.argmax(trans_out, axis=1)
        # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
        value[t, :] = log_prob[t, :] + np.max(trans_out, axis=1)

    # Now roll backward

    # Get the last state
    state[-1] = np.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    return state, logp

'''
import cupy as cp
def _viterbi_cuda(log_prob, log_trans, log_p_init):  # pragma: no cover
    # to GPU
    log_trans = cp.asarray(log_trans)
    log_prob = cp.asarray(log_prob)

    n_steps, n_states = log_prob.shape

    state = cp.zeros(n_steps, dtype=np.uint16)
    value = cp.zeros((n_steps, n_states), dtype=np.float32)
    ptr = cp.zeros((n_steps, n_states), dtype=np.uint16)

    # factor in initial state distribution
    value[0] = log_prob[0] + cp.asarray(log_p_init)

    log_trans_t = log_trans.T
    # log_trans_t = cp.ascontiguousarray(log_trans_t, dtype=log_trans.dtype)
    # log_trans_t = cp.reshape(log_trans_t, log_trans_t.shape, order='C')

    for t in range(1, n_steps):

        trans_out = value[t - 1, :] + log_trans_t
        
        # Unroll the max/argmax loop to enable numba support
        # for j in range(n_states):
        #     ptr[t, j] = np.argmax(trans_out[j])
        #     # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
        #     value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]
        ptr[t, :] = cp.argmax(trans_out, axis=1)
        # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
        value[t, :] = log_prob[t, :] + cp.max(trans_out, axis=1)

    # Now roll backward

    # Get the last state
    state[-1] = cp.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    state = cp.asnumpy(state)
    logp = cp.asnumpy(logp)
    return state, logp
'''

import torch
def _viterbi_torch(log_prob, log_trans, log_p_init, device="cpu"):  # pragma: no cover
    # to GPU
    log_trans = torch.tensor(log_trans, dtype=torch.float32, device=device)
    log_prob = torch.tensor(log_prob, dtype=torch.float32, device=device)

    n_steps, n_states = log_prob.shape

    state = torch.zeros(n_steps, dtype=torch.int16, device=device)
    value = torch.zeros((n_steps, n_states), dtype=torch.float64, device=device)
    ptr = torch.zeros((n_steps, n_states), dtype=torch.int16, device=device)

    # factor in initial state distribution
    value[0] = log_prob[0] + torch.tensor(log_p_init, dtype=torch.float32, device=device)

    log_trans_t = log_trans.T
    # log_trans_t = cp.ascontiguousarray(log_trans_t, dtype=log_trans.dtype)
    # log_trans_t = cp.reshape(log_trans_t, log_trans_t.shape, order='C')

    for t in range(1, n_steps):

        trans_out = value[t - 1, :] + log_trans_t
        
        # Unroll the max/argmax loop to enable numba support
        '''
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]
        '''
        values, indices = torch.max(trans_out, dim=1)
        ptr[t, :] = indices
        # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
        value[t, :] = log_prob[t, :] + values

    # Now roll backward

    # Get the last state
    state[-1] = torch.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    state = state.cpu().numpy()
    logp = logp.cpu().numpy()
    return state, logp

@deprecate_positional_args
def viterbi(prob, transition, *, p_init=None, return_logp=False, device="cpu"):
    """Viterbi decoding from observation likelihoods.

    Given a sequence of observation likelihoods ``prob[s, t]``,
    indicating the conditional likelihood of seeing the observation
    at time ``t`` from state ``s``, and a transition matrix
    ``transition[i, j]`` which encodes the conditional probability of
    moving from state ``i`` to state ``j``, the Viterbi algorithm [#]_ computes
    the most likely sequence of states from the observations.

    .. [#] Viterbi, Andrew. "Error bounds for convolutional codes and an
        asymptotically optimum decoding algorithm."
        IEEE transactions on Information Theory 13.2 (1967): 260-269.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_states, n_steps), non-negative]
        ``prob[..., s, t]`` is the probability of observation at time ``t``
        being generated by state ``s``.
    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        ``transition[i, j]`` is the probability of a transition from i->j.
        Each row must sum to 1.
    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, a uniform distribution is assumed.
    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_steps,)]
        The most likely state sequence.
        If ``prob`` contains multiple channels of input, then each channel is
        decoded independently.
    logp : scalar [float] or np.ndarray
        If ``return_logp=True``, the log probability of ``states`` given
        the observations.

    See Also
    --------
    viterbi_discriminative : Viterbi decoding from state likelihoods

    Examples
    --------
    Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    In this example, we have two states ``healthy`` and ``fever``, with
    initial probabilities 60% and 40%.

    We have three observation possibilities: ``normal``, ``cold``, and
    ``dizzy``, whose probabilities given each state are:

    ``healthy => {normal: 50%, cold: 40%, dizzy: 10%}`` and
    ``fever => {normal: 10%, cold: 30%, dizzy: 60%}``

    Finally, we have transition probabilities:

    ``healthy => healthy (70%)`` and
    ``fever => fever (60%)``.

    Over three days, we observe the sequence ``[normal, cold, dizzy]``,
    and wish to know the maximum likelihood assignment of states for the
    corresponding days, which we compute with the Viterbi algorithm below.

    >>> p_init = np.array([0.6, 0.4])
    >>> p_emit = np.array([[0.5, 0.4, 0.1],
    ...                    [0.1, 0.3, 0.6]])
    >>> p_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> path, logp = librosa.sequence.viterbi(p_emit, p_trans, p_init=p_init,
    ...                                       return_logp=True)
    >>> print(logp, path)
    -4.19173690823075 [0 0 1]
    """

    n_states, n_steps = prob.shape[-2:]

    if transition.shape != (n_states, n_states):
        raise ParameterError(
            "transition.shape={}, must be "
            "(n_states, n_states)={}".format(transition.shape, (n_states, n_states))
        )

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError(
            "Invalid transition matrix: must be non-negative "
            "and sum to 1 on each row."
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError("Invalid probability values: must be between 0 and 1.")

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = tiny(prob)

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif (
        np.any(p_init < 0)
        or not np.allclose(p_init.sum(), 1)
        or p_init.shape != (n_states,)
    ):
        raise ParameterError(
            "Invalid initial state distribution: " "p_init={}".format(p_init)
        )

    log_trans = np.log(transition + epsilon)
    log_prob = np.log(prob + epsilon)
    log_p_init = np.log(p_init + epsilon)

    def _helper(lp):
        # Transpose input
        _state, logp = _viterbi_torch(lp.T, log_trans, log_p_init, device)
        # Transpose outputs for return
        return _state.T, logp

    if log_prob.ndim == 2:
        states, logp = _helper(log_prob)
    else:
        # Vectorize the helper
        __viterbi = np.vectorize(
            _helper, otypes=[np.uint16, np.float64], signature="(s,t)->(t),(1)"
        )

        states, logp = __viterbi(log_prob)

        # Flatten out the trailing dimension introduced by vectorization
        logp = logp[..., 0]

    if return_logp:
        return states, logp

    return states

def transition_loop(n_states, prob):
    """Construct a self-loop transition matrix over ``n_states``.

    The transition matrix will have the following properties:

        - ``transition[i, i] = p`` for all ``i``
        - ``transition[i, j] = (1 - p) / (n_states - 1)`` for all ``j != i``

    This type of transition matrix is appropriate when states tend to be
    locally stable, and there is no additional structure between different
    states.  This is primarily useful for de-noising frame-wise predictions.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length ``n_states``, ``p[i]`` is the probability of self-transition in state ``i``

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_loop(3, 0.5)
    array([[0.5 , 0.25, 0.25],
           [0.25, 0.5 , 0.25],
           [0.25, 0.25, 0.5 ]])

    >>> librosa.sequence.transition_loop(3, [0.8, 0.5, 0.25])
    array([[0.8  , 0.1  , 0.1  ],
           [0.25 , 0.5  , 0.25 ],
           [0.375, 0.375, 0.25 ]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    transition = np.empty((n_states, n_states), dtype=np.float64)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float64)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ParameterError(
            "prob={} must have length equal to n_states={}".format(prob, n_states)
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError(
            "prob={} must have values in the range [0, 1]".format(prob)
        )

    for i, prob_i in enumerate(prob):
        transition[i] = (1.0 - prob_i) / (n_states - 1)
        transition[i, i] = prob_i

    return transition

@deprecate_positional_args
def transition_local(n_states, width, *, window="triangle", wrap=False):
    """Construct a localized transition matrix.

    The transition matrix will have the following properties:

        - ``transition[i, j] = 0`` if ``|i - j| > width``
        - ``transition[i, i]`` is maximal
        - ``transition[i, i - width//2 : i + width//2]`` has shape ``window``

    This type of transition matrix is appropriate for state spaces
    that discretely approximate continuous variables, such as in fundamental
    frequency estimation.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    width : int >= 1 or iterable
        The maximum number of states to treat as "local".
        If iterable, it should have length equal to ``n_states``,
        and specify the width independently for each state.

    window : str, callable, or window specification
        The window function to determine the shape of the "local" distribution.

        Any window specification supported by `filters.get_window` will work here.

        .. note:: Certain windows (e.g., 'hann') are identically 0 at the boundaries,
            so and effectively have ``width-2`` non-zero values.  You may have to expand
            ``width`` to get the desired behavior.

    wrap : bool
        If ``True``, then state locality ``|i - j|`` is computed modulo ``n_states``.
        If ``False`` (default), then locality is absolute.

    See Also
    --------
    librosa.filters.get_window

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    Triangular distributions with and without wrapping

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=False)
    array([[0.667, 0.333, 0.   , 0.   , 0.   ],
           [0.25 , 0.5  , 0.25 , 0.   , 0.   ],
           [0.   , 0.25 , 0.5  , 0.25 , 0.   ],
           [0.   , 0.   , 0.25 , 0.5  , 0.25 ],
           [0.   , 0.   , 0.   , 0.333, 0.667]])

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=True)
    array([[0.5 , 0.25, 0.  , 0.  , 0.25],
           [0.25, 0.5 , 0.25, 0.  , 0.  ],
           [0.  , 0.25, 0.5 , 0.25, 0.  ],
           [0.  , 0.  , 0.25, 0.5 , 0.25],
           [0.25, 0.  , 0.  , 0.25, 0.5 ]])

    Uniform local distributions with variable widths and no wrapping

    >>> librosa.sequence.transition_local(5, [1, 2, 3, 3, 1], window='ones', wrap=False)
    array([[1.   , 0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.   , 0.   , 0.   ],
           [0.   , 0.333, 0.333, 0.333, 0.   ],
           [0.   , 0.   , 0.333, 0.333, 0.333],
           [0.   , 0.   , 0.   , 0.   , 1.   ]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)

    if width.shape != (n_states,):
        raise ParameterError(
            "width={} must have length equal to n_states={}".format(width, n_states)
        )

    if np.any(width < 1):
        raise ParameterError("width={} must be at least 1")

    transition = np.zeros((n_states, n_states), dtype=np.float64)

    # Fill in the widths.  This is inefficient, but simple
    for i, width_i in enumerate(width):
        trans_row = pad_center(
            get_window(window, width_i, fftbins=False), size=n_states
        )
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)

        if not wrap:
            # Knock out the off-diagonal-band elements
            trans_row[min(n_states, i + width_i // 2 + 1) :] = 0
            trans_row[: max(0, i - width_i // 2)] = 0

        transition[i] = trans_row

    # Row-normalize
    transition /= transition.sum(axis=1, keepdims=True)

    return transition
