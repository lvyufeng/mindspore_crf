from typing import List, Optional
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Parameter
from mindspore.common.initializer import initializer, Uniform
from mindspore._checkparam import Validator as validator
from mindspore import ms_function


def sequence_mask(seq_length, max_length):
    """generate mask matrix by seq_length"""
    range_vector = mnp.arange(0, max_length, 1, seq_length.dtype)
    result = range_vector < seq_length.view(seq_length.shape + (1,))
    return result.astype(mindspore.int64)

class CRF(nn.Cell):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
        reduction: Specifies  the reduction to apply to the output:
            ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
            ``sum``: the output will be summed over batches. ``mean``: the output will be
            averaged over batches. ``token_mean``: the output will be averaged over tokens.

    Attributes:
        start_transitions (`~Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False, reduction: str = 'sum') -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.reduction = reduction
        self.start_transitions = Parameter(initializer(Uniform(0.1), (num_tags,)), name='start_transitions')
        self.end_transitions = Parameter(initializer(Uniform(0.1), (num_tags,)), name='end_transitions')
        self.transitions = Parameter(initializer(Uniform(0.1), (num_tags, num_tags)), name='transitions')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def construct(self, emissions, tags, seq_length=None):
        max_length = tags.shape[1] if self.batch_first else tags.shape[0]

        if self.batch_first:
            emissions = emissions.swapaxes(0, 1)
            tags = tags.swapaxes(0, 1)

        if seq_length is None:
            mask = mnp.ones_like(tags, dtype=mindspore.int64)
        else:
            mask = sequence_mask(seq_length, max_length)

        if not self.batch_first:
            mask = mask.swapaxes(0, 1)
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if self.reduction == 'none':
            return llh
        elif self.reduction == 'sum':
            return llh.sum()
        elif self.reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.astype(emissions.dtype).sum()


    def decode(self, emissions, mask):
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        # self._validate(emissions, mask=mask)
        if mask is None:
            mask = mnp.ones(emissions.shape[:2], dtype=mindspore.int64)

        if self.batch_first:
            emissions = emissions.swapaxes(0, 1)
            mask = mask.swapaxes(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        # assert emissions.ndim == 3 and tags.ndim == 2
        # assert emissions.shape[:2] == tags.shape
        # assert emissions.shape[2] == self.num_tags
        # assert mask.shape == tags.shape
        # assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.astype(emissions.dtype)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, mnp.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, mnp.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.astype(mindspore.int64).sum(axis=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, mnp.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # assert emissions.ndim == 3 and mask.ndim == 2
        # assert emissions.shape[:2] == mask.shape
        # assert emissions.shape[2] == self.num_tags
        # assert mask[0].all()

        seq_length = emissions.shape[0]

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.expand_dims(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].expand_dims(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = mnp.log(mnp.sum(mnp.exp(next_score), axis=1))

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = mnp.where(mask[i].expand_dims(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return mnp.log(mnp.sum(mnp.exp(score), axis=1))

    @ms_function
    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # assert emissions.ndim == 3 and mask.ndim == 2
        # assert emissions.shape[:2] == mask.shape
        # assert emissions.shape[2] == self.num_tags
        # assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = ()

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.expand_dims(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].expand_dims(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            indices = next_score.argmax(axis=1)
            next_score = next_score.max(axis=1)
            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = mnp.where(mask[i].expand_dims(1), next_score, score)
            history += (indices,)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.astype(mindspore.int64).sum(axis=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            best_last_tag = score[idx].argmax(axis=0)
            best_tags = [best_last_tag]
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                # print(history[:seq_ends[idx]], hist)
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list