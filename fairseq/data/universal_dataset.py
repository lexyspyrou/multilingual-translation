# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq.data import FairseqDataset
from fairseq.data import data_utils
from .. import utils


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True, universal=False, dict_len=None, distill_topk=4
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_lng = torch.LongTensor([s['src_lng'] for s in samples])
    tgt_lng = torch.LongTensor([s['tgt_lng'] for s in samples])
    dataset_id = torch.LongTensor([s['dataset_id'] for s in samples])

    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_lng = src_lng.index_select(0, sort_order)
    tgt_lng = tgt_lng.index_select(0, sort_order)
    dataset_id = dataset_id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'dataset_id': dataset_id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'lng': torch.stack([src_lng, tgt_lng], dim=1) if universal else None
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if 'topk_idx' in samples[0] and samples[0]['topk_idx'] is not None:
        sizes = max(v['topk_idx'].size(0) for v in samples), distill_topk
        teacher_outputs = samples[0]['topk_idx'][0].new(len(samples), *sizes).fill_(pad_idx).long(), \
                          samples[0]['topk_prob'][0].new(len(samples), *sizes).fill_(pad_idx)

        def copy_tensor(src, dst):
            dst.copy_(src)

        for i, v in enumerate(samples):
            vv = v['topk_idx'].long()  # T * K
            copy_tensor(vv[:, :distill_topk], teacher_outputs[0][i, :len(vv), :distill_topk])
            vv = v['topk_prob']
            copy_tensor(vv[:, :distill_topk], teacher_outputs[1][i, :len(vv), :distill_topk])

        batch['teacher_output'] = teacher_outputs[0].index_select(0, sort_order), \
                                  teacher_outputs[1].index_select(0, sort_order)
        batch['alpha'] = torch.FloatTensor([s['alpha'] for s in samples]).view(-1, 1).expand(-1, target.shape[1])
    return batch


class UniversalDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side.
            Default: ``True``
        left_pad_target (bool, optional): pad target tensors on the left side.
            Default: ``False``
        max_source_positions (int, optional): max number of tokens in the source
            sentence. Default: ``1024``
        max_target_positions (int, optional): max number of tokens in the target
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
        remove_eos_from_source (bool, optional): if set, removes eos from end of
            source if it's present. Default: ``False``
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent. Default: ``False``
    """

    def __init__(
            self, args, src, src_sizes, src_dict, src_lngs, tgt_lngs,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            dataset_ids=None, lng_borders=None, dataset_names=None,
            topk_idxs=None, topk_probs=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
            upsampling_max=100000, expert_scores=None, is_train=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.args = args
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_lngs = src_lngs
        self.tgt_lngs = tgt_lngs
        self.lng_borders = lng_borders if lng_borders is not None else [0, len(src)]
        self.num_dataset = len(self.lng_borders) - 1
        self.dataset_ids = dataset_ids if dataset_ids is not None else [0] * len(src)
        self.dataset_names = dataset_names

        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.universal = self.num_dataset > 1 or args.universal
        self.dict_len = len(tgt_dict)

        if self.universal:
            print("| [Universal] dataset init.")
        else:
            print("| [bilingual] dataset init.")

        if self.shuffle:
            self.lng_max_size = 0
            for left, right in zip(self.lng_borders[:-1], self.lng_borders[1:]):
                self.lng_max_size = min(max(self.lng_max_size, right - left), upsampling_max)

        self.topk_idxs = topk_idxs
        self.topk_probs = topk_probs
        if expert_scores is None:
            self.expert_scores = [None for _ in range(self.num_dataset)]
        else:
            self.expert_scores = expert_scores
        self.student_scores = [0 for _ in range(self.num_dataset)]
        self.DEFAULT_ALPHA = 0.9
        self.is_train = is_train

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        it = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'src_lng': self.src_lngs[index],
            'tgt_lng': self.tgt_lngs[index] if self.tgt_lngs is not None else None,
            'dataset_id': self.dataset_ids[index] if self.dataset_ids is not None else None,
        }
        if self.topk_idxs is not None and self.is_train:
            assert self.topk_idxs[index].shape[0] == self.tgt[index].shape[0], (
                self.topk_idxs[index].shape, self.tgt[index].shape)
            it['topk_idx'] = self.topk_idxs[index]
            it['topk_prob'] = self.topk_probs[index] if self.tgt_lngs is not None else None
            if self.expert_scores[self.dataset_ids[index]] is None:
                it['alpha'] = self.DEFAULT_ALPHA
            else:
                it['alpha'] = self.get_alpha(self.dataset_ids[index])
        return it

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, universal=self.universal, dict_len=self.dict_len
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'src_lng': 0,
                'tgt_lng': 0,
                'topk_idx': None,
                'topk_prob': None,
                'dataset_id': 0
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.src, 'supports_prefetch')
                and self.src.supports_prefetch
                and hasattr(self.tgt, 'supports_prefetch')
                and self.tgt.supports_prefetch
        )

    def get_alpha(self, ds_id):
        if self.args.alpha_strategy == 'fix':
            return self.DEFAULT_ALPHA
        if self.args.alpha_strategy == 'threshold':
            if self.student_scores[ds_id] < self.expert_scores[ds_id] + 1:
                return self.DEFAULT_ALPHA
            else:
                return 0
        if self.args.alpha_strategy == 'adaptive':
            if self.student_scores[ds_id] < self.expert_scores[ds_id] - 2:
                return self.DEFAULT_ALPHA
            else:
                return self.DEFAULT_ALPHA * (0.5 ** (self.student_scores[ds_id] - self.expert_scores[ds_id] + 2))
