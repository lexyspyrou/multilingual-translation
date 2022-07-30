import hashlib
import logging
import os
import sys

import numpy as np
import torch
import ujson as json
from tqdm import tqdm
import torch

from fairseq import distributed_utils
from fairseq import utils
from fairseq.data.indexed_dataset import IndexedCachedDataset
from fairseq.data.indexed_dataset import IndexedDatasetBuilder
# We need to setup root logger before importing any fairseq libraries.
from fairseq.dataclass.configs import FairseqConfig
from fairseq.distributed.utils import get_global_group

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")
FED_VERSION_FN = 'fed_version.v3.idx'


def dist2topk(out_dist, k):
    topk_prob, topk_idx = torch.topk(out_dist, k, dim=-1)
    topk_prob = topk_prob.view(-1, k)  # (B x T) x k
    topk_prob = topk_prob / topk_prob.sum(1, keepdim=True)
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_prob


def output2topk(output, k):
    topk_outp, topk_idx = torch.topk(output, k, dim=-1)  # (B, T, k), where T: target sequence length
    topk_outp = topk_outp.view(-1, k)  # (B x T) x k
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_outp


def get_sample_key(ids):
    if not hasattr(get_sample_key, 'sample_key_cache'):
        get_sample_key.sample_key_cache = {}
    ids_str = ','.join([str(id) for id in sorted(ids)])
    if ids_str not in get_sample_key.sample_key_cache:
        hash_object = hashlib.md5(ids_str.encode())
        get_sample_key.sample_key_cache[ids_str] = hash_object.hexdigest()
    return get_sample_key.sample_key_cache[ids_str]


class TeacherOutputDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))


# class TeacherOutputDataset(IndexedCachedDatasetLegacy):

class TeacherOutputDataset(IndexedCachedDataset):
    dtype2size = {
        float: 8,
        int: 4,
    }

    def __init__(self, prefix):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a)
        if self.dtype == np.int32 or self.dtype == np.int or self.dtype == np.int64:
            item = item.long()
        else:
            item = item.float()
        return item


def gen_outputs(cfg: FairseqConfig, task, trainer):
    trainer.model.eval()
    itr = task.get_batch_iterator(
        dataset=task.dataset('train'),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.max_tokens_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        # Defines the number examples fetched at each batch iteration per language
        # required_batch_size_multiple=8,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
    ).next_epoch_itr(shuffle=False)
    # We initialize the outputs as a dictionary with lang pair as keys and None values
    # for the size of the language pair (examples)
    outputs = {lang_pair: [None for _ in range(task.dataset('train').datasets[lang_pair].__len__())]
               for lang_pair in task.lang_pairs}
    # This was equal to the number of eng-rus sentences (Investigate).

    for sample in tqdm(itr):
        if not sample: continue
        for lang_pair, lang_pair_values in sample.items():
            with torch.no_grad():
                if lang_pair_values is None or len(lang_pair_values) == 0:
                    continue
                lang_pair_values = utils.move_to_cuda(lang_pair_values)

                batch_size, src_len = lang_pair_values['net_input']['src_tokens'].shape
                # tgt_len the length of the target sentence in the current batch per language (padded if necessary)
                _, tgt_len = lang_pair_values['target'].shape
                # Pass all the examples (in parallel) from the Transformer network.
                output = trainer.model.models[lang_pair](**lang_pair_values['net_input'])[0].detach()

                non_padding_mask = lang_pair_values['target'].ne(task.target_dictionary.pad()).cpu()
                top_k_idx, top_k_v = output2topk(output, cfg.distillation.distill_topk)
                top_k_x_shape = (batch_size, tgt_len, cfg.distillation.distill_topk)
                # Asserted that both vectors have the expected shape (B, T, k)
                top_k_idx, top_k_v = top_k_idx.view(*top_k_x_shape).cpu().numpy(), top_k_v.view(
                    *top_k_x_shape).cpu().numpy()
                non_padding_mask = non_padding_mask.view(*top_k_x_shape[:2]).cpu().numpy().astype(bool)
                # logger.info(f"non_padding_mask: {non_padding_mask.shape}")
                for example_id in range(batch_size):
                    outputs[lang_pair][lang_pair_values['id'][example_id].item()] = \
                        tuple((top_k_idx[example_id, non_padding_mask[example_id]].tolist(),
                               top_k_v[example_id, non_padding_mask[example_id]].tolist()))
    for lang_pair in task.lang_pairs:
        print(f"Language pair: {lang_pair}")
        print(f"outputs length: {len(outputs[lang_pair])}")
    return outputs


def save_expert_outputs(cfg: FairseqConfig, task, trainer):
    logger.info("| Start saving expert outputs..")

    # We adapted and fixed this function.expert_outputs = Dict[lang_pair] -> list
    expert_outputs_dict = gen_outputs(cfg, task, trainer)

    output_path = os.path.join(cfg.checkpoint.save_dir,
                               'train_output.json.{}'.format(cfg.distributed_training.distributed_rank))
    logger.info(f" Saving results to path: {output_path}")

    json.dump(expert_outputs_dict, open(output_path, 'w'))

    # Wait for all the processes to complete
    if torch.distributed.is_initialized():
        torch.distributed.barrier(get_global_group())
    logger.info("All processes have now finished computing the probabilities.")

    # The main process will aggregate the results into 1 dataset.
    if distributed_utils.is_master(cfg.distributed_training):
        aggregated_expert_outputs_ = {lang_pair: [] for lang_pair in task.lang_pairs}
        for worker_id in range(cfg.distributed_training.distributed_world_size):

            saved_output_path = os.path.join(cfg.checkpoint.save_dir, 'train_output.json.{}'.format(worker_id))
            logger.info(f'Loading expert output from worker_id {worker_id} and path {saved_output_path}')
            saved_experts_dict = json.load(open(saved_output_path, 'r'))

            # Each language pair is key. The value is a list containing distributed_world_size lists.
            # Each of those lists, is a list of number of examples of that language pairs.
            # However, many indexes will be invalid (None), because that example didn't belong to the partition
            # that specific worker. Each example is
            # only processed by 1 worker (Remember we want to parallelize the work)
            for lang_pair, top_k_tuples in saved_experts_dict.items():
                aggregated_expert_outputs_[lang_pair].append(top_k_tuples)

        for lang_pair in task.lang_pairs:
            for lang_example_idx in range(len(aggregated_expert_outputs_[lang_pair][0])):
                for worker_id in range(cfg.distributed_training.distributed_world_size):
                    if aggregated_expert_outputs_[lang_pair][worker_id][lang_example_idx] is not None:
                        expert_outputs_dict[lang_pair][lang_example_idx] = \
                            aggregated_expert_outputs_[lang_pair][worker_id][lang_example_idx]
                        break
                if expert_outputs_dict[lang_pair][lang_example_idx] is None:
                    logger.warning(
                        f'{lang_pair}: Skipping sentence: {lang_example_idx}  due to '
                        f'invalid size - max_positions')
                    expert_outputs_dict[lang_pair][lang_example_idx] = ([], [])

        for lang_pair in task.lang_pairs:
            src, tgt = lang_pair.split("-")
            path = os.path.join(cfg.checkpoint.save_dir,
                                '{}_{}_top_{}_idx'.format(src, tgt,
                                                          cfg.checkpoint.distill_topk))
            TeacherOutputDataset.save_bin(path, [word_idx for word_idx, _ in expert_outputs_dict[lang_pair]],
                                          np.int32)

            path = os.path.join(cfg.checkpoint.save_dir,
                                '{}_{}_top_{}_prob'.format(src, tgt,
                                                           cfg.checkpoint.distill_topk))
            TeacherOutputDataset.save_bin(path, [word_proba for _, word_proba in expert_outputs_dict[lang_pair]],
                                          np.float64)

            logger.info("| Saved expert@{}_{}".format(src, tgt))
