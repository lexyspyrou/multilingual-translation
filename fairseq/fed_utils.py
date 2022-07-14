import hashlib
import logging
import os
import sys

import numpy as np
import torch
import ujson as json
from tqdm import tqdm

from fairseq import distributed_utils
from fairseq import utils
from fairseq.data.indexed_dataset import IndexedCachedDataset
from fairseq.data.indexed_dataset import IndexedDatasetBuilder
# We need to setup root logger before importing any fairseq libraries.
from fairseq.dataclass.configs import FairseqConfig

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
    topk_outp, topk_idx = torch.topk(output, k, dim=-1)
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
        required_batch_size_multiple=8,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
    ).next_epoch_itr(shuffle=False)

    logger.info(f"ITR is being set")

    outputs = [None for _ in range(len(task.dataset('train')))]
    for sample in tqdm(itr, mininterval=5):
        with torch.no_grad():
            if sample is None or len(sample) == 0:
                continue
            sample = utils.move_to_cuda(sample)

            bs, srclen = sample['net_input']['src_tokens'].shape
            output = trainer.model(**sample['net_input'])[0].detach()
            non_padding_mask = sample['target'].ne(task.target_dictionary.pad()).cpu()
            _, tgtlen = sample['target'].shape
            topk_idx, topk_v = output2topk(output, cfg.checkpoint.distill_topk)
            topk_x_shape = (bs, tgtlen, cfg.checkpoint.distill_topk)
            topk_idx, topk_v = topk_idx.view(*topk_x_shape).cpu().numpy(), topk_v.view(*topk_x_shape).cpu().numpy()
            non_padding_mask = non_padding_mask.view(*topk_x_shape[:2]).cpu().numpy().astype(bool)
            for b in range(bs):
                outputs[sample['id'][b].item()] = \
                    topk_idx[b, non_padding_mask[b]].tolist(), \
                    topk_v[b, non_padding_mask[b]].tolist()
    return outputs


def save_expert_outputs(cfg: FairseqConfig, task, trainer):
    logger.info("| Start saving expert outputs..")
    expert_outputs = gen_outputs(cfg, task, trainer)
    output_path = os.path.join(cfg.checkpoint.save_dir,
                               'train_output.json.{}'.format(cfg.distributed_training.distributed_rank))
    json.dump(expert_outputs, open(output_path, 'w'))
    # distributed_utils.barrier(cfg, 'save_expert_outputs')
    if distributed_utils.is_master(cfg.distributed_training):
        expert_outputs_ = []
        val_bleu_path1 = os.path.join(cfg.checkpoint.save_dir, 'val_bleu.json')
        # FIXME was cfg.data[0]
        val_bleu_path2 = os.path.join(cfg.checkpoint.save_dir,
                                      'expert_bleu_{}_{}.json'.format(cfg.task.source_lang, cfg.task.target_lang))
        os.system('cp {} {}'.format(val_bleu_path1, val_bleu_path2))

        for i in range(cfg.distributed_training.distributed_world_size):
            output_path = os.path.join(cfg.checkpoint.save_dir, 'train_output.json.{}'.format(i))
            expert_outputs_.append(json.load(open(output_path, 'r')))
            try:
                os.remove(output_path)
            except:
                pass
        for j in range(len(expert_outputs_[0])):
            for i in range(cfg.distributed_training.distributed_world_size):
                if expert_outputs_[i][j] is not None:
                    expert_outputs[j] = expert_outputs_[i][j]
                    break
            if j > 20: break
            assert expert_outputs[j] is not None

        path = os.path.join(cfg.checkpoint.save_dir,
                            '{}_{}_top_{}_idx'.format(cfg.task.source_lang, cfg.task.target_lang,
                                                      cfg.checkpoint.distill_topk))
        TeacherOutputDataset.save_bin(path, [o[0] for o in expert_outputs], np.int32)

        path = os.path.join(cfg.checkpoint.save_dir,
                            '{}_{}_top_{}_prob'.format(cfg.task.source_lang, cfg.task.target_lang,
                                                       cfg.checkpoint.distill_topk))
        TeacherOutputDataset.save_bin(path, [o[1] for o in expert_outputs], np.float64)

    logger.info("| Save expert@{}_{}".format(cfg.task.source_lang, cfg.task.target_lang))
