# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig


@dataclass
class DistillLabelSmoothedCrossEntropyCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--distill-temp', default=0.6, type=float, metavar='D')
        parser.add_argument('--alpha-strategy', default='fix', choices=['fix', 'threshold', 'adaptive'])


@register_criterion('distill_label_smoothed_cross_entropy',
                    dataclass=DistillLabelSmoothedCrossEntropyCriterionConfig,
                    )
class DistillLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.t = args.distill_temp

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        net_output = net_output[0].float()
        lprobs = F.log_softmax(net_output, -1)
        lprobs = lprobs.view(-1, lprobs.shape[-1])
        target = sample['target'].view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        if 'alpha' in sample:
            alpha = sample['alpha'].view(-1, 1)[non_pad_mask]
        else:
            alpha = 0

        nll_prob = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        eps_i = self.eps / lprobs.size(-1)

        if 'teacher_output' in sample and sample['teacher_output'] is not None and torch.is_tensor(alpha):
            teacher_output = sample['teacher_output']
            net_output_lprobs_t = F.log_softmax(net_output / self.t, -1)
            net_output_lprobs_t = net_output_lprobs_t.view(-1, net_output_lprobs_t.shape[-1])

            topk_idx, topk_prob = teacher_output
            topk_idx = topk_idx.view(-1, topk_idx.shape[-1])
            topk_prob = topk_prob.view(-1, topk_prob.shape[-1])

            topk_prob = F.softmax(topk_prob / self.t, -1)

            distill_loss = - (net_output_lprobs_t.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1,
                                                                                                  keepdim=True)
            distill_loss = (distill_loss[non_pad_mask] * alpha).sum()

            nll_loss = (nll_prob[non_pad_mask] * (1 - alpha)).sum()
            smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha)).sum()
            s_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

            loss = distill_loss * self.t * self.t + s_loss
            nll_loss = nll_prob[non_pad_mask].sum()
        else:
            nll_loss = nll_prob[non_pad_mask].sum()
            smooth_loss = smooth_loss[non_pad_mask].sum()
            s_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
            loss = s_loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'nsentences': sample['target'].size(0),
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
