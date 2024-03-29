import os
import tempfile
from collections import namedtuple
from typing import Union

import bito
import bito.phylo_gradient_mapkeys as gradient_keys
import bito.phylo_model_mapkeys as model_keys
import numpy as np
import torch
from bito import beagle_flags
from torch.distributions import StickBreakingTransform
from torchtree.core.model import CallableModel
from torchtree.core.parameter import TransformedParameter
from torchtree.core.utils import JSONParseError, process_object
from torchtree.evolution.alignment import Alignment
from torchtree.evolution.branch_model import BranchModel
from torchtree.evolution.site_model import SiteModel
from torchtree.evolution.site_pattern import SitePattern
from torchtree.evolution.substitution_model.abstract import SubstitutionModel
from torchtree.evolution.tree_model import (
    ReparameterizedTimeTreeModel,
    TreeModel,
    UnRootedTreeModel,
)
from torchtree.typing import ID

from torchtree_bito.utils import flatten_2D


class TreeLikelihoodModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        inst,
        tree_model: Union[ReparameterizedTimeTreeModel, UnRootedTreeModel],
        subst_model: SubstitutionModel,
        site_model: SiteModel,
        clock_model: BranchModel = None,
        thread_count=1,
    ):
        super().__init__(id_)
        self.inst = inst
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        self.thread_count = thread_count
        self.rescale = False

    def _call(self):
        treelike = TreeLikelihoodAutogradFunction.apply
        clock_rate = None
        weibull_shape = (
            None if self.site_model.rates().shape[-1] == 1 else self.site_model.shape
        )

        if self.clock_model:
            branch_parameters = self.tree_model._internal_heights.tensor
            clock_rate = self.clock_model._rates.tensor
        else:
            branch_parameters = self.tree_model.branch_lengths()

        requires_grad = branch_parameters.requires_grad

        subst_rates = None
        subst_frequencies = None
        if isinstance(self.subst_model._frequencies, TransformedParameter):
            subst_frequencies = self.subst_model._frequencies.x.tensor

        if hasattr(self.subst_model, '_rates') and isinstance(
            self.subst_model._rates, TransformedParameter
        ):
            subst_rates = self.subst_model._rates.x.tensor
        elif hasattr(self.subst_model, '_kappa'):
            subst_rates = self.subst_model.kappa

        while True:
            log_P = treelike(
                self.inst,
                branch_parameters,
                clock_rate,
                subst_rates,
                subst_frequencies,
                weibull_shape,
                requires_grad,
                self.thread_count,
            )
            if not self.rescale and torch.any(torch.isinf(log_P)):
                self.inst.set_rescaling(True)
                self.rescale = True
            else:
                break
        return log_P

    def handle_parameter_changed(self, variable, index, event):
        pass

    def _sample_shape(self) -> torch.Size:
        return max([model.sample_shape for model in self.models()], key=len)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        data[TreeModel.tag]['use_postorder_indices'] = True
        tree_model = process_object(data[TreeModel.tag], dic)

        site_model = process_object(data[SiteModel.tag], dic)
        subst_model = process_object(data[SubstitutionModel.tag], dic)
        thread_count = data.get('thread_count', 1)
        use_tip_states = data.get('use_tip_states', False)
        use_bito_gpu = data.get('use_gpu', False)
        use_sse = data.get('use_sse', True)

        if BranchModel.tag in data:
            clock_model = process_object(data[BranchModel.tag], dic)
            inst = bito.rooted_instance(id_)
        else:
            clock_model = None
            inst = bito.unrooted_instance(id_)

        if 'file' in data[TreeModel.tag]:
            inst.read_newick_file(data[TreeModel.tag]['file'])
        else:
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
            try:
                options = {'schema': 'newick', 'suppress_internal_node_labels': True}
                if clock_model is None and tree_model.tree.is_rooted:
                    tree2 = tree_model.tree.clone(2)
                    tree2.deroot()
                    tmp.write(tree2.as_string(**options) + ';')
                else:
                    tmp.write(tree_model.tree.as_string(**options) + ';')
            finally:
                tmp.close()
                inst.read_newick_file(tmp.name)
                os.unlink(tmp.name)

        # Ignore site_pattern and parse alignment instead

        # alignment is a reference to an object already parsed
        if isinstance(data[SitePattern.tag]['alignment'], str):
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
            try:
                alignment = dic[data[SitePattern.tag]['alignment']]
                for _, sequence in enumerate(alignment):
                    tmp.write('>' + sequence.taxon + '\n')
                    tmp.write(sequence.sequence + '\n')
            finally:
                tmp.close()
                inst.read_fasta_file(tmp.name)
                os.unlink(tmp.name)
        # alignment contains a file entry
        elif 'file' in data[SitePattern.tag]['alignment']:
            inst.read_fasta_file(data[SitePattern.tag]['alignment']['file'])
        # alignment contains a dictionary of sequences
        elif 'sequences' in data[SitePattern.tag]['alignment']:
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
            try:
                alignment = Alignment.from_json(data[SitePattern.tag]['alignment'], dic)
                for idx, sequence in enumerate(alignment):
                    tmp.write('>' + sequence.taxon + '\n')
                    tmp.write(sequence.sequence + '\n')
            finally:
                tmp.close()
                inst.read_fasta_file(tmp.name)
                os.unlink(tmp.name)
        else:
            raise JSONParseError('site_pattern is misspecified')

        model_name = data[SubstitutionModel.tag]['type'].split('.')[-1]
        if model_name not in ('JC69', 'HKY', 'GTR'):
            raise JSONParseError('Substitution model should be JC69, HKY or GTR')

        site_name = (
            'weibull+' + str(site_model.rates().shape[0])
            if site_model.rates().shape[0] > 1
            else 'constant'
        )

        if clock_model:
            clock_name = 'strict'
            if torch.max(tree_model.sampling_times) == 0.0:
                inst.set_dates_to_be_constant(False)
            else:
                inst.parse_dates_from_taxon_names(False)
            spec = bito.PhyloModelSpecification(
                substitution=model_name, site=site_name, clock=clock_name
            )
        else:
            spec = bito.PhyloModelSpecification(
                substitution=model_name, site=site_name, clock='strict'
            )
        bito_beagle_flags = []
        if use_bito_gpu:
            bito_beagle_flags.append(beagle_flags.PROCESSOR_GPU)
        elif use_sse:
            bito_beagle_flags.append(beagle_flags.VECTOR_SSE)
        else:
            bito_beagle_flags.append(beagle_flags.VECTOR_NONE)

        inst.prepare_for_phylo_likelihood(
            spec, thread_count, bito_beagle_flags, use_tip_states, thread_count
        )

        tree_model.inst = inst
        return cls(
            id_, inst, tree_model, subst_model, site_model, clock_model, thread_count
        )


Gradient = namedtuple(
    'gradient',
    [
        'branch_lengths',
        'clock_rates',
        'subst_rates',
        'subst_frequencies',
        'weibull_shape',
    ],
)


class TreeLikelihoodAutogradFunction(torch.autograd.Function):
    @staticmethod
    def update_bito(
        inst, branch_lengths, clock_rates, subst_rates, subst_frequencies, weibull_shape
    ):
        phylo_model_param_block_map = inst.get_phylo_model_param_block_map()

        tree_count = len(inst.tree_collection.trees)

        if clock_rates is not None:
            # Set ratios in tree
            for idx in range(tree_count):
                inst.tree_collection.trees[
                    idx
                ].initialize_time_tree_using_height_ratios(
                    branch_lengths[idx].detach().numpy()
                )
                # Set clock rate in tree
                if idx < clock_rates.shape[0]:
                    inst_rates = np.array(
                        inst.tree_collection.trees[idx].rates, copy=False
                    )
                    inst_rates[:] = clock_rates[idx].detach().numpy()
        else:
            for idx in range(tree_count):
                inst_branch_lengths = np.array(
                    inst.tree_collection.trees[idx].branch_lengths, copy=False
                )
                inst_branch_lengths[:-1] = branch_lengths.detach().numpy()

        if weibull_shape is not None:
            phylo_model_param_block_map[model_keys.SITE_MODEL][
                :
            ] = weibull_shape.detach().numpy()

        if subst_rates is not None:
            # HKY
            if subst_rates.shape[-1] == 1:
                phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_RATES][
                    :
                ] = subst_rates.detach().numpy()
            # GTR
            else:
                t = StickBreakingTransform()
                phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_RATES][:] = t(
                    subst_rates.detach()
                ).numpy()

        if subst_frequencies is not None:
            t = StickBreakingTransform()
            phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_FREQUENCIES][
                :
            ] = t(subst_frequencies.detach()).numpy()

    @staticmethod
    def calculate_gradient(
        inst, branch_lengths, clock_rates, subst_rates, subst_frequencies, weibull_shape
    ):
        clock_rate_grad = None
        subst_rates_grad = None
        subst_frequencies_grad = None
        weibull_grad = None

        tree_count = len(inst.tree_collection.trees)

        bito_result = inst.phylo_gradients()
        options = {'dtype': branch_lengths.dtype, 'device': branch_lengths.device}

        log_likelihoods = torch.tensor(
            [tree.log_likelihood for tree in bito_result], **options
        )

        if clock_rates is not None:
            branch_grad = []
            for idx in range(tree_count):
                branch_grad.append(
                    np.array(
                        bito_result[idx].gradient[gradient_keys.RATIOS_ROOT_HEIGHT]
                    )
                )
            branch_grad = torch.tensor(np.stack(branch_grad), **options)

            if clock_rates.requires_grad:
                clock_rate_grad = []
                for idx in range(tree_count):
                    clock_rate_grad.append(
                        np.array(bito_result[idx].gradient[gradient_keys.CLOCK_MODEL])
                    )
                clock_rate_grad = torch.tensor(np.stack(clock_rate_grad), **options)
        else:
            branch_grad = []
            for idx in range(tree_count):
                branch_grad = np.array(
                    bito_result[idx].gradient[gradient_keys.BRANCH_LENGTHS]
                )[:-2]
            branch_grad = torch.tensor(np.stack(branch_grad), **options)

        if subst_rates is not None:
            subst_rates_grad = []
            for idx in range(tree_count):
                subst_rates_grad.append(
                    np.array(
                        bito_result[idx].gradient[
                            gradient_keys.SUBSTITUTION_MODEL_RATES
                        ]
                    )
                )
            subst_rates_grad = torch.tensor(np.stack(subst_rates_grad), **options)

        if subst_frequencies is not None:
            subst_frequencies_grad = []
            for idx in range(tree_count):
                subst_frequencies_grad.append(
                    np.array(
                        bito_result[idx].gradient[
                            gradient_keys.SUBSTITUTION_MODEL_FREQUENCIES
                        ]
                    )
                )
            subst_frequencies_grad = torch.tensor(
                np.stack(subst_frequencies_grad), **options
            )

        if weibull_shape is not None:
            weibull_grad = []
            for idx in range(tree_count):
                weibull_grad.append(
                    np.array(bito_result[idx].gradient[gradient_keys.SITE_MODEL])
                )
            weibull_grad = torch.tensor(np.stack(weibull_grad), **options)

        return log_likelihoods, Gradient(
            branch_grad,
            clock_rate_grad,
            subst_rates_grad,
            subst_frequencies_grad,
            weibull_grad,
        )

    @staticmethod
    def forward(
        ctx,
        inst,
        branch_lengths,
        clock_rates=None,
        subst_rates=None,
        subst_frequencies=None,
        weibull_shape=None,
        save_grad=False,
        thread_count=1,
    ):
        ctx.inst = inst
        ctx.thread_count = thread_count
        ctx.time = clock_rates is not None
        ctx.weibull = weibull_shape is not None
        ctx.subst_rates = subst_rates is not None
        ctx.subst_frequencies = subst_frequencies is not None
        ctx.save_for_backward(
            branch_lengths, clock_rates, subst_rates, subst_frequencies, weibull_shape
        )
        options = {'dtype': branch_lengths.dtype, 'device': branch_lengths.device}

        log_p = []
        all_grad = []

        branch_lengths2 = flatten_2D(branch_lengths)
        clock_rates2 = flatten_2D(clock_rates)
        subst_rates2 = flatten_2D(subst_rates)
        subst_frequencies2 = flatten_2D(subst_frequencies)
        weibull_shape2 = flatten_2D(weibull_shape)

        for batch_idx in range(0, branch_lengths2.shape[0], thread_count):
            end_idx = batch_idx + thread_count
            params = [
                None if p is None else p[batch_idx:end_idx]
                for p in (
                    branch_lengths2,
                    clock_rates2,
                    subst_rates2,
                    subst_frequencies2,
                    weibull_shape2,
                )
            ]
            TreeLikelihoodAutogradFunction.update_bito(
                inst,
                *params,
            )
            if save_grad:
                (
                    log_likelihoods,
                    grads,
                ) = TreeLikelihoodAutogradFunction.calculate_gradient(
                    inst,
                    *params,
                )
                all_grad.append(grads)
                log_p.append(log_likelihoods)
            else:
                log_p.append(torch.tensor(np.array(inst.log_likelihoods()), **options))

        log_p = torch.concat(log_p, 0)
        if len(branch_lengths.shape[:-1]) > 1:
            log_p = log_p.view(branch_lengths.shape[:-1])
        ctx.grads = all_grad if save_grad else None
        return log_p

    @staticmethod
    def backward(ctx, grad_output):
        (
            branch_lengths,
            clock_rates,
            subst_rates,
            subst_frequencies,
            weibull_shape,
        ) = ctx.saved_tensors

        if ctx.grads is not None:
            all_grads = ctx.grads
        else:
            all_grads = []
            branch_lengths2 = flatten_2D(branch_lengths)
            clock_rates2 = flatten_2D(clock_rates)
            subst_rates2 = flatten_2D(subst_rates)
            subst_frequencies2 = flatten_2D(subst_frequencies)
            weibull_shape2 = flatten_2D(weibull_shape)

            for batch_idx in range(0, branch_lengths2.shape[0], ctx.thread_count):
                end_idx = batch_idx + ctx.thread_count
                params = [
                    None if p is None else p[batch_idx:end_idx]
                    for p in (
                        branch_lengths2,
                        clock_rates2,
                        subst_rates2,
                        subst_frequencies2,
                        weibull_shape2,
                    )
                ]
                if grad_output.shape[0] > 1:
                    TreeLikelihoodAutogradFunction.update_bito(
                        ctx.inst,
                        *params,
                    )
                _, grads = TreeLikelihoodAutogradFunction.calculate_gradient(
                    ctx.inst,
                    *params,
                )
                all_grads.append(grads)

        branch_grad = torch.cat(
            list(map(lambda x: x.branch_lengths, all_grads))
        ) * grad_output.unsqueeze(-1)

        if ctx.time and clock_rates.requires_grad:
            clock_rate_grad = torch.cat(
                list(map(lambda x: x.clock_rates, all_grads))
            ) * grad_output.unsqueeze(-1)
        else:
            clock_rate_grad = None

        if ctx.subst_rates:
            subst_rates_grad = torch.cat(
                list(map(lambda x: x.subst_rates, all_grads))
            ) * grad_output.unsqueeze(-1)
        else:
            subst_rates_grad = None

        if ctx.subst_frequencies:
            subst_frequencies_grad = torch.cat(
                list(map(lambda x: x.subst_frequencies, all_grads))
            ) * grad_output.unsqueeze(-1)
        else:
            subst_frequencies_grad = None

        if ctx.weibull:
            weibull_grad = torch.cat(
                list(map(lambda x: x.weibull_shape, all_grads))
            ) * grad_output.unsqueeze(-1)
        else:
            weibull_grad = None

        return (
            None,
            branch_grad,
            clock_rate_grad,
            subst_rates_grad,
            subst_frequencies_grad,
            weibull_grad,
            None,
            None,
        )
