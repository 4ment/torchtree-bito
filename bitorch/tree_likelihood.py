import os
import tempfile
from collections import namedtuple
from typing import Union

import libsbn
import libsbn.beagle_flags as beagle_flags
import numpy as np
import torch
from phylotorch.core.model import CallableModel
from phylotorch.core.parameter import TransformedParameter
from phylotorch.core.utils import JSONParseError, process_object
from phylotorch.evolution.alignment import Alignment
from phylotorch.evolution.branch_model import BranchModel
from phylotorch.evolution.site_model import SiteModel
from phylotorch.evolution.substitution_model import SubstitutionModel
from phylotorch.evolution.tree_model import (
    ReparameterizedTimeTreeModel,
    TreeModel,
    UnRootedTreeModel,
)
from phylotorch.typing import ID
from torch.distributions import StickBreakingTransform


class TreeLikelihoodModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        inst,
        tree_model: Union[ReparameterizedTimeTreeModel, UnRootedTreeModel],
        subst_model: SubstitutionModel,
        site_model: SiteModel,
        clock_model: BranchModel = None,
    ):
        super().__init__(id_)
        self.inst = inst
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
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

        log_P = treelike(
            self.inst,
            branch_parameters,
            clock_rate,
            subst_rates,
            subst_frequencies,
            weibull_shape,
            requires_grad,
        )
        return log_P

    def handle_model_changed(self, model, obj, index):
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self):
        return max([model.sample_shape for model in self.models()], key=len)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        data[TreeModel.tag]['use_postorder_indices'] = True
        tree_model = process_object(data[TreeModel.tag], dic)

        site_model = process_object(data[SiteModel.tag], dic)
        subst_model = process_object(data[SubstitutionModel.tag], dic)
        thread_count = data.get('thread_count', 1)

        if BranchModel.tag in data:
            clock_model = process_object(data[BranchModel.tag], dic)
            inst = libsbn.rooted_instance(id_)
        else:
            clock_model = None
            inst = libsbn.unrooted_instance(id_)

        if 'file' in data[TreeModel.tag]:
            inst.read_newick_file(data[TreeModel.tag]['file'])
        else:
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
            try:
                if tree_model.tree.is_rooted:
                    tree2 = tree_model.tree.clone(2)
                    tree2.deroot()
                    tmp.write(str(tree2) + ';')
                else:
                    tmp.write(str(tree_model.tree) + ';')
            finally:
                tmp.close()
                inst.read_newick_file(tmp.name)
                os.unlink(tmp.name)

        # Ignore site_pattern and parse alignment instead
        if 'file' in data['site_pattern']['alignment']:
            inst.read_fasta_file(data['site_pattern']['alignment']['file'])
        elif 'sequences' in data['site_pattern']['alignment']:
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
            try:
                alignment = Alignment.from_json(data['site_pattern']['alignment'], dic)
                for idx, sequence in enumerate(alignment):
                    tmp.write('>' + sequence.taxon + '\n')
                    tmp.write(sequence.sequence + '\n')
            finally:
                tmp.close()
                inst.read_fasta_file(tmp.name)
                os.unlink(tmp.name)

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
            if torch.max(tree_model.bounds) == 0.0:
                inst.set_dates_to_be_constant(False)
            else:
                inst.parse_dates_from_taxon_names(False)
            spec = libsbn.PhyloModelSpecification(
                substitution=model_name, site=site_name, clock=clock_name
            )
        else:
            spec = libsbn.PhyloModelSpecification(
                substitution=model_name, site=site_name, clock='strict'
            )

        inst.prepare_for_phylo_likelihood(
            spec, thread_count, [beagle_flags.VECTOR_SSE], False
        )

        tree_model.inst = inst
        return cls(id_, inst, tree_model, subst_model, site_model, clock_model)


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
    def update_libsbn(
        inst,
        branch_lengths,
        clock_rates,
        subst_rates,
        subst_frequencies,
        weibull_shape,
        batch_idx=0,
    ):
        phylo_model_param_block_map = inst.get_phylo_model_param_block_map()

        if clock_rates is not None:
            # Set ratios in tree
            inst.tree_collection.trees[0].initialize_time_tree_using_height_ratios(
                branch_lengths[batch_idx, :].detach().numpy()
            )
            # Set clock rate in tree
            # inst.tree_collection.trees[0].set_relaxed_clock(clock_rates)
            # inst.tree_collection.trees[0].set_strict_clock(clock_rates)
            inst_rates = np.array(inst.tree_collection.trees[0].rates, copy=False)
            inst_rates[:] = clock_rates[batch_idx, :].detach().numpy()
            # libsbn does not use phylo_model_param_block_map for clock rates
            # phylo_model_param_block_map["clock rate"][:] = clock.detach().numpy()
        else:
            inst_branch_lengths = np.array(
                inst.tree_collection.trees[0].branch_lengths, copy=False
            )
            inst_branch_lengths[:-1] = branch_lengths[batch_idx, :].detach().numpy()

        if weibull_shape is not None:
            phylo_model_param_block_map["Weibull shape"][:] = (
                weibull_shape[batch_idx, :].detach().numpy()
            )

        if subst_rates is not None:
            if subst_rates.shape[-1] == 1:
                phylo_model_param_block_map["substitution model rates"][:] = (
                    subst_rates[batch_idx, :].detach().numpy()
                )
            else:
                t = StickBreakingTransform()
                phylo_model_param_block_map["substitution model rates"][:] = t(
                    subst_rates[batch_idx, :].detach()
                ).numpy()

        if subst_frequencies is not None:
            t = StickBreakingTransform()
            phylo_model_param_block_map["substitution model frequencies"][:] = t(
                subst_frequencies[batch_idx, :].detach()
            ).numpy()

    @staticmethod
    def calculate_gradient(
        inst, branch_lengths, clock_rates, subst_rates, subst_frequencies, weibull_shape
    ):
        clock_rate_grad = None
        subst_rates_grad = None
        subst_frequencies_grad = None
        weibull_grad = None

        libsbn_result = inst.phylo_gradients()[0]

        if clock_rates is not None:
            branch_grad = torch.tensor(
                np.array(libsbn_result.gradient['ratios_root_height'])
            )
            clock_rate_grad = torch.tensor(
                np.array(libsbn_result.gradient['clock_model'])
            )
        else:
            branch_grad = torch.tensor(
                np.array(libsbn_result.gradient['branch_lengths'])[:-2]
            )

        if subst_rates is not None:
            substitution_model_grad = np.array(
                libsbn_result.gradient['substitution_model']
            )
            subst_rates_grad = torch.tensor(substitution_model_grad[:-3])

        if subst_frequencies is not None:
            substitution_model_grad = np.array(
                libsbn_result.gradient['substitution_model']
            )
            subst_frequencies_grad = torch.tensor(substitution_model_grad[-3:])

        if weibull_shape is not None:
            weibull_grad = torch.tensor(np.array(libsbn_result.gradient['site_model']))

        return Gradient(
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
    ):
        ctx.inst = inst
        ctx.time = clock_rates is not None
        ctx.weibull = weibull_shape is not None
        ctx.subst_rates = subst_rates is not None
        ctx.subst_frequencies = subst_frequencies is not None
        ctx.save_for_backward(
            branch_lengths, clock_rates, subst_rates, subst_frequencies, weibull_shape
        )

        log_p = []
        all_grad = []

        for batch_idx in range(branch_lengths.shape[0]):
            TreeLikelihoodAutogradFunction.update_libsbn(
                inst,
                branch_lengths,
                clock_rates,
                subst_rates,
                subst_frequencies,
                weibull_shape,
                batch_idx,
            )
            if save_grad:
                grads = TreeLikelihoodAutogradFunction.calculate_gradient(
                    inst,
                    branch_lengths,
                    clock_rates,
                    subst_rates,
                    subst_frequencies,
                    weibull_shape,
                )
                all_grad.append(grads)

            log_p.append(torch.tensor(np.array(inst.log_likelihoods())[0]))

        ctx.grads = all_grad if save_grad else None
        return torch.stack(log_p)

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
            for batch_idx in range(grad_output.shape[0]):
                if grad_output.shape[0] > 1:
                    TreeLikelihoodAutogradFunction.update_libsbn(
                        ctx.inst,
                        branch_lengths,
                        clock_rates,
                        subst_rates,
                        subst_frequencies,
                        weibull_shape,
                        batch_idx,
                    )
                grads = TreeLikelihoodAutogradFunction.calculate_gradient(
                    ctx.inst,
                    branch_lengths,
                    clock_rates,
                    subst_rates,
                    subst_frequencies,
                    weibull_shape,
                )
                all_grads.append(grads)

        branch_grad = torch.stack(
            list(map(lambda x: x.branch_lengths, ctx.grads))
        ) * grad_output.unsqueeze(-1)

        if ctx.time:
            clock_rate_grad = torch.stack(
                list(map(lambda x: x.clocks_rate, ctx.grads))
            ) * grad_output.unsqueeze(-1)
        else:
            clock_rate_grad = None

        if ctx.subst_rates:
            subst_rates_grad = torch.stack(
                list(map(lambda x: x.subst_rates, ctx.grads))
            ) * grad_output.unsqueeze(-1)
        else:
            subst_rates_grad = None

        if ctx.subst_frequencies:
            subst_frequencies_grad = torch.stack(
                list(map(lambda x: x.subst_frequencies, ctx.grads))
            ) * grad_output.unsqueeze(-1)
        else:
            subst_frequencies_grad = None

        if ctx.weibull:
            weibull_grad = torch.stack(
                list(map(lambda x: x.weibull_shape, ctx.grads))
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
        )
