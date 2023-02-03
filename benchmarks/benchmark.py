#!/usr/bin/env python

import argparse
import os
import tempfile
from timeit import default_timer as timer

import bito
import bito.beagle_flags as beagle_flags
import dendropy
import numpy as np
import torch
from dendropy import Tree
from torchtree.evolution.tree_model import setup_dates, setup_indexes

from torchtree_bito.tree_likelihood import TreeLikelihoodAutogradFunction
from torchtree_bito.tree_model import NodeHeightAutogradFunction


def read_tree(tree, dated=True, heterochornous=True):
    taxa = dendropy.TaxonNamespace()
    tree_format = 'newick'
    with open(tree) as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'

    tree = Tree.get(
        path=tree,
        schema=tree_format,
        tree_offset=0,
        taxon_namespace=taxa,
        preserve_underscores=True,
        rooting='force-rooted',
    )
    tree.resolve_polytomies(update_bipartitions=True)
    setup_indexes(tree)
    if dated:
        setup_dates(tree, heterochornous)

    return tree


def create_instance(rooted, tree, args, subst_model='JC69'):
    if rooted:
        inst = bito.rooted_instance('id_')
    else:
        inst = bito.unrooted_instance('id_')
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    tmp.write(str(tree) + ';')
    tmp.close()
    inst.read_newick_file(tmp.name)
    os.unlink(tmp.name)
    inst.read_fasta_file(args.input)

    if rooted:
        inst.parse_dates_from_taxon_names(True)

    spec = bito.PhyloModelSpecification(
        substitution=subst_model, site='constant', clock='strict'
    )

    inst.prepare_for_phylo_likelihood(spec, 1, [beagle_flags.VECTOR_SSE], False)
    return inst


def benchmark(f):
    def timed(replicates, *args):
        start = timer()
        for _ in range(replicates):
            out = f(*args)
        end = timer()
        total_time = end - start
        return total_time, out

    return timed


@benchmark
def tree_likelihood(inst, branch_lengths, rates=None, frequencies=None):
    treelike = TreeLikelihoodAutogradFunction.apply
    log_prob = treelike(
        inst,
        branch_lengths,
        None,
        rates,
        frequencies,
        None,
        False,
    )
    return log_prob


@benchmark
def gradient_tree_likelihood(inst, branch_lengths, rates=None, frequencies=None):
    treelike = TreeLikelihoodAutogradFunction.apply
    log_prob = treelike(
        inst,
        branch_lengths,
        None,
        rates,
        frequencies,
        None,
        True,
    )
    log_prob.backward()
    return log_prob


def unrooted_treelikelihood(args, subst_model):
    tree = read_tree(args.tree, False, False)
    tree.collapse_basal_bifurcation()

    inst = create_instance(False, tree, args, subst_model)

    branch_lengths = torch.tensor(
        np.array(inst.tree_collection.trees[0].branch_lengths)[:-1]
    ).unsqueeze(0)
    branch_lengths = branch_lengths * args.scaler
    branch_lengths = torch.clamp(branch_lengths, min=1.0e-6)

    if subst_model == 'GTR':
        rates = torch.full((1, 5), 0.0)
        frequencies = torch.full((1, 3), 0.0)
    else:
        rates = None
        frequencies = None

    total_time, log_prob = tree_likelihood(
        args.replicates, inst, branch_lengths, rates, frequencies
    )
    print(f'  {args.replicates} evaluations: {total_time} ({log_prob})')

    branch_lengths.requires_grad = True
    if subst_model == 'GTR':
        rates.requires_grad = True
        frequencies.requires_grad = True

    grad_total_time, grad_log_prob = gradient_tree_likelihood(
        args.replicates, inst, branch_lengths, rates, frequencies
    )
    print(
        f'  {args.replicates} gradient evaluations: {grad_total_time} ({grad_log_prob})'
    )

    if args.output:
        args.output.write(
            f"treelikelihood{subst_model},evaluation,off,{total_time},"
            f"{log_prob.squeeze().item()}\n"
        )
        args.output.write(
            f"treelikelihood{subst_model},gradient,off,{grad_total_time},"
            f"{grad_log_prob.squeeze().item()}\n"
        )

    if torch.any(torch.isinf(log_prob)):
        inst.set_rescaling(True)
        total_time_r, log_prob_r = tree_likelihood(
            args.replicates, inst, branch_lengths
        )
        print(
            f'  {args.replicates} evaluations rescaled: {total_time_r}'
            f' ({log_prob_r.squeeze().item()})'
        )

        branch_lengths.requires_grad = True
        grad_total_time_r, grad_log_prob_r = gradient_tree_likelihood(
            args.replicates, inst, branch_lengths
        )
        print(
            f'  {args.replicates} gradient evaluations rescaled: {grad_total_time_r}'
            f' ({grad_log_prob_r.squeeze().item()})'
        )

        if args.output:
            args.output.write(
                f"treelikelihood{subst_model}_rescaled,evaluation,off,{total_time_r},"
                f"{log_prob_r.squeeze().item()}\n"
            )
            args.output.write(
                f"treelikelihood{subst_model}_rescaled,gradient,off,"
                f"{grad_total_time_r},{grad_log_prob_r.squeeze().item()}\n"
            )


@benchmark
def ratio_transform_fn(inst, branch_lengths):
    fn = NodeHeightAutogradFunction.apply
    return fn(inst, branch_lengths)


@benchmark
def gradient_ratio_transform_fn(inst, branch_lengths):
    fn = NodeHeightAutogradFunction.apply
    log_P = fn(inst, branch_lengths)
    log_P.backward(torch.ones_like(branch_lengths))
    return None


def ratio_transform(args):
    tree = read_tree(args.tree, True, True)

    inst = create_instance(True, tree, args)

    root_height_ratios = torch.tensor(
        np.array(inst.tree_collection.trees[0].height_ratios)
    )
    root_height_ratios = root_height_ratios.unsqueeze(0)

    total_time, _ = ratio_transform_fn(args.replicates, inst, root_height_ratios)
    print(f'  {args.replicates} evaluations: {total_time}')

    root_height_ratios.requires_grad = True
    grad_total_time, _ = gradient_ratio_transform_fn(
        args.replicates, inst, root_height_ratios
    )
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(f"ratio_transform,evaluation,off,{total_time},\n")
        args.output.write(f"ratio_transform,gradient,off,{grad_total_time},\n")


@benchmark
def transform_jacobian(inst):
    return bito.log_det_jacobian_of_height_transform(inst.tree_collection.trees[0])


@benchmark
def gradient_transform_jacobian(inst):
    return bito.gradient_log_det_jacobian_of_height_transform(
        inst.tree_collection.trees[0]
    )


def heights_from_branch_lengths(tree, eps=1.0e-6):
    heights = torch.empty(2 * len(tree.taxon_namespace) - 1)
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            heights[node.index] = node.date
        else:
            heights[node.index] = max(
                [
                    heights[c.index] + max(eps, c.edge_length)
                    for c in node.child_node_iter()
                ]
            )
    return heights


def ratio_transform_jacobian(args):
    tree = read_tree(args.tree, True, True)

    heights = heights_from_branch_lengths(tree).tolist()
    for n in tree.postorder_internal_node_iter():
        for c in n.child_nodes():
            c.edge_length = heights[n.index] - heights[c.index]

    inst = create_instance(True, tree, args)

    root_height_ratios = torch.tensor(
        np.array(inst.tree_collection.trees[0].height_ratios)
    )
    inst.tree_collection.trees[0].initialize_time_tree_using_height_ratios(
        root_height_ratios
    )

    total_time, log_p = transform_jacobian(args.replicates, inst)
    print(f'  {args.replicates} evaluations: {total_time} ({log_p})')

    grad_total_time, grad_log_p = gradient_transform_jacobian(args.replicates, inst)
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(
            f"ratio_transform_jacobian,evaluation,off,{total_time},{log_p}\n"
        )
        args.output.write(f"ratio_transform_jacobian,gradient,off,{grad_total_time},\n")


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help="""Alignment file""")
parser.add_argument('-t', '--tree', required=True, help="""Tree file""")
parser.add_argument(
    '-r',
    '--replicates',
    required=True,
    type=int,
    help="""Number of replicates""",
)
parser.add_argument(
    "-o",
    "--output",
    type=argparse.FileType("w"),
    help="""csv output file""",
)
parser.add_argument(
    "-s",
    "--scaler",
    type=float,
    default=1.0,
    help="""scale branch lengths""",
)
parser.add_argument('--debug', action='store_true', help="""Debug mode""")
parser.add_argument(
    '--gtr',
    action='store_true',
    help="""Include gradient calculation of GTR parameters""",
)
args = parser.parse_args()

if args.output:
    args.output.write("function,mode,JIT,time,logprob\n")

print('Tree likelihood unrooted:')
print('JC69')
unrooted_treelikelihood(args, 'JC69')
print()

if args.gtr:
    print('GTR')
    unrooted_treelikelihood(args, 'GTR')
    print()

print('Height transform log det Jacobian:')
ratio_transform_jacobian(args)
print()

print('Node height transform:')
ratio_transform(args)

if args.output:
    args.output.close()
