#!/usr/bin/env python

import argparse
import os
import tempfile
from timeit import default_timer as timer

import bito
import bito.beagle_flags as beagle_flags
import numpy as np
import torch
from phylotorch.io import read_tree

from bitorch.tree_likelihood import TreeLikelihoodAutogradFunction
from bitorch.tree_model import NodeHeightAutogradFunction


def create_instance(rooted, tree, args):
    if rooted:
        inst = bito.rooted_instance('id_')
    else:
        inst = bito.unrooted_instance('id_')

    if tree.is_rooted and not rooted:
        # bito expects a trifurcation at the root if the tree is unrooted
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        tree2 = tree.clone(2)
        tree2.deroot()
        tmp.write(str(tree2) + ';')
        tmp.close()
        inst.read_newick_file(tmp.name)
        os.unlink(tmp.name)
    else:
        inst.read_newick_file(args.tree)

    inst.read_fasta_file(args.input)

    if rooted:
        inst.parse_dates_from_taxon_names(True)

    spec = bito.PhyloModelSpecification(
        substitution='JC69', site='constant', clock='strict'
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
def tree_likelihood(inst, branch_lengths):
    treelike = TreeLikelihoodAutogradFunction.apply
    log_prob = treelike(
        inst,
        branch_lengths,
        None,
        None,
        None,
        None,
        False,
    )
    return log_prob


@benchmark
def gradient_tree_likelihood(inst, branch_lengths):
    treelike = TreeLikelihoodAutogradFunction.apply
    log_prob = treelike(
        inst,
        branch_lengths,
        None,
        None,
        None,
        None,
        True,
    )
    return log_prob.backward()


def unrooted_treelikelihood(args):
    tree = read_tree(args.tree, False, False)

    inst = create_instance(False, tree, args)

    branch_lengths = torch.tensor(
        np.array(inst.tree_collection.trees[0].branch_lengths)[:-1]
    ).unsqueeze(0)
    branch_lengths = branch_lengths * 0.001

    total_time, log_prob = tree_likelihood(args.replicates, inst, branch_lengths)
    print(f'  {args.replicates} evaluations: {total_time} ({log_prob})')

    branch_lengths.requires_grad = True
    grad_total_time, _ = gradient_tree_likelihood(args.replicates, inst, branch_lengths)
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(f"treelikelihood,evaluation,off,{total_time}\n")
        args.output.write(f"treelikelihood,gradient,off,{grad_total_time}\n")


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

    branch_lengths = torch.tensor(np.array(inst.tree_collection.trees[0].height_ratios))
    branch_lengths = branch_lengths.unsqueeze(0)

    total_time, _ = ratio_transform_fn(args.replicates, inst, branch_lengths)
    print(f'  {args.replicates} evaluations: {total_time}')

    branch_lengths.requires_grad = True
    total_time, _ = gradient_ratio_transform_fn(args.replicates, inst, branch_lengths)
    print(f'  {args.replicates} gradient evaluations: {total_time}')


@benchmark
def transform_jacobian(inst, branch_lengths):
    inst.tree_collection.trees[0].initialize_time_tree_using_height_ratios(
        branch_lengths
    )
    return bito.log_det_jacobian_height_transform(inst.tree_collection.trees[0])


@benchmark
def gradient_transform_jacobian(inst, branch_lengths):
    inst.tree_collection.trees[0].initialize_time_tree_using_height_ratios(
        branch_lengths
    )
    return bito.gradient_log_det_jacobian_height_transform(
        inst.tree_collection.trees[0]
    )


def ratio_transform_jacobian(args):
    tree = read_tree(args.tree, True, True)

    inst = create_instance(True, tree, args)

    branch_lengths = torch.tensor(np.array(inst.tree_collection.trees[0].height_ratios))

    total_time, log_p = transform_jacobian(args.replicates, inst, branch_lengths)
    print(f'  {args.replicates} evaluations: {total_time} ({log_p})')

    grad_total_time, _ = gradient_transform_jacobian(
        args.replicates, inst, branch_lengths
    )
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(f"ratio_transform_jacobian,evaluation,off,{total_time}\n")
        args.output.write(f"ratio_transform_jacobian,gradient,off,{grad_total_time}\n")


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
    default=None,
    help="""csv output file""",
)
parser.add_argument(
    '--debug', required=False, action='store_true', help="""Debug mode"""
)
args = parser.parse_args()

if args.output:
    args.output.write("function,mode,JIT,time\n")

print('Tree likelihood unrooted:')
unrooted_treelikelihood(args)
print()

try:
    print('Height transform log det Jacobian:')
    ratio_transform_jacobian(args)
    print()
except AttributeError:
    pass

print('Node height transform:')
ratio_transform(args)

if args.output:
    args.output.close()
