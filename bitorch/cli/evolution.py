def process_tree_likelihood(arg, json_tree_likelihood):
    json_tree_likelihood['type'] = (
        'bitorch.tree_likelihood.' + json_tree_likelihood['type']
    )
    json_tree_likelihood['tree_model']['type'] = (
        'bitorch.tree_model.' + json_tree_likelihood['tree_model']['type']
    )
