def process_tree_likelihood(arg, json_tree_likelihood):
    if isinstance(json_tree_likelihood, dict):
        if 'torchtree_bito' not in json_tree_likelihood['type']:
            json_tree_likelihood['type'] = (
                'torchtree_bito.' + json_tree_likelihood['type'].split('.')[-1]
            )
            if isinstance(json_tree_likelihood['tree_model'], dict):
                json_tree_likelihood['tree_model']['type'] = (
                    'torchtree_bito.'
                    + json_tree_likelihood['tree_model']['type'].split('.')[-1]
                )
