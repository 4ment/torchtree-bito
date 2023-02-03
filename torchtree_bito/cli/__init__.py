from torchtree.cli.plugin import Plugin


class BITO(Plugin):
    def load_arguments(self, subparsers):
        for name, parser in subparsers._name_parser_map.items():
            parser.add_argument(
                '--bito',
                action="store_true",
                help="""use bito""",
            )
            parser.add_argument(
                '--bito_gpu',
                action="store_true",
                help="use GPU in bito/BEAGLE",
            )
            parser.add_argument(
                '--bito_disable_sse',
                action="store_true",
                help="disable SSE in bito/BEAGLE",
            )

    def process_tree_likelihood(self, arg, json_tree_likelihood):
        if arg.bito:
            if isinstance(json_tree_likelihood, dict):
                json_tree_likelihood['type'] = (
                    'torchtree_bito.' + json_tree_likelihood['type'].split('.')[-1]
                )
            if isinstance(json_tree_likelihood['tree_model'], dict):
                json_tree_likelihood['tree_model']['type'] = (
                    'torchtree_bito.'
                    + json_tree_likelihood['tree_model']['type'].split('.')[-1]
                )

            if arg.bito_gpu:
                json_tree_likelihood['use_gpu'] = True

            if arg.bito_disable_sse:
                json_tree_likelihood['use_sse'] = False
