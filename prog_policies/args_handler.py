from .config import Config
from typing import get_type_hints

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    hints = get_type_hints(Config, include_extras=True)
    for param_name in Config.__annotations__:
        param_hints = hints.get(param_name).__dict__.get('__metadata__')
        joined_hints = ', '.join(param_hints) if param_hints else ''
        param_type = hints.get(param_name).__dict__.get('__origin__')
        default_value = Config.__dict__.get(param_name)
        if param_type == bool:
            if default_value == True:
                arg_action = 'store_false'
            else:
                arg_action = 'store_true'
            parser.add_argument(f'--{param_name}', action=arg_action, help=joined_hints)
        else:
            parser.add_argument(f'--{param_name}', default=default_value, type=param_type, help=joined_hints)

    args_dict = vars(parser.parse_args())

    for param_name in Config.__annotations__:
        if Config.__dict__[param_name] != args_dict[param_name]:
            setattr(Config, param_name, args_dict[param_name])
