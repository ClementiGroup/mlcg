"""pyyaml does not support serializing nested type hints from typing and torch_geometric.nn.MessagePassing does include such information in the class in the form of inspect.Parameter objects.
This files registers to yaml a (de)serialization procedure for inspect.Parameter objects that supports type hint annotation such as `typing.Optional[typing.Tuple[int,torch.Tensor, typing.List[torch.Tensor]]]`.
These annotations are stored using their string representation, e.g. `"typing.Optional[int]"`, so the desirialization corresponds to recursively manipulate the string to convert the individual type strings to into python types/classes.
"""

import inspect
import yaml
import builtins


def split_on_comma_not_in_bracket(string):
    out = [""]
    # count the brackets so that select comma only when they are balanced
    brackets = {"[": 0, "]": 0}
    # to skip the space after the comma
    skip_next = False
    for c in string:
        if skip_next:
            skip_next = False
            if c == " ":
                continue
        if c == "[":
            brackets["["] += 1
        if c == "]":
            brackets["]"] -= 1
        if c == "," and sum(brackets.values()) == 0:
            out.append("")
            skip_next = True
            continue
        out[-1] += c
    return out


def get_cls(class_path):
    if "." in class_path:
        class_module, class_name = class_path.rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    else:
        args_class = builtins.__dict__.get(class_path)
        assert args_class is not None
    return args_class


def is_sbrk(string):
    if "[" in string and "]" in string:
        return True
    else:
        return False


def string2typing(string):
    if is_sbrk(string):
        ll = string.find("[") + 1
        rr = string.rfind("]")
        args = tuple(
            [
                string2typing(sss)
                for sss in split_on_comma_not_in_bracket(string[ll:rr])
            ]
        )
        if len(args) == 1:
            args = args[0]
        Type = get_cls(string[: ll - 1])
        return Type[args]
    else:
        return get_cls(string)


def represent_inspect_parameter(dumper, data):
    out = {
        "name": data.name,
        "kind": data.kind.value,
        "default": data.default,
        "annotation": str(data.annotation),
    }
    return dumper.represent_mapping("!inspect.Parameter", out)


def construct_inspect_parameter(loader, node):
    dd = loader.construct_mapping(node)
    dd["annotation"] = string2typing(dd["annotation"])
    return inspect.Parameter(**dd)


yaml.add_representer(inspect.Parameter, represent_inspect_parameter)
yaml.add_constructor("!inspect.Parameter", construct_inspect_parameter)
