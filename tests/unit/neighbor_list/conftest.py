import itertools

import pytest

from nl_testing import CASES, DEVICES, DTYPES, NL_METHODS, supports


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nl_methods(predicate): run only on the neighbour-list methods "
        "for which predicate(method) is true",
    )


def pytest_generate_tests(metafunc):
    """A test taking ``nl_method`` or ``case`` runs on every combination
    of method, case, self_interaction, device and dtype it asks for,
    except those the method does not support."""
    if not {"nl_method", "case"} & set(metafunc.fixturenames):
        return
    marker = metafunc.definition.get_closest_marker("nl_methods")
    axes = {
        "nl_method": [
            (m, m.name)
            for m in NL_METHODS
            if marker is None or marker.args[0](m)
        ],
        "case": [(c, c.id) for c in CASES],
        "self_interaction": [(False, "False"), (True, "True")],
        "device": [(d, d) for d in DEVICES],
        "dtype": [(t, n) for n, t in DTYPES.items()],
    }
    names = [n for n in axes if n in metafunc.fixturenames]
    params = []
    for combo in itertools.product(*(axes[n] for n in names)):
        values = dict(zip(names, (value for value, _ in combo)))
        method = values.get("nl_method")
        if method is not None and not supports(**values):
            continue
        missing = method is not None and method.fn is None
        params.append(
            pytest.param(
                *values.values(),
                id="-".join(i for _, i in combo),
                marks=(
                    pytest.mark.skip(f"{method.name} not installed")
                    if missing
                    else ()
                ),
            )
        )
    metafunc.parametrize(names, params)
