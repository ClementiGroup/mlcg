def pytest_addoption(parser):
    parser.addoption(
        "--runner_idx",
        default=0,
        type=int,
        help="Description of my_option",  # help message
    )

    parser.addoption(
        "--num_containers",
        default=1,
        type=int,
        help="Description of my_option",  # help message
    )
