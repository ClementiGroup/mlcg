class _Prior(object):
    """Abstract prior class"""

    def __init__(self) -> None:
        super(_Prior, self).__init__()

    @staticmethod
    def fit_from_values():
        raise NotImplementedError

    @staticmethod
    def data2features():
        raise NotImplementedError

    @staticmethod
    def data2parameters():
        raise NotImplementedError
