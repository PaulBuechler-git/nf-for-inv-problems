import abc


class Operator:
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Operator has to be callable')
