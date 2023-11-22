import abc

from torch import Tensor


class Regulariser:
    """
    Abstract class for a regulariser. This abstract class can be used to build
    a custom regulariser for the variational model.
    """

    @abc.abstractmethod
    def evaluate(self, x: Tensor):
        """Method to evaluate the regulariser for an input x"""
        raise NotImplementedError('Evaluate not implemented')

    @abc.abstractmethod
    def get_hparams(self):
        """Method that should return the hyperparameters fo the regulariser"""
        raise NotImplementedError('Regulariser should return the hyperparameters with this function')
