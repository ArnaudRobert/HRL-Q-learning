from abc import ABC, abstractmethod

class HL(ABC):
    """
    Base class with minimal requirements for all
    high-level agent impleentations.
    """

    @abstractmethod
    def select_subgoal(self):
        pass

    @abstractmethod
    def update(self, transition):
        pass

    @abstractmethod
    def get_V(self, goal):
        pass


class LL(ABC):
    """
    Base class with minimal requirements for all
    low-level agent implementations.
    """
    @abstractmethod
    def select_action(self, state, goal, greedy=False):
        pass

    @abstractmethod
    def update(self, transition):
        pass

    @abstractmethod
    def get_V(self, goal):
        pass
