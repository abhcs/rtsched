from fractions import Fraction
from typing import Optional


class Task:
    """This class implements a hard real-time task.

    Attributes:
        wcet: worst-case execution time
        period: minimum duration between successive request arrivals
        deadline: maximum duration between request arrival and response
        jitter: maximum duration between request arrival and the task becoming
                eligible for execution

    We assume that wcet, period, and deadline are positive integers, and jitter
    and offset are nonnegative integers. By default, deadline is equal to
    period and jitter is zero. We assume that wcet + jitter <= deadline.

    """

    def __init__(self, wcet: int, period: int, deadline: Optional[int] = None,
                 jitter: int = 0) -> None:
        """Constructs a hard real-time task.

        Examples
        --------

        >>> Task(1, 2)
        Task(1, 2, 2, 0)

        >>> Task(wcet = 1, period = 2, jitter = 1)
        Task(1, 2, 2, 1)
        """
        if deadline is None: # implicit deadline
            deadline = period
        assert wcet > 0 and isinstance(wcet, int)
        assert period > 0 and isinstance(period, int)
        assert deadline > 0 and isinstance(deadline, int)
        assert jitter >= 0 and isinstance(jitter, int)
        assert wcet + jitter <= deadline
        self.wcet = wcet
        self.period = period
        self.deadline = deadline
        self.jitter = jitter

    @property
    def utilization(self) -> Fraction:
        """The utilization of the task.

        >>> Task(1, 2).utilization
        Fraction(1, 2)
        """
        return Fraction(self.wcet, self.period)

    @property
    def dj(self) -> int:
        return self.deadline - self.jitter

    def __repr__(self):
        """repr(self)"""
        return (f'{self.__class__.__name__}({self.wcet}, '
                f'{self.period}, {self.deadline}, '
                f'{self.jitter})')

    def __str__(self):
        """str(self)"""
        return (f'(wcet: {self.wcet}, period: {self.period}, '
                f'deadline: {self.deadline}, jitter: {self.jitter})')
