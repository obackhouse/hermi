"""Screening."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from typing import Iterable

    from numpy import float64

    from pymmd._typing import array
    from pymmd.basis import Basis, ShellPairView, ShellView

from pymmd.util import num_cartesian_functions, num_spherical_functions


class Screening[N: Literal[1, 2, 3, 4]](ABC):
    """Abstract base class for screening methods."""

    basis: Basis

    @overload
    def __call__(self: Screening[Literal[1]], views: ShellView) -> Iterable[ShellView]: ...
    @overload
    def __call__(self: Screening[Literal[2]], views: ShellPairView) -> Iterable[ShellPairView]: ...
    @overload
    def __call__(
        self: Screening[Literal[3]], views: tuple[ShellView, ShellPairView]
    ) -> Iterable[tuple[ShellView, ShellPairView]]: ...
    @overload
    def __call__(
        self: Screening[Literal[4]], views: tuple[ShellPairView, ShellPairView]
    ) -> Iterable[tuple[ShellPairView, ShellPairView]]: ...

    @abstractmethod
    def __call__[T: (ShellView, ShellPairView, tuple[ShellView, ShellPairView], tuple[ShellPairView, ShellPairView])](
        self, views: T
    ) -> Iterable[T]:
        """Yield (sub-)views that pass the screening criteria.

        Args:
            views: Views (over a shell, shell pair, shell triplet, or shell quartet) to be screened.

        Yields:
            Views that pass the screening criteria. In cases where the screening results in non-contiguous views,
            contiguous sub-views are yielded.
        """
        pass


class NoScreening[N: Literal[1, 2, 3, 4]](Screening[N]):
    """No screening; yield all views."""

    basis: Basis

    @overload
    def __call__(self: Screening[Literal[1]], views: ShellView) -> Iterable[ShellView]: ...
    @overload
    def __call__(self: Screening[Literal[2]], views: ShellPairView) -> Iterable[ShellPairView]: ...
    @overload
    def __call__(
        self: Screening[Literal[3]], views: tuple[ShellView, ShellPairView]
    ) -> Iterable[tuple[ShellView, ShellPairView]]: ...
    @overload
    def __call__(
        self: Screening[Literal[4]], views: tuple[ShellPairView, ShellPairView]
    ) -> Iterable[tuple[ShellPairView, ShellPairView]]: ...

    def __call__[T: (ShellView, ShellPairView, tuple[ShellView, ShellPairView], tuple[ShellPairView, ShellPairView])](
        self, views: T
    ) -> Iterable[T]:
        """Yield the input views unchanged."""
        yield views


@dataclass(frozen=True)
class AttenuationScreening(Screening[Literal[2]]):
    """Screening based on maximum shell attenuation."""

    basis: Basis
    threshold: float = 1e-3
    prefactor: float = 1.0

    def __call__(self, views: ShellPairView) -> Iterable[ShellPairView]:
        """Yield shell-pair views that pass the attenuation threshold."""
        coefficient = self.basis.shells.coefficient
        quantity = np.max(views.attenuation, axis=-1)
        quantity *= np.abs(np.multiply.outer(coefficient[views.indices_i], coefficient[views.indices_j]))

        starts_i = self.basis.contracted_shell_starts(views.indices_i)
        starts_j = self.basis.contracted_shell_starts(views.indices_j)
        quantity = np.maximum.reduceat(quantity, starts_i, axis=0)
        quantity = np.maximum.reduceat(quantity, starts_j, axis=1)

        mask = quantity > (self.threshold / self.prefactor)
        yield from views.iter_masked_views(mask)


@dataclass(frozen=True)
class CauchySchwarzScreening(Screening[Literal[4]]):
    """Screen using AO-pair diagonal ERIs and the Cauchy--Schwarz inequality.

    The metric contains ``(μν|μν)`` values in the selected AO convention.
    """

    basis: Basis
    metric: array[tuple[int, int], float64]
    threshold: float = 1e-6
    spherical: bool = False

    def _ao_slices(self, view: ShellPairView) -> tuple[slice, slice]:
        return view.spherical_ao_slices if self.spherical else view.cartesian_ao_slices

    def __call__(self, views: tuple[ShellPairView, ShellPairView]) -> Iterable[tuple[ShellPairView, ShellPairView]]:
        """Yield shell-pair view tuples that pass the Schwarz threshold."""
        num_functions = num_spherical_functions if self.spherical else num_cartesian_functions

        def shell_pair_metric(view: ShellPairView) -> array[tuple[int, int], float64]:
            ao_metric = self.metric[self._ao_slices(view)]
            ao_metric = ao_metric.reshape(
                view.shell_i.num_shells,
                num_functions(view.shell_i.angular_momentum),
                view.shell_j.num_shells,
                num_functions(view.shell_j.angular_momentum),
            )
            return np.maximum(np.max(ao_metric, axis=(1, 3)), 0)

        metric_ab = shell_pair_metric(views[0])
        metric_cd = shell_pair_metric(views[1])

        def single_pair_view(view: ShellPairView, index: tuple[int, int]) -> ShellPairView:
            mask = np.zeros((view.shell_i.num_shells, view.shell_j.num_shells), dtype=np.bool_)
            mask[index] = True
            return next(iter(view.iter_masked_views(mask)))

        def iter_pairs(
            view_ab: ShellPairView,
            view_cd: ShellPairView,
            metric_ab: array[tuple[int, int], float64],
            metric_cd: array[tuple[int, int], float64],
        ) -> Iterable[tuple[ShellPairView, ShellPairView]]:
            for i, j in np.ndindex(metric_ab.shape):
                if metric_ab[i, j] == 0:
                    continue
                pair_ab = single_pair_view(view_ab, (i, j))
                mask_cd = metric_ab[i, j] * metric_cd > self.threshold
                for pair_cd in view_cd.iter_masked_views(mask_cd):
                    yield pair_ab, pair_cd

        if metric_ab.size <= metric_cd.size:
            yield from iter_pairs(views[0], views[1], metric_ab, metric_cd)
        else:
            for pair_cd, pair_ab in iter_pairs(views[1], views[0], metric_cd, metric_ab):
                yield pair_ab, pair_cd
