from dataclasses import dataclass


@dataclass(frozen=True)
class CurvatureGrid:
    values: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)

