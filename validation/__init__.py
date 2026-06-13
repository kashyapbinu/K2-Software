"""
K2 Aerospace — Physics Validation Suite
========================================

Benchmarks every analysis engine against an *independent* reference so the
physics can be trusted, not just internally consistent:

    Sim 6DOF      ↔  OpenRocket          (validation.sim)
    CFD aero      ↔  SU2  + Taylor–Maccoll exact cone flow   (validation.cfd)
    Structures    ↔  CalculiX + textbook closed form          (validation.structures)

Each domain exposes ``run_benchmarks()`` returning a list of
:class:`validation.harness.Benchmark`. The pytest gates in ``tests/validation``
assert each Benchmark passes; ``validation.report`` renders them to a markdown
report with overlay plots.

This is distinct from ``core.validation`` / ``structures.validation`` which only
check *internal* consistency and NASA-STD-7009A credibility tags. This package
checks agreement with *external* references.
"""
