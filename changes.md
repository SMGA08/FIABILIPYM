# Changes made to fiabilipy (latest repo: fiabilipyg-2.8.1)

Goal: Modernize Python-2-era code to run on Python 3.9+ without changing public API or mathematical behavior, then repackage as a standalone `fiabilipym` distribution with a modern layout.

## Summary of edits

- Python 2 to 3 compatibility fixes (mechanical only).
- Package renamed and restructured into a standalone distribution.
- Tests updated to run on Python 3 without `unittest2`.
- Dependencies declared for visualization and numerical backends.

## File-by-file details (code compatibility)

### `src/fiabilipym/system.py`
- Imported `reduce` from `functools` so the existing `reduce(...)` calls work in Python 3.
- Replaced `collections.Iterable` with `collections.abc.Iterable` (Python 3 location).
- Replaced `xrange(...)` with `range(...)` in code and doctest examples.

Why: `xrange` and `collections.Iterable` are Python 2-only; `reduce` moved to `functools` in Python 3.

### `src/fiabilipym/voter.py`
- Replaced `xrange(...)` with `range(...)`.

Why: `xrange` is Python 2-only.

### `src/fiabilipym/markov.py`
- Replaced `xrange(...)` with `range(...)` in code and doctest examples.
- Updated numpy indexing line from `self.matrix[xrange(...), xrange(...)]` to `self.matrix[range(...), range(...)]`.

Why: `xrange` is Python 2-only; `range` is the Python 3 equivalent and works with numpy indexing.

### `tests/test_system.py`
- Replaced `dict.iteritems()` with `dict.items()`.
- Replaced `xrange(...)` with `range(...)`.
- Migrated to `unittest` instead of `unittest2`.

### `tests/test_markov.py`
- Replaced `dict.iteritems()` with `dict.items()`.
- Replaced `xrange(...)` with `range(...)`.
- Migrated to `unittest` instead of `unittest2`.

Why: `iteritems` and `xrange` are Python 2-only; Python 3 uses `items()` and `range()`.

## Packaging and structure changes (new)

- Renamed the importable package to `fiabilipym`.
- Moved package sources under `src/fiabilipym/`.
- Moved tests to `tests/` and updated imports to `fiabilipym`.
- Added modern `pyproject.toml` metadata and a standalone `setup.py`.
- Declared runtime dependencies: numpy, scipy, sympy, networkx, matplotlib, pygraphviz.
- Documented Graphviz requirements and uv-based workflows in `README.md`.

## Notes on behavior

- The computational behavior is unchanged; updates are mechanical and packaging-focused.
- Visualization continues to rely on matplotlib plus Graphviz layout via `pygraphviz`.

## 2026-01-29: Monte Carlo aging + Weibull distributions + notebook demo

- Added lifetime distribution support with a Weibull implementation (`distribution.py`).
- Components can now sample failure times, including aging via conditional residual life.
- Systems and voters expose Monte Carlo simulation (`monte_carlo`, `time_to_failure_sample`).
- Fixed graph construction to treat string nodes (e.g. `"N1"`) as single nodes.
- Added `Voter.draw()` for visualization parity with `System.draw()`.
- Added `fiabilipym_weibull_montecarlo_demo.ipynb` with architectures + plots.
- Added tests for Monte Carlo behavior, Weibull mean matching, and voter sampling.
- Exported Weibull helpers at the package root.
