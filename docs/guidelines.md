# Coding Guidelines

## General

1. Keep code modularized around clear ownership boundaries.
1. Split a function with several meaningful steps into helpers, each with one responsibility.
1. Spell names out in full: `note`, not `n`.
1. Give every semantic value a name — a `Final` constant, promoted to a shared module once the concept is reused.
1. Avoid the _tramp data_ antipattern: threading a value through functions that only pass it along.
1. Make the inputs logic depends on explicit. The parameters and configuration instances it relies on are required, not optional. Reserve default values for settings seldom changed (e.g. `seed`), and declare each such default as a top-level `Final` constant.
1. Derive booleans rather than storing them. A boolean computed from existing state belongs in a `@property` (or `@computed_field` on a Pydantic model), since a stored flag creates hidden state that drifts out of sync.
1. State type expectations explicitly, and reach attributes by direct access rather than dynamic `getattr` or `hasattr`.
1. Prefer protocols over inheritance.
1. Prefer `match` statements over long `isinstance` chains, and for enumeration handling.
1. Prefer `pathlib.Path` over `os.path`.
1. Separate function options with `*`, and choose positional arguments intentionally.
1. Change internal APIs, configs, and data shapes freely; preserve backward compatibility only when the user explicitly asks.
1. Run `pre-commit` on new files after each change.

## Shared Ownership

1. Put general-purpose, non-model-specific helpers in shared or common modules.
1. Search the repository with `rg` for existing logic before adding a helper.
1. When new code would duplicate existing logic, extract the shared rule first and route both call sites through it.
1. Import a shared helper straight from the module that implements it; a re-export or delegated-import module that exists only to route imports through is disallowed.
1. An `__init__` exposes only names from within its own tree hierarchy.
1. Give each module a single area of responsibility.
1. Prefer subpackages over a flat directory structure.
1. Isolate platform-, desktop-, or external-tool-specific behaviour behind a `Protocol` with one implementation per target, selected by a runtime factory that probes availability and environment. Callers depend only on the `Protocol` and stay platform-agnostic.
1. Wrap a third-party library or OS tool whose behaviour differs across platforms behind our own typed interface, and encode each quirk inside the matching implementation. A comment naming the third-party behaviour is warranted there.

## Type Hints

1. Type every function signature — parameters and return, including `None`.
1. Use fully parameterized, classic typing constructs: `Dict[str, int]` over a bare `Dict` or `dict[str, int]`, and `Optional[X]` over `X | None`.
1. Write type names unquoted, using `from __future__ import annotations` (only when needed), `Self`, or `TYPE_CHECKING`.
1. Reserve `Any` and `object` for boundaries that genuinely accept arbitrary data.
1. Cast or silence a type error only at an untyped or mistyped third-party boundary.
1. Type numpy arrays as `NDArray[np.float64]` (or the concrete dtype), never a bare `np.ndarray`, and state each array's shape where the one code comment allowance below permits.
1. Validate with `mypy`.

## Error Handling

1. Let a failure crash unless the code can recover from it meaningfully.
1. Handle errors at the execution boundary where possible.
1. Catch an exception only to recover from it; a `try`/`except` that repackages a failure without recovering adds nothing.
1. Bare `except` and `except Exception` are forbidden.
1. Scope each `try` to the statements that can actually fail, absent a specific reason to widen it.

## Models

1. Prefer Pydantic models for validated or serialized data.
1. Freeze models whose instances stay constant after construction (`frozen=True`).
1. Use a dataclass for a small internal state object that needs neither serialization nor validation, test-case dataclasses included.

## Docstrings and Comments

1. A docstring explains the intention of a class or function and the context of its use.
1. State functionality in positive terms. Describe what a class or function *does* — not what it avoids, omits, skips, differs from, or no longer does. Reframe every negation ("does not", "rather than", "instead of", "without", "never", "cannot", "no longer") into the behaviour that actually happens. Do not contrast with rejected alternatives as justification; the positive statement carries the meaning.
1. Negative phrasing is allowed only where the condition itself is the contract: exception triggers in `Raises:` clauses, precondition/postcondition bounds (prefer "must be at least X" over "cannot be less than X" where natural), and documented edge-case returns. Outside these concrete cases, negative descriptions are information noise and must be removed.
1. Justify an arbitrary choice in the docstring rather than a code comment, and frame the justification by what the choice achieves.
1. Let clear names carry the meaning, and skip comments or docstrings that restate the code.
1. Avoid code comments; they are warranted for tensor shapes, third-party API quirks, or non-obvious invariants.
1. Code comments and docstrings are not for recording changes or progress.
1. Don't write module docstrings.

## Documents

1. A document under `docs/` explains a subsystem to someone about to change it. Open by stating what it governs and when to consult it, so a reader learns in one paragraph whether they are in the right place.
1. Lead with principles, then mechanics. A principle is a design truth you reason from; state the principles first, and let concrete conventions and reference tables follow as the way each principle is realized.
1. Keep principles, conventions, and descriptions distinct. A principle is a reason; a convention is a handy mechanic that serves it; a description is a fact about how something works. A convention promoted to a principle, or a principle buried in a description, misleads the reader about what is load-bearing.
1. Prefer a few strong principles to many narrow rules. When several rules are facets of one idea, state the idea once and derive them. A growing list of ad-hoc rules signals a principle that has gone unstated.
1. State the design in positive terms, as it stands today. This is the docstring rule above applied to prose: describe what the design is and does, not what it avoids, omits, or once was.
1. Write for a reader who never saw the history. A document is not a changelog or a devlog: do not argue against past states, resolved problems, or rejected alternatives the reader never knew existed. The design as it stands carries its own justification; history belongs in commit messages and release notes.
1. Reach for a negative example only when the contrast teaches something the positive statement cannot, and use it sparingly. One well-placed "what to avoid" illuminates; a document written mostly in negatives is noise.
1. State each fact once, in the document that owns it, and cross-reference sibling documents rather than repeating them.
1. Ground claims in concrete referents — name the file, type, or path — and justify an arbitrary choice by what it achieves.

## Tests

1. A test file mirrors the ownership of the code it exercises.
1. When functionality moves between packages, move its direct unit tests in the same change.
1. Parametrize tests that share a body, using a test-case dataclass.
1. For a multi-step scenario, use a test-scenario suite class — a series of functions with assertions.
1. Prefer fixtures over factories, and define shared fixtures in an appropriate place.
1. Do not assert default values of configurations, layouts, settings, and similar. Defaults are not contracts, and pinning them overconstrains the tests. Test behavior instead: validation bounds, serialization round-trips, and invariants. The exception is when values must match by contract rather than equal a chosen constant — e.g. project metadata at creation or after a save/load round-trip should be asserted to match, never hardcoded to a version string.
1. Unit tests may mock system boundaries (file I/O, external services, IPC channels), but must not mock the domain logic that is the subject of the test. Integration tests must exercise real computation pipelines against real (synthetically built) data.
1. When a test expectation diverges from the production code's actual behaviour, determine which is wrong before acting. A failing test is evidence of a potential bug in the production code unless the test itself is demonstrably incorrect (wrong imports, misread API contract, incorrect fixture). Never silently delete or weaken a test to make it pass. If uncertain, flag the divergence explicitly and ask before changing either side.
