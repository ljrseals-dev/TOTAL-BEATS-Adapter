# Python Import and Optional Dependency Policy

## Rule

Fake, placeholder, and silently ignored imports are prohibited. If a code path
requires a dependency, importing or invoking that path must fail with a clear,
actionable error when the dependency is unavailable. Never hide a broken
environment with `except ImportError: pass`, a fabricated module, or a
placeholder object that is later used as though the dependency loaded.

## Required dependencies

- Declare runtime requirements in `pyproject.toml`.
- Install them in every CI job that imports or exercises the corresponding
  runtime path.
- Let a genuinely required import fail, or translate its `ImportError` into a
  domain-specific exception that names the missing dependency and installation
  extra. Preserve the original exception with `raise ... from exc`.
- Do not catch broad exceptions around imports.

## Optional capabilities

Optional integrations may be lazy-loaded so users of the base adapter do not
need heavyweight extras. The boundary must remain explicit:

- Import the optional package only when its capability is requested.
- Raise a clear, actionable exception if it is absent.
- Name the correct extra, for example `pip install -e ".[neural]"` or
  `pip install -e ".[train]"`.
- Do not silently switch to demo defaults, fabricated inference, or a reduced
  implementation when an optional dependency is missing.
- If a capability flag is exposed, it must reflect the actual import result;
  requesting an unavailable capability must still raise an explicit error.

For TOTAL-BEATS-Adapter, NumPy is a base dependency. PyTorch is optional:
controller inference uses the `neural` extra and controller training uses the
`train` extra. The adapter's explicit safe-default demo mode is not a
replacement for a missing neural controller.

## Tests and CI

- Tests must use real installed packages or deliberate, scoped mocks.
- Never insert fake modules into `sys.modules` merely to make imports succeed.
- Tests for missing-dependency behavior may deliberately intercept the import
  and raise `ImportError`; they must assert the actionable error and preserved
  cause.
- CI jobs for the base adapter install `.[dev]`. Jobs that exercise controller
  inference or training install the relevant extras and fail if installation,
  import, or execution fails.
- Do not use `continue-on-error` for required dependency installation or
  required import/integration checks.

## Review checklist

Before accepting an import-related change, verify:

1. Is the dependency required or optional for this exact code path?
2. Is it declared in the correct project dependency group/extra?
3. Does a missing dependency produce a direct, actionable error?
4. Is any fallback explicit, documented, and semantically honest?
5. Do tests cover both the installed and missing-dependency paths where
   appropriate?
6. Does CI install and exercise the dependencies required by its job?

