"""
Console helpers.

We use `rich` for nicer console UX when available, but always keep a safe fallback
that works in minimal environments (pipes, CI logs, missing deps).
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import AbstractContextManager, nullcontext
from typing import Callable, Protocol, TextIO, runtime_checkable, cast


@runtime_checkable
class ConsoleLike(Protocol):
    """Minimal surface we rely on for console output."""

    def print(self, *objects: object, **kwargs: object) -> None: ...

    def rule(self, title: str | None = None) -> None: ...

    def status(self, status: str, *, spinner: str | None = None) -> AbstractContextManager[object]: ...


class PlainConsole:
    """Fallback console when `rich` is unavailable."""

    def print(self, *objects: object, **kwargs: object) -> None:
        """Print objects with optional formatting."""
        sep = cast(str, kwargs.get("sep", " "))
        end = cast(str, kwargs.get("end", "\n"))
        flush = bool(kwargs.get("flush", False))
        file_obj = kwargs.get("file", sys.stdout)
        try:
            # Cast to TextIO for type checker since kwargs are untyped
            file = cast(TextIO, file_obj) if file_obj is not None else sys.stdout
            # Best-effort compatibility with rich-style kwargs; ignore style/highlight.
            _ = kwargs.get("style", None)
            _ = kwargs.get("highlight", None)
            print(*(str(o) for o in objects), sep=sep, end=end, file=file, flush=flush)
        except (OSError, ValueError, TypeError):
            try:
                print(*(str(o) for o in objects), flush=True)
            except Exception:
                pass

    def rule(self, title: str | None = None) -> None:
        """Print a horizontal rule."""
        msg = f"--- {title} ---" if title else "---"
        self.print(msg)

    def status(self, status: str, *, spinner: str | None = None) -> AbstractContextManager[object]:
        """Return a context manager that does nothing."""
        _ = status
        _ = spinner
        return nullcontext()


class RichConsoleAdapter:
    """Adapter that makes `rich.console.Console` behave like our ConsoleLike.

    Why:
    - We want callsites to be able to pass `flush=True` like builtin `print()`.
    - Rich's `Console.print()` does not accept `flush`, so we strip it here.
    """

    def __init__(self, rich_console: object) -> None:
        self._c: object = rich_console

    def print(self, *objects: object, **kwargs: object) -> None:
        # Builtin-print style kwargs that rich doesn't accept.
        _ = kwargs.pop("flush", None)
        _ = kwargs.pop("file", None)

        fn_obj = getattr(self._c, "print", None)
        if not callable(fn_obj):
            return
        fn: Callable[..., object] = fn_obj
        try:
            _ = fn(*objects, **kwargs)
        except TypeError:
            # Fall back to a minimal call if kwargs mismatch the rich version.
            try:
                _ = fn(*objects)
            except Exception:
                pass
        except Exception:
            # Console output must never crash the program.
            pass

    def rule(self, title: str | None = None) -> None:
        fn_obj = getattr(self._c, "rule", None)
        if not callable(fn_obj):
            return
        fn = fn_obj
        try:
            _ = fn(title)
        except Exception:
            pass

    def status(self, status: str, *, spinner: str | None = None) -> AbstractContextManager[object]:
        # Only use live renderers when stdout is a TTY; otherwise keep output stable.
        if not rich_live_enabled():
            return nullcontext()
        fn_obj = getattr(self._c, "status", None)
        if not callable(fn_obj):
            return nullcontext()
        fn = fn_obj
        try:
            cm = fn(status, spinner=spinner)
            return cast(AbstractContextManager[object], cm)
        except Exception:
            return nullcontext()


def rich_enabled() -> bool:
    """Global toggle (for users/CI): set NO_RICH=1 to disable rich output."""
    v = str(os.environ.get("NO_RICH", "")).strip().lower()
    return v not in ("1", "true", "yes", "on")


def rich_live_enabled() -> bool:
    """Only use live renderers (progress/status) when stdout is a TTY."""
    try:
        return bool(rich_enabled() and sys.stdout.isatty())
    except (OSError, AttributeError):
        return False


# Module-level cache container (mutable list to avoid global statement)
_cached_console: list[ConsoleLike] = []


def get_console() -> ConsoleLike:
    """Return a cached console instance (rich when available, otherwise plain)."""
    if _cached_console:
        return _cached_console[0]

    if not rich_enabled():
        console = PlainConsole()
        _cached_console.append(console)
        return console

    try:
        mod = importlib.import_module("rich.console")
        console_class = getattr(mod, "Console", None)
        if not callable(console_class):
            raise ImportError("rich.console.Console not found")
    except ImportError:
        console = PlainConsole()
        _cached_console.append(console)
        return console

    try:
        # Let rich decide terminal behavior; it degrades gracefully in pipes.
        c = console_class()
        console = cast(ConsoleLike, RichConsoleAdapter(c))
        _cached_console.append(console)
        return console
    except (OSError, ValueError, TypeError):
        console = PlainConsole()
        _cached_console.append(console)
        return console


