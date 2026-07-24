"""Privacy-safe voice routing from sanitized workspace-copilot context."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HARNESS_COMMANDS = frozenset({"codex", "claude", "omp"})


@dataclass(frozen=True, slots=True)
class Route:
    project: str
    cwd: Path
    pane_id: str
    session: str
    window: str
    active: bool = False


@dataclass(frozen=True, slots=True)
class Resolution:
    route: Route | None
    reason: str
    candidates: tuple[Route, ...] = ()


def load_routes(path: Path) -> dict[str, Path]:
    value = json.loads(path.read_text())
    routes = value.get("projects", value)
    return {str(project): Path(cwd) for project, cwd in routes.items()}


def _candidates(context: dict[str, Any], routes: dict[str, Path]) -> list[Route]:
    """Every live harness pane in a routed project, focused or not.

    Requiring pane-level `active` here made auto-attach impossible whenever a
    dashboard or build pane held the cursor inside the harness window; pane
    activity is now only a tiebreak inside one window.
    """
    result: list[Route] = []
    for session in context.get("tmux", ()):
        session_id = str(session.get("session", ""))
        for window in session.get("windows", ()):
            window_id = str(window.get("index", ""))
            for pane in window.get("panes", ()):
                project = str(pane.get("project", ""))
                if (
                    not pane.get("dead")
                    and pane.get("command") in HARNESS_COMMANDS
                    and project in routes
                ):
                    result.append(
                        Route(
                            project=project,
                            cwd=routes[project],
                            pane_id=str(pane.get("id", "")),
                            session=session_id,
                            window=window_id,
                            active=bool(pane.get("active")),
                        )
                    )
    return result


def _narrow(matches: list[Route]) -> Route | None:
    """One harness, or the single pane-active harness among several."""
    if len(matches) == 1:
        return matches[0]
    pane_active = [route for route in matches if route.active]
    if len(pane_active) == 1:
        return pane_active[0]
    return None


def resolve_context(
    context: dict[str, Any],
    routes: dict[str, Path],
) -> Resolution:
    candidates = _candidates(context, routes)
    if not candidates:
        return Resolution(None, "no_active_harness")
    focus = context.get("tmux_focus") or {}
    focused_pane = str(focus.get("pane_id", ""))
    if focused_pane:
        matches = [route for route in candidates if route.pane_id == focused_pane]
        if len(matches) == 1:
            return Resolution(matches[0], "focused_pane", tuple(candidates))
        focused_window = (str(focus.get("session", "")), str(focus.get("window", "")))
        if all(focused_window):
            matches = [
                route
                for route in candidates
                if (route.session, route.window) == focused_window
            ]
            if matches:
                # The user is looking at this window. Never reroute a spoken
                # turn to a background window when its harness is ambiguous.
                route = _narrow(matches)
                if route:
                    return Resolution(route, "focused_window", tuple(candidates))
                return Resolution(
                    None, "ambiguous_active_windows", tuple(matches)
                )

    # Forward-compatible with a privacy-safe `active` bit on the window.
    active_windows: set[tuple[str, str]] = set()
    for session in context.get("tmux", ()):
        for window in session.get("windows", ()):
            if window.get("active"):
                active_windows.add(
                    (str(session.get("session", "")), str(window.get("index", "")))
                )
    if active_windows:
        matches = [
            route
            for route in candidates
            if (route.session, route.window) in active_windows
        ]
        if matches:
            route = _narrow(matches)
            if route:
                return Resolution(route, "active_window", tuple(candidates))
            return Resolution(None, "ambiguous_active_windows", tuple(matches))

    route = _narrow(candidates)
    if route:
        return Resolution(route, "unique_active_harness", tuple(candidates))
    return Resolution(None, "ambiguous_active_windows", tuple(candidates))


class WorkspaceRouter:
    def __init__(
        self,
        routes: dict[str, Path],
        *,
        command: tuple[str, ...] = ("workspace-copilot", "--json", "context"),
        timeout: float = 3.0,
    ):
        self.routes = routes
        self.command = command
        self.timeout = timeout

    def resolve(self) -> Resolution:
        output = subprocess.run(
            self.command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return resolve_context(json.loads(output.stdout), self.routes)

