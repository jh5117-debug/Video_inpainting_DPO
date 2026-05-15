"""Optional project-wide process title hook.

Python imports ``sitecustomize`` automatically when this repository is on
``PYTHONPATH``. GPU launch scripts set WORLDMODELPHY_PROCESS_NAME so Python,
Accelerate, and torch distributed workers show a consistent process name in
process monitors without renaming repo paths or config keys.
"""

from __future__ import annotations

import os


def _set_process_title() -> None:
    title = (
        os.environ.get("WORLDMODELPHY_PROCESS_NAME")
        or os.environ.get("PROCESS_TITLE")
        or ""
    ).strip()
    if not title:
        return

    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        # The /proc fallback still updates the short Linux comm name used by
        # many process monitors. It is limited to 15 visible characters.
        pass

    if os.name == "posix":
        try:
            with open("/proc/self/comm", "w", encoding="utf-8") as handle:
                handle.write(title[:15])
        except Exception:
            pass


_set_process_title()
