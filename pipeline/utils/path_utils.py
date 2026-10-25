"""Portable path helpers for deeply nested pipeline outputs."""

import os


def normalize_output_root(path):
    """Return an absolute output root, using Windows extended-length syntax.

    Pipeline stages preserve detailed configuration names in a deep directory
    tree. On Windows, an ordinary absolute path can therefore exceed the legacy
    ``MAX_PATH`` limit before all stages are written. The extended-length prefix
    avoids that limit without changing component names or requiring an OS
    setting. UNC paths use the corresponding extended UNC form; other platforms
    receive a normal absolute path.
    """
    raw_path = os.fspath(path)
    if not raw_path:
        raise ValueError("output root must not be empty")

    if os.name == "nt" and raw_path.startswith("\\\\?\\"):
        return os.path.normpath(raw_path)

    absolute = os.path.abspath(os.path.expanduser(raw_path))
    if os.name != "nt":
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute
