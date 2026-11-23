"""Per-bank inventory shared by the AWC exporter and the directory WAV decoder."""

import errno
import json
import os


INVENTORY_NAME = ".awc-ogg.json"


def read_inventory(directory):
    path = os.path.join(directory, INVENTORY_NAME)
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return set(), set()
    except ValueError as exc:
        raise OSError(errno.EINVAL, "invalid AWC Ogg inventory", path) from exc

    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), list) and all(isinstance(name, str) for name in data[key])
        for key in ("known", "current")
    ):
        raise OSError(errno.EINVAL, "invalid AWC Ogg inventory", path)
    known, current = set(data["known"]), set(data["current"])
    if not current.issubset(known):
        raise OSError(errno.EINVAL, "invalid AWC Ogg inventory", path)
    return known, current


def inventory_bytes(known, current):
    data = {"known": sorted(known), "current": sorted(current)}
    return (json.dumps(data) + "\n").encode("utf-8")
