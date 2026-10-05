"""Configuration loading, defaults and validation.

OPAL configs are JSON files with two required sections and one optional one::

    {
      "simulation_params": {"num_steps": 1000, "time_per_tick": 1e-3, "seed": 1},
      "channel_params":    {"phase_screen_size": 512, "pixel_size": 0.01, ...},
      "pass_params":       {...}    # optional, used by `opal pass`
    }

Each parameter may be a plain value or a ``{"value": ..., "description": ...}``
object. Internally everything is normalised to the ``{"value": ...}`` form.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
from typing import Any, Mapping

__all__ = [
    "DEFAULTS",
    "PASS_DEFAULTS",
    "ALIASES",
    "RENAMED_KEYS",
    "load_config",
    "normalize_config",
    "validate_config",
    "rename_legacy_keys",
    "to_plain",
    "get_value",
    "set_value",
    "apply_overrides",
    "config_hash",
    "save_config",
]


DEFAULTS: dict[str, dict[str, tuple[Any, str]]] = {
    "simulation_params": {
        "num_steps": (1000, "Number of simulation ticks."),
        "time_per_tick": (1e-3, "Duration of one tick (s)."),
        "seed": (None, "Master seed; per-layer seeds are derived from it. null = random."),
    },
    "channel_params": {
        "phase_screen_size": (512, "Grid size N (pixels, even)."),
        "pixel_size": (0.01, "Grid spacing (m)."),
        "num_layers": (10, "Number of phase-screen layers."),
        "total_r0": (0.2, "Total path Fried parameter r0 (m)."),
        "wavefront_wavelength": (1.55e-6, "Optical wavelength (m)."),
        "L0": (25.0, "Von Karman outer scale (m)."),
        "ground_wind_speed": (5.0, "Ground wind speed in the Bufton model (m/s)."),
        "satellite_orbit_altitude": (6e5, "Circular-orbit altitude of the satellite (m); sets the "
                                          "slew-induced wind."),
        "satellite_zenith_angle": (0.0, "Zenith angle of the line of sight (deg, 0 = overhead) at which "
                                        "the satellite's slew rate (apparent wind) is evaluated. It does "
                                        "not rescale total_r0 or the propagation distances."),
    },
}

# Keys renamed in OPAL 1.1. Old names are rejected with a pointer to the new one.
RENAMED_KEYS: dict[str, str] = {
    "satellite_orbit": "satellite_orbit_altitude",
    "satellite_zenith": "satellite_zenith_angle",
}

# Top-level sections that are accepted and ignored (legacy AtmoTurbSim files).
IGNORED_SECTIONS = {"save_data"}

PASS_DEFAULTS: dict[str, Any] = {
    "altitude_m": None,
    "min_elevation_deg": 20.0,
    "num_segments": 30,
    "pass_num_samples": 1024,
    "pass_window_s": None,
    "wavefront_captures": 200,
    "capture_to_host": True,
    "regenerate_on_switch": False,
    "match_initial_r0": True,
    "validation_guard_frac": 0.0,
}

# Short names accepted by overrides, the CLI and the dataset sweeps.
ALIASES: dict[str, tuple[str, str]] = {
    "steps": ("simulation_params", "num_steps"),
    "num_steps": ("simulation_params", "num_steps"),
    "dt": ("simulation_params", "time_per_tick"),
    "time_per_tick": ("simulation_params", "time_per_tick"),
    "seed": ("simulation_params", "seed"),
    "N": ("channel_params", "phase_screen_size"),
    "grid": ("channel_params", "phase_screen_size"),
    "phase_screen_size": ("channel_params", "phase_screen_size"),
    "dx": ("channel_params", "pixel_size"),
    "pixel_size": ("channel_params", "pixel_size"),
    "layers": ("channel_params", "num_layers"),
    "num_layers": ("channel_params", "num_layers"),
    "r0": ("channel_params", "total_r0"),
    "total_r0": ("channel_params", "total_r0"),
    "wavelength": ("channel_params", "wavefront_wavelength"),
    "wvl": ("channel_params", "wavefront_wavelength"),
    "wavefront_wavelength": ("channel_params", "wavefront_wavelength"),
    "L0": ("channel_params", "L0"),
    "wind": ("channel_params", "ground_wind_speed"),
    "ground_wind_speed": ("channel_params", "ground_wind_speed"),
    "orbit": ("channel_params", "satellite_orbit_altitude"),
    "satellite_orbit_altitude": ("channel_params", "satellite_orbit_altitude"),
    "zenith": ("channel_params", "satellite_zenith_angle"),
    "satellite_zenith_angle": ("channel_params", "satellite_zenith_angle"),
}

_INT_KEYS = {"num_steps", "phase_screen_size", "num_layers", "seed"}


def _unwrap(v):
    if isinstance(v, Mapping) and "value" in v:
        return v["value"]
    return v


def _coerce(key, value):
    if value is None:
        return None
    if isinstance(value, bool):
        if key in _INT_KEYS:
            raise ValueError(f"{key} must be an integer, got {value!r}")
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}") from None
    if key in _INT_KEYS:
        if not f.is_integer():
            raise ValueError(f"{key} must be an integer, got {value!r}")
        return int(f)
    return f


def _unknown_key_error(where, key, known):
    path = f"{where}{key}"
    if key in RENAMED_KEYS:
        return ValueError(f"{path!r} was renamed to {where + RENAMED_KEYS[key]!r} in OPAL 1.1; "
                          "please update your config")
    hint = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
    if hint:
        return ValueError(f"Unknown parameter {path!r}; did you mean {where + hint[0]!r}?")
    return ValueError(f"Unknown parameter {path!r}. Known: {', '.join(sorted(known))}")


def _check_keys(config: Mapping) -> None:
    """Reject misspelled sections and parameters instead of silently ignoring them."""
    known_sections = set(DEFAULTS) | _PASSTHROUGH
    for section in config:
        if section in known_sections or section in IGNORED_SECTIONS or str(section).startswith("_"):
            continue
        hint = difflib.get_close_matches(str(section), sorted(known_sections), n=1, cutoff=0.6)
        raise ValueError(f"Unknown config section {section!r}"
                         + (f"; did you mean {hint[0]!r}?" if hint else
                            f". Known: {', '.join(sorted(known_sections))}"))
    for section, params in DEFAULTS.items():
        for key in (config.get(section) or {}):
            if key not in params and not str(key).startswith("_"):
                raise _unknown_key_error(f"{section}.", key, params)
    for key in (config.get("pass_params") or {}):
        if key not in PASS_DEFAULTS and not str(key).startswith("_"):
            raise _unknown_key_error("pass_params.", key, PASS_DEFAULTS)


_PASSTHROUGH = {"pass_params", "dataset", "analysis"}


def normalize_config(config: Mapping) -> dict:
    """Return a new config in ``{"value": ...}`` form with defaults filled in.

    ``pass_params``, ``dataset`` and ``analysis`` are kept with plain values.
    Unknown sections or parameters raise ``ValueError`` (with a "did you mean"
    hint); keys starting with ``_`` (comments) and the legacy AtmoTurbSim
    ``save_data`` section are ignored.
    """
    _check_keys(config)
    out: dict = {}
    for section, params in DEFAULTS.items():
        src = config.get(section, {}) or {}
        sec_out = {}
        for key, (default, desc) in params.items():
            if key in src:
                raw = src[key]
                description = raw.get("description", desc) if isinstance(raw, Mapping) else desc
                value = _coerce(key, _unwrap(raw))
            else:
                description = desc
                value = default
            sec_out[key] = {"value": value, "description": description}
        out[section] = sec_out

    for section in _PASSTHROUGH:
        if section in config and config[section] is not None:
            out[section] = {k: _unwrap(v) for k, v in config[section].items()
                            if not str(k).startswith("_")}
    if "pass_params" in out:
        merged = dict(PASS_DEFAULTS)
        merged.update(out["pass_params"])
        out["pass_params"] = merged
    return out


def rename_legacy_keys(config: Mapping) -> dict:
    """Copy of a stored (plain or ``{"value"}``) config with pre-1.1 key names updated.

    Used when reading manifests and exports written by OPAL 1.0. User configs
    are not migrated silently; :func:`load_config` rejects the old names.
    """
    out = copy.deepcopy(dict(config))
    ch = out.get("channel_params")
    if isinstance(ch, dict):
        for old, new in RENAMED_KEYS.items():
            if old in ch and new not in ch:
                ch[new] = ch.pop(old)
    return out


def validate_config(config: Mapping) -> None:
    """Raise ``ValueError`` if a normalised config is physically invalid."""
    sim = config["simulation_params"]
    ch = config["channel_params"]
    N = ch["phase_screen_size"]["value"]
    if N < 16 or N % 2:
        raise ValueError(f"phase_screen_size must be an even integer >= 16, got {N}")
    if ch["num_layers"]["value"] < 1:
        raise ValueError("num_layers must be >= 1")
    if sim["num_steps"]["value"] < 1:
        raise ValueError("num_steps must be >= 1")
    for key in ("pixel_size", "total_r0", "wavefront_wavelength", "L0", "satellite_orbit_altitude"):
        if not ch[key]["value"] > 0:
            raise ValueError(f"{key} must be > 0, got {ch[key]['value']}")
    if not sim["time_per_tick"]["value"] > 0:
        raise ValueError("time_per_tick must be > 0")
    if ch["ground_wind_speed"]["value"] < 0:
        raise ValueError("ground_wind_speed must be >= 0")
    z = ch["satellite_zenith_angle"]["value"]
    if not 0 <= z < 90:
        raise ValueError(f"satellite_zenith_angle must be in [0, 90) deg, got {z}")
    seed = sim["seed"]["value"]
    if seed is not None and not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative integer")


def get_value(config: Mapping, key: str):
    """Read a parameter by alias (``"r0"``) or dotted path (``"channel_params.total_r0"``)."""
    section, name = _resolve(key)
    entry = config[section][name]
    return _unwrap(entry)


def set_value(config: dict, key: str, value) -> dict:
    """Set a parameter in place by alias or dotted path. Returns ``config``."""
    section, name = _resolve(key)
    if section in DEFAULTS:
        config.setdefault(section, {})
        entry = config[section].get(name)
        coerced = _coerce(name, value)
        if isinstance(entry, dict):
            entry["value"] = coerced
        else:
            config[section][name] = {"value": coerced, "description": DEFAULTS[section][name][1]}
    else:
        config.setdefault(section, {})[name] = value
    return config


def _resolve(key: str) -> tuple[str, str]:
    if key in ALIASES:
        return ALIASES[key]
    if "." in key:
        section, name = key.split(".", 1)
        if section in DEFAULTS and name not in DEFAULTS[section]:
            raise KeyError(str(_unknown_key_error(f"{section}.", name, DEFAULTS[section])))
        if section == "pass_params" and name not in PASS_DEFAULTS:
            raise KeyError(str(_unknown_key_error("pass_params.", name, PASS_DEFAULTS)))
        return section, name
    if key in RENAMED_KEYS:
        raise KeyError(str(_unknown_key_error("", key, ALIASES)))
    hint = difflib.get_close_matches(key, sorted(ALIASES), n=1, cutoff=0.6)
    raise KeyError(f"Unknown parameter {key!r}"
                   + (f"; did you mean {hint[0]!r}?" if hint else f". Known: {sorted(ALIASES)}"))


def apply_overrides(config: dict, overrides: Mapping | None) -> dict:
    """Apply ``{alias_or_path: value}`` overrides in place."""
    for k, v in (overrides or {}).items():
        if v is not None:
            set_value(config, k, v)
    return config


def load_config(source, overrides: Mapping | None = None, validate: bool = True) -> dict:
    """Load, normalise and validate a configuration.

    Parameters
    ----------
    source : str, os.PathLike, Mapping or None
        Path to a JSON file, an already-loaded dict, or ``None`` for the
        built-in defaults.
    overrides : Mapping, optional
        ``{alias_or_path: value}`` applied after loading, e.g.
        ``{"r0": 0.1, "steps": 500}``.
    """
    if source is None:
        raw = {}
    elif isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raw = copy.deepcopy(dict(source))
    config = normalize_config(raw)
    apply_overrides(config, overrides)
    if validate:
        validate_config(config)
    return config


def to_plain(config: Mapping) -> dict:
    """Strip descriptions: ``{section: {key: value}}``."""
    return {
        section: {k: _unwrap(v) for k, v in params.items()}
        for section, params in config.items()
        if isinstance(params, Mapping)
    }


def config_hash(config: Mapping, length: int = 12) -> str:
    """Stable short hash of the physical parameters (seed and step count excluded)."""
    plain = to_plain(config)
    physics = {
        "channel_params": plain.get("channel_params", {}),
        "time_per_tick": plain.get("simulation_params", {}).get("time_per_tick"),
    }
    blob = json.dumps(physics, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def save_config(config: Mapping, path) -> None:
    """Write a config as plain-valued JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_plain(config), fh, indent=2)
