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
import hashlib
import json
import os
from typing import Any, Mapping

__all__ = [
    "DEFAULTS",
    "ALIASES",
    "load_config",
    "normalize_config",
    "validate_config",
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
        "satellite_orbit": (6e5, "Circular-orbit altitude (m); sets the slew-induced wind."),
        "satellite_zenith": (0.0, "Zenith angle (deg) at which the slew rate is evaluated."),
    },
}

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
    "orbit": ("channel_params", "satellite_orbit"),
    "satellite_orbit": ("channel_params", "satellite_orbit"),
    "zenith": ("channel_params", "satellite_zenith"),
    "satellite_zenith": ("channel_params", "satellite_zenith"),
}

_INT_KEYS = {"num_steps", "phase_screen_size", "num_layers", "seed"}


def _unwrap(v):
    if isinstance(v, Mapping) and "value" in v:
        return v["value"]
    return v


def _coerce(key, value):
    if value is None:
        return None
    if key in _INT_KEYS:
        f = float(value)
        if not f.is_integer():
            raise ValueError(f"{key} must be an integer, got {value!r}")
        return int(f)
    if isinstance(value, bool):
        return value
    return float(value)


def normalize_config(config: Mapping) -> dict:
    """Return a new config in ``{"value": ...}`` form with defaults filled in.

    Unknown sections (e.g. ``pass_params``, ``dataset``) are kept with plain
    values. Legacy AtmoTurbSim sections such as ``save_data`` are ignored.
    """
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

    passthrough = {"pass_params", "dataset", "analysis"}
    for section in passthrough:
        if section in config and config[section] is not None:
            out[section] = {k: _unwrap(v) for k, v in config[section].items()}
    if "pass_params" in out:
        merged = dict(PASS_DEFAULTS)
        merged.update(out["pass_params"])
        out["pass_params"] = merged
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
    for key in ("pixel_size", "total_r0", "wavefront_wavelength", "L0", "satellite_orbit"):
        if not ch[key]["value"] > 0:
            raise ValueError(f"{key} must be > 0, got {ch[key]['value']}")
    if not sim["time_per_tick"]["value"] > 0:
        raise ValueError("time_per_tick must be > 0")
    if ch["ground_wind_speed"]["value"] < 0:
        raise ValueError("ground_wind_speed must be >= 0")
    z = ch["satellite_zenith"]["value"]
    if not 0 <= z < 90:
        raise ValueError(f"satellite_zenith must be in [0, 90) deg, got {z}")
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
        return section, name
    raise KeyError(f"Unknown parameter {key!r}. Known: {sorted(ALIASES)}")


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
    source : str, os.PathLike or Mapping
        Path to a JSON file or an already-loaded dict.
    overrides : Mapping, optional
        ``{alias_or_path: value}`` applied after loading, e.g.
        ``{"r0": 0.1, "steps": 500}``.
    """
    if isinstance(source, (str, os.PathLike)):
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
