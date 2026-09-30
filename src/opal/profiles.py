"""Turbulence and wind profiles along the slant path.

The Hufnagel-Valley (HV) Cn^2 model is anchored to a requested total r0 by
solving for its ground-layer strength ``A``. The path is then sliced into
layers that each carry an equal share of the path-weighted turbulence
``int Cn^2(h) h^(5/6) dh`` (i.e. equal contributions to the scintillation
index), and every layer receives its own r0 and Bufton wind speed including
the apparent satellite slew ``omega_s * h``.
"""

import numpy as np
import scipy.integrate as integrate
from scipy import optimize
from scipy.integrate import quad
from scipy.optimize import fsolve

__all__ = [
    "solve_for_A",
    "circ_orbit_geo",
    "calc_r0_profile",
    "calculate_number_of_extrusions",
]


def solve_for_A(k, w, delta_z, r0_pw_given):
    """Solve for the HV ground-layer strength ``A`` that yields a plane-wave r0.

    Parameters
    ----------
    k : float
        Optical wavenumber 2 pi / lambda.
    w : float
        RMS upper-atmosphere wind speed (m/s).
    delta_z : float
        Upper integration limit (m).
    r0_pw_given : float
        Target plane-wave Fried parameter (m).
    """

    def Cn2(h, A):
        term1 = 0.00594 * (w / 27) ** 2 * (10**-5 * h) ** 10 * np.exp(-h / 1000)
        term2 = 2.7 * 10**-16 * np.exp(-h / 1500)
        term3 = A * np.exp(-h / 100)
        return term1 + term2 + term3

    def integral_function(A):
        integral_value, _ = quad(lambda z: Cn2(z, A), 0, delta_z)
        return integral_value

    def equation_to_solve(A):
        r0_pw_calculated = (0.423 * k**2 * integral_function(A)) ** (-3 / 5)
        return r0_pw_calculated - r0_pw_given

    A_initial_guess = 1.7e-14
    return fsolve(equation_to_solve, A_initial_guess)[0]


def circ_orbit_geo(h_orbit, elev_angle_deg=90):
    """Range and velocity components of a circular orbit at a given elevation.

    Returns
    -------
    (range_m, v_tangential, v_radial)
    """
    G = 6.67430e-11
    M = 5.972e24
    r_earth = 6.378e6

    R = r_earth + h_orbit
    v_orbit = np.sqrt(G * M / R)

    gamma_horiz_rad = np.deg2rad(elev_angle_deg)

    # Line-of-sight distance from the law of cosines.
    a = 1
    b = -2 * r_earth * np.cos(gamma_horiz_rad + np.pi / 2)
    c = r_earth**2 - R**2
    L = max(np.roots([a, b, c]))

    gamma_earth_rad = np.arccos((L**2 - R**2 - r_earth**2) / (-2 * R * r_earth))

    range_m = np.sqrt(R**2 + r_earth**2 - 2 * R * r_earth * np.cos(gamma_earth_rad))

    psi = np.pi - abs(gamma_earth_rad) - (np.pi / 2 + gamma_horiz_rad)
    v_radial = v_orbit * np.cos(np.pi / 2 - psi)
    v_tangential = v_orbit * np.sin(np.pi / 2 - psi)

    return range_m, v_tangential, v_radial


def calc_r0_profile(ground_wind_speed, satellite_orbit_height, print_results=False, r0=None,
                    num_layers=20, lamda=1550e-9, A=1.7e-14, max_altitude=20e3, zenith_angle=0):
    """Slice the HV profile into layers and assign r0, height and wind to each.

    Parameters
    ----------
    ground_wind_speed : float
        Ground wind speed in the Bufton model (m/s).
    satellite_orbit_height : float
        Circular-orbit altitude (m); sets the slew rate.
    r0 : float, optional
        Total plane-wave r0 (m). When given, the HV ``A`` term is solved for.
    num_layers : int
        Number of equal path-weighted-turbulence slices.
    lamda : float
        Wavelength (m).
    max_altitude : float
        Top of the turbulent atmosphere (m).
    zenith_angle : float
        Zenith angle (deg) used for the slew rate.

    Returns
    -------
    (r0_array, h_arr, wind_profile, cn2)
        Per-layer r0 (m), layer heights (m), layer wind speeds (m/s), and the
        ``cn2(h)`` callable.
    """
    k = 2 * np.pi / lamda

    # Apparent slew rate omega_s = v_tangential / range (rad/s).
    ws = circ_orbit_geo(satellite_orbit_height, 90 - zenith_angle)[1] / circ_orbit_geo(
        satellite_orbit_height, 90 - zenith_angle
    )[0]

    # Bufton wind model plus the slew-induced apparent wind.
    def bufton_function(h):
        return ws * h + ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)

    # The slew term is geometric, so it does not enter the HV wind parameter.
    def bufton_function_Cn2(h):
        return (ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)) ** 2

    rms_wind = np.sqrt(1 / (15 * 1e3) * integrate.quad(bufton_function_Cn2, 5e3, max_altitude)[0])

    if r0 is not None:
        A = solve_for_A(k, rms_wind, max_altitude, r0)

    def cn2(h):
        result = (.00594 * (rms_wind / 27) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000)
                  + 2.7e-16 * np.exp(-h / 1500) + A * np.exp(-h / 100))
        if np.iscomplexobj(result):
            return np.real(result)
        return result

    # Each layer carries 1/num_layers of int Cn^2(h) h^(5/6) dh.
    threshold = (1 / num_layers) * integrate.quad(lambda h: cn2(h) * (h - 0) ** (5 / 6), 0, max_altitude)[0]
    if print_results:
        print(f"Threshold: {threshold}")

    H = result = 0
    h_arr = [H]
    while result <= max_altitude:
        h0 = result
        initial_guess = h0

        def equation_to_solve(H):
            return integrate.quad(lambda h: cn2(h) * (h) ** (5 / 6), h0, H)[0] - threshold

        result = optimize.fsolve(equation_to_solve, initial_guess)
        h_arr.append(result)

    def r0_calc(h1, h2):
        return (.423 * k ** 2 * integrate.quad(cn2, h1, h2)[0]) ** (-3 / 5)

    r0_array = []
    for i in range(len(h_arr) - 1):
        r0_array.append(r0_calc(h_arr[i], h_arr[i + 1]))
    h_arr = h_arr[1:]

    wind_profile = list(map(bufton_function, h_arr))

    if print_results:
        print(f"r0_array: {r0_array}")
        print(f"Wind profile: {wind_profile}")
        print(f"Number of layers: {len(h_arr)}")

    h_arr = [item.item() if isinstance(item, np.ndarray) else item for item in h_arr]
    wind_profile = [item.item() if isinstance(item, np.ndarray) else item for item in wind_profile]
    return r0_array, h_arr, wind_profile, cn2


def calculate_number_of_extrusions(pixel_scale, v_total, simulation_tick_time_sec):
    """Rows extruded per tick for a layer moving at ``v_total`` (m/s)."""
    delta_x = v_total * simulation_tick_time_sec
    return delta_x / pixel_scale
