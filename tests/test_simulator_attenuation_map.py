"""Attenuation maps in :func:`zea.simulator.simulate_rf`.

A uniform map is the homogeneous medium, and a layered map attenuates each echo by the line
integral of the coefficient along its straight rays, linearly in frequency.
"""

import keras
import numpy as np
import pytest

import zea
from zea.ops import Pipeline, Simulate
from zea.simulator import pressure_field, simulate_rf

from . import simulator_helpers
from .simulator_helpers import (
    CENTER_FREQUENCY,
    N_AX,
    SAMPLING_FREQUENCY,
    SOUND_SPEED,
    assert_close,
    hann_waveform,
    linear_probe,
    matrix_probe,
    rel_err,
    tensors,
    to_np,
)

# The attenuation of the shared scan, dB/cm/MHz.
COEF = 0.5


def _grid(x=(-0.02, 0.02), z=(-0.005, 0.04), nx=41, nz=46, y=None, ny=None):
    """The grid arguments and the map shape; 3D when a ``y`` range is given."""
    grids = {
        "map_grid_x": np.linspace(*x, nx).astype(np.float32),
        "map_grid_z": np.linspace(*z, nz).astype(np.float32),
    }
    shape = (nz, nx)
    if y is not None:
        grids["map_grid_y"] = np.linspace(*y, ny).astype(np.float32)
        shape = (nz, nx, ny)
    return grids, shape


def _map(coef, **kwargs):
    """A uniform attenuation map with its grids."""
    grids, shape = _grid(**kwargs)
    return {"attenuation_map": np.full(shape, coef, np.float32), **grids}


def _layered_map(coef_near, coef_far, interface, axis="z", **kwargs):
    """Two layers along ``axis``, ``coef_far`` from the map row at ``interface`` on. Bilinear
    interpolation ramps over the row before it, so the effective interface is half a row earlier."""
    duo = _map(coef_near, **kwargs)
    grid = duo[f"map_grid_{axis}"]
    assert interface in grid
    index = (slice(None),) * {"z": 0, "x": 1, "y": 2}[axis] + (grid >= interface,)
    duo["attenuation_map"][index] = coef_far
    return duo


HANN_WAVEFORM = hann_waveform()


def case(geometry, n_tx=2, **overrides):
    """The shared phantom in the shared scan (attenuation ``COEF``), two transmits."""
    return simulator_helpers.case(geometry, n_tx, attenuation_coef=COEF, **overrides)


def _point_echo(position, maps, geometry, transmit, coef, **overrides):
    """Echo of one unit scatterer for one transmitting element, as a clean tone per element:
    no directivity, no scattering gain, attenuated by ``coef`` where ``maps`` do not apply."""
    apod = np.zeros((1, len(geometry)), np.float32)
    apod[0, transmit] = 1.0
    kwargs = case(geometry, n_tx=1)
    kwargs.update(
        scatterer_positions=np.asarray([position], np.float32),
        scatterer_magnitudes=np.ones(1, np.float32),
        t0_delays=np.zeros((1, len(geometry)), np.float32),
        tx_apodizations=apod,
        element_width=1e-6,
        attenuation_coef=coef,
        scatter_exponent=0.0,
        n_ax=N_AX,
        waveforms_two_way=HANN_WAVEFORM,
        n_sos_ray_samples=1024,
        **maps,
        **overrides,
    )
    return to_np(simulate_rf(**tensors(kwargs)))[0, :, :, 0]


def _spectral_ratio_db(rf, reference):
    """Ratio of the spectra of two records [dB] over the bins of the band that carry the
    tone, as (frequencies, ratio) per element."""
    f = np.fft.rfftfreq(N_AX, 1 / SAMPLING_FREQUENCY)
    spectrum, ref = np.fft.rfft(rf, axis=0), np.fft.rfft(reference, axis=0)
    out = []
    for e in range(rf.shape[1]):
        keep = (np.abs(f - CENTER_FREQUENCY) < 1e6) & (
            np.abs(ref[:, e]) > 0.1 * np.abs(ref[:, e]).max()
        )
        out.append((f[keep], 20 * np.log10(np.abs(spectrum[keep, e]) / np.abs(ref[keep, e]))))
    return out


UNIFORM_CASES = {
    "linear": case(linear_probe()),
    "matrix": case(matrix_probe()),
    "two_dimensional": case(linear_probe(), two_dimensional=True),
    "sub_elements": case(linear_probe(), n_sub_elements=(3, 2)),
    "lens": case(linear_probe(), apply_lens_correction=True, lens_attenuation_coef=1.0),
    "matrix_3d_map": case(matrix_probe(), **_map(COEF, y=(-0.003, 0.003), ny=7)),
}


def test_straight_ray_sampler_rejects_no_samples():
    from zea.func.ultrasound import straight_ray_mean

    axis = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    with pytest.raises(ValueError, match="n_samples"):
        straight_ray_mean(
            np.zeros((2, 3), np.float32),
            np.zeros((1, 3), np.float32),
            np.ones((4, 4), np.float32),
            axis,
            axis,
            0.0,
            n_samples=0,
        )


@pytest.mark.parametrize("name", list(UNIFORM_CASES))
def test_uniform_map_is_the_homogeneous_medium(name):
    kwargs = dict(UNIFORM_CASES[name])
    if "attenuation_map" not in kwargs:
        kwargs.update(_map(COEF))
    homogeneous = {
        k: v for k, v in kwargs.items() if k != "attenuation_map" and not k.startswith("map_grid_")
    }
    reference = simulate_rf(**tensors(homogeneous))
    assert to_np(reference).any()
    assert_close(reference, simulate_rf(**tensors(kwargs)))


def test_map_at_another_coefficient_is_the_medium_at_that_coefficient():
    """A map covering every path at ``coef2`` is the medium at ``coef2``; a map that covers no
    path is the medium at ``attenuation_coef``."""
    coef2 = 1.5
    kwargs = case(matrix_probe())
    reference = simulate_rf(**tensors({**kwargs, "attenuation_coef": coef2}))
    mapped = simulate_rf(**tensors({**kwargs, **_map(coef2)}))
    assert_close(reference, mapped)
    assert rel_err(simulate_rf(**tensors(kwargs)), mapped) > 0.1

    aside = _map(coef2, x=(0.1, 0.2))
    assert_close(simulate_rf(**tensors(kwargs)), simulate_rf(**tensors({**kwargs, **aside})))


@pytest.mark.parametrize(
    "axis, position, transmit, grid",
    [
        ("z", [0.002, 0.0, 0.025], 3, dict(z=(-0.004, 0.04), nz=45)),
        ("y", [0.001, 0.01, 0.02], 5, dict(y=(-0.002, 0.012), ny=15)),
    ],
)
def test_layered_map_attenuates_each_echo_by_its_line_integral(axis, position, transmit, grid):
    """Behind a layer of higher attenuation every echo loses, in dB, the extra coefficient
    times the ray length in the layer times the frequency, on transmit and on receive. A layer
    in y needs a 3D map."""
    coef_near, coef_far, interface = COEF, 1.5, 0.012 if axis == "z" else 0.004
    duo = _layered_map(coef_near, coef_far, interface, axis=axis, **grid)
    geometry = linear_probe()
    position = np.array(position)
    mapped = _point_echo(position, duo, geometry, transmit, coef_near)
    plain = _point_echo(position, {}, geometry, transmit, coef_near)

    edge = interface - 0.5e-3  # the map ramps over the 1 mm row before the interface
    column = {"z": 2, "y": 1}[axis]
    far = 1 - edge / position[column]  # fraction of every ray beyond the interface
    dist = np.linalg.norm(position - geometry, axis=1)
    per_cm_mhz = (coef_far - coef_near) * 100 * CENTER_FREQUENCY * 1e-6
    expected = per_cm_mhz * far * (dist[transmit] + dist)  # dB at fc, transmit plus receive
    assert expected.min() > 3
    for e, (f, ratio) in enumerate(_spectral_ratio_db(mapped, plain)):
        assert len(f) > 50
        assert np.abs(ratio + expected[e] * f / CENTER_FREQUENCY).max() < 1e-3


def test_both_maps_together_are_each_map_alone_when_the_other_is_uniform():
    kwargs = case(linear_probe())
    grids, shape = _grid()
    sos_map = np.full(shape, SOUND_SPEED, np.float32)
    sos_map[grids["map_grid_z"] >= 0.015] = 1450.0
    attenuation_map = np.full(shape, COEF, np.float32)
    attenuation_map[grids["map_grid_z"] >= 0.015] = 1.5

    def run(**maps):
        return simulate_rf(**tensors({**kwargs, **grids, **maps}))

    uniform_sos = np.full(shape, SOUND_SPEED, np.float32)
    uniform_att = np.full(shape, COEF, np.float32)
    assert_close(run(sos_map=sos_map), run(sos_map=sos_map, attenuation_map=uniform_att))
    assert_close(
        run(attenuation_map=attenuation_map),
        run(sos_map=uniform_sos, attenuation_map=attenuation_map),
    )
    both = run(sos_map=sos_map, attenuation_map=attenuation_map)
    assert rel_err(run(sos_map=sos_map), both) > 0.1
    assert rel_err(run(attenuation_map=attenuation_map), both) > 0.1


def test_map_and_grids_are_validated():
    kwargs = case(linear_probe())
    duo = _map(COEF)

    def run(**overrides):
        return simulate_rf(**tensors({**kwargs, **overrides}))

    with pytest.raises(ValueError, match="without sos_map or attenuation_map"):
        run(map_grid_x=duo["map_grid_x"], map_grid_z=duo["map_grid_z"])
    with pytest.raises(ValueError, match="attenuation_map needs the coordinates"):
        run(attenuation_map=duo["attenuation_map"])
    with pytest.raises(ValueError, match="attenuation_map of shape"):
        run(**{**duo, "attenuation_map": duo["attenuation_map"].T})
    with pytest.raises(ValueError, match="non-negative"):
        run(**{**duo, "attenuation_map": -duo["attenuation_map"]})
    # Zero is allowed: no attenuation inside the map.
    reference = run(**{**duo, "attenuation_map": np.zeros_like(duo["attenuation_map"])})
    assert_close(reference, run(attenuation_coef=0.0))


def test_pressure_field_follows_the_map():
    """A stronger map lowers the field, and more so deeper into it."""
    kwargs = case(linear_probe())
    names = (
        "probe_geometry",
        "sound_speed",
        "center_frequency",
        "sampling_frequency",
        "t0_delays",
        "initial_times",
        "element_width",
        "tx_apodizations",
        "t_peak",
        "attenuation_coef",
    )
    transmit = tensors({k: kwargs[k] for k in names})
    z = np.linspace(0.005, 0.03, 12)
    grid = np.stack([np.zeros_like(z), np.zeros_like(z), z], -1).astype(np.float32)
    reference = to_np(pressure_field(grid, **transmit, n_ax=N_AX))
    uniform = pressure_field(grid, **transmit, n_ax=N_AX, **tensors(_map(COEF)))
    assert_close(reference, uniform)
    stronger = to_np(pressure_field(grid, **transmit, n_ax=N_AX, **tensors(_map(2.0))))
    ratio = stronger / reference
    assert (ratio < 1).all() and (np.diff(ratio, axis=-1) < 0).all()


def test_simulate_op_and_parameters_take_the_map():
    kwargs = case(linear_probe())
    duo = _layered_map(COEF, 1.5, 0.015)
    reference = simulate_rf(**tensors({**kwargs, **duo}))
    assert rel_err(simulate_rf(**tensors(kwargs)), reference) > 0.1
    op = Simulate(jit_compile=True, with_batch_dim=False)
    assert_close(reference, op(**tensors({**kwargs, **duo}))[op.output_key])
    with pytest.raises(ValueError, match="attenuation_map is only supported"):
        Simulate(jit_compile=False, with_batch_dim=False)(
            **tensors({**kwargs, **duo}), method="time_domain"
        )

    parameters = zea.Parameters(
        n_tx=2,
        n_el=16,
        n_ax=N_AX,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
        probe_geometry=kwargs["probe_geometry"],
        t0_delays=kwargs["t0_delays"],
        initial_times=kwargs["initial_times"],
        t_peak=kwargs["t_peak"],
        tx_apodizations=kwargs["tx_apodizations"],
        sound_speed=SOUND_SPEED,
        selected_transmits="all",
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        element_width=0.27e-3,
        attenuation_coef=COEF,
        **duo,
    )
    pipeline = Pipeline([Simulate()], with_batch_dim=False, jit_options="pipeline")
    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(
        **inputs,
        scatterer_positions=kwargs["scatterer_positions"],
        scatterer_magnitudes=kwargs["scatterer_magnitudes"],
        scatter_exponent=kwargs["scatter_exponent"],
    )
    assert_close(reference, outputs["data"])


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax gradients")
def test_gradient_with_respect_to_the_map():
    """More attenuation anywhere can only lower the energy of the record, and the gradient
    matches finite differences on pixels the rays cross."""
    import jax
    import jax.numpy as jnp

    kwargs = case(linear_probe(n_el=8), n_fft=1024)
    duo = _layered_map(COEF, 1.5, 0.015, nx=21, nz=23)
    attenuation_map = jnp.asarray(duo.pop("attenuation_map"))
    kwargs.update(duo)

    def energy(m):
        return jnp.sum(simulate_rf(attenuation_map=m, **kwargs) ** 2)

    grad = jax.grad(energy)(attenuation_map)
    assert np.isfinite(to_np(grad)).all() and to_np(grad).any()
    assert (to_np(grad) <= 0).all()
    for i, j in ((10, 10), (15, 12)):
        bump = jnp.zeros_like(attenuation_map).at[i, j].set(0.05)
        fd = (energy(attenuation_map + bump) - energy(attenuation_map - bump)) / 0.1
        assert abs(grad[i, j] / fd - 1) < 1e-2, (i, j)
