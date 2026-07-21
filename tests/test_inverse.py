"""Tests for the inverse beamforming subpackage (``zea.inverse``)."""

import numpy as np
import pytest
from keras import ops

from zea import Parameters
from zea.beamform.delays import compute_t0_delays_planewave
from zea.inverse import (
    DASOperator,
    ScattererSimulator,
    cgls,
    invert_direct,
    invert_scatterers,
    linear_adjoint,
    seed_scatterers,
)

N_EL = 8  # number of transducer elements
N_TX = 3  # number of plane-wave transmits
N_AX = 128  # number of axial samples
SOUND_SPEED = 1540.0  # m/s
SAMPLING_FREQ = 20e6  # Hz
CENTER_FREQ = 5e6  # Hz
WAVEFORM_SAMPLING_FREQ = 250e6  # Hz


def _make_parameters(grid_size_x=24, grid_size_z=32):
    """Small plane-wave scan with a windowed-sine two-way waveform."""
    xs = np.linspace(-4e-3, 4e-3, N_EL)
    probe_geometry = np.stack([xs, np.zeros(N_EL), np.zeros(N_EL)], axis=-1).astype(np.float32)
    polar_angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)

    n_pulse = int(2 / CENTER_FREQ * WAVEFORM_SAMPLING_FREQ)
    t = np.arange(n_pulse) / WAVEFORM_SAMPLING_FREQ
    pulse = (np.sin(2 * np.pi * CENTER_FREQ * t) * np.hanning(n_pulse)).astype(np.float32)

    return Parameters(
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=np.ones((N_TX, N_EL), dtype=np.float32),
        initial_times=np.zeros(N_TX, dtype=np.float32),
        sampling_frequency=SAMPLING_FREQ,
        center_frequency=CENTER_FREQ,
        sound_speed=SOUND_SPEED,
        n_ax=N_AX,
        n_el=N_EL,
        n_tx=N_TX,
        n_ch=1,
        n_frames=1,
        focus_distances=np.full(N_TX, np.inf, dtype=np.float32),
        polar_angles=polar_angles,
        azimuth_angles=np.zeros(N_TX, dtype=np.float32),
        transmit_origins=np.zeros((N_TX, 3), dtype=np.float32),
        waveforms_two_way=np.tile(pulse, (N_TX, 1)),
        element_width=float(xs[1] - xs[0]) * 0.9,
        # n_ax = 128 samples at 20 MHz records echoes from depths up to ~4.5 mm.
        xlims=(-3e-3, 3e-3),
        zlims=(1e-3, 4.5e-3),
        grid_size_x=grid_size_x,
        grid_size_z=grid_size_z,
        f_number=0.8,
    )


def _make_targets(seed=0, n_scatterers=8):
    """Random point targets inside the imaging region."""
    rng = np.random.default_rng(seed)
    positions = np.stack(
        [
            rng.uniform(-2.5e-3, 2.5e-3, n_scatterers),
            np.zeros(n_scatterers),
            rng.uniform(1.5e-3, 4e-3, n_scatterers),
        ],
        axis=1,
    ).astype(np.float32)
    magnitudes = rng.uniform(0.5, 1.5, n_scatterers).astype(np.float32)
    return positions, magnitudes


def _correlation(a, b):
    """Absolute normalized correlation between two arrays."""
    a = np.asarray(ops.convert_to_numpy(a)).ravel()
    b = np.asarray(ops.convert_to_numpy(b)).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


@pytest.fixture(scope="module")
def parameters():
    return _make_parameters()


@pytest.fixture(scope="module")
def operator(parameters):
    return DASOperator(parameters)


@pytest.fixture(scope="module")
def simulator(parameters):
    return ScattererSimulator(parameters, chunk_size=64)


@pytest.fixture(scope="module")
def measurement(operator, simulator):
    """Ground-truth channel data and its beamformed image."""
    positions, magnitudes = _make_targets()
    channel_data = simulator(magnitudes, positions=positions)
    image = operator.forward(channel_data)
    return channel_data, image


def test_linear_adjoint_matches_matrix_transpose():
    """linear_adjoint reproduces the transpose of an explicit matrix."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((6, 10)).astype(np.float32)
    rmatvec = linear_adjoint(lambda x: ops.matmul(matrix, x), ops.zeros(10))
    y = rng.standard_normal(6).astype(np.float32)
    np.testing.assert_allclose(ops.convert_to_numpy(rmatvec(y)), matrix.T @ y, atol=1e-5)


def test_cgls_matches_lstsq():
    """CGLS run to convergence matches the least-squares solution."""
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((20, 8)).astype(np.float32)
    b = rng.standard_normal(20).astype(np.float32)
    x = cgls(
        lambda v: ops.matmul(matrix, v),
        lambda v: ops.matmul(matrix.T, v),
        b,
        ops.zeros(8),
        n_iter=30,
    )
    expected = np.linalg.lstsq(matrix, b, rcond=None)[0]
    np.testing.assert_allclose(ops.convert_to_numpy(x), expected, atol=1e-3)


def test_cgls_minimum_norm_solution():
    """On an underdetermined system, CGLS from zero returns the pseudo-inverse solution."""
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((5, 12)).astype(np.float32)
    b = rng.standard_normal(5).astype(np.float32)
    x = cgls(
        lambda v: ops.matmul(matrix, v),
        lambda v: ops.matmul(matrix.T, v),
        b,
        ops.zeros(12),
        n_iter=30,
    )
    np.testing.assert_allclose(ops.convert_to_numpy(x), np.linalg.pinv(matrix) @ b, atol=1e-3)


def test_das_operator_adjoint_identity(operator):
    """The DAS operator satisfies the adjoint identity <Ax, y> == <x, A^T y>."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(operator.input_shape).astype(np.float32)
    y = rng.standard_normal(int(operator.n_pix)).astype(np.float32)
    lhs = float(ops.convert_to_numpy(ops.sum(operator.forward(x) * y)))
    rhs = float(ops.convert_to_numpy(ops.sum(ops.convert_to_tensor(x) * operator.adjoint(y))))
    assert abs(lhs - rhs) / abs(lhs) < 1e-3


def test_das_operator_is_linear(operator, measurement):
    """forward(a x1 + b x2) == a forward(x1) + b forward(x2)."""
    channel_data, _ = measurement
    rng = np.random.default_rng(4)
    other = rng.standard_normal(operator.input_shape).astype(np.float32)
    combined = operator.forward(2.0 * channel_data + 3.0 * ops.convert_to_tensor(other))
    separate = 2.0 * operator.forward(channel_data) + 3.0 * operator.forward(other)
    np.testing.assert_allclose(
        ops.convert_to_numpy(combined), ops.convert_to_numpy(separate), rtol=1e-4, atol=1e-4
    )


def test_das_operator_to_grid(operator, measurement):
    """to_grid reshapes the flat image to (grid_size_z, grid_size_x)."""
    _, image = measurement
    reshaped = operator.to_grid(image)
    assert tuple(reshaped.shape) == operator.parameters.grid.shape[:-1]


def test_simulator_echo_arrival_time(parameters, simulator):
    """A single scatterer produces its echo peak at the expected sample.

    The expected arrival follows the simulator's delay model: first-arrival
    transmit time (minimum over firing elements) plus the receive travel time,
    with the envelope peaking ``t_peak`` later.
    """
    from scipy.signal import hilbert

    position = np.array([[0.0, 0.0, 3e-3]], dtype=np.float32)
    channel_data = simulator(np.ones(1, dtype=np.float32), positions=position)
    channel_data = ops.convert_to_numpy(channel_data)

    tx = 1  # zero-angle plane wave
    element = int(np.argmin(np.abs(parameters.probe_geometry[:, 0])))
    element_distances = np.linalg.norm(parameters.probe_geometry - position[0], axis=1)
    transmit_time = np.min(parameters.t0_delays[tx] + element_distances / SOUND_SPEED)
    receive_time = element_distances[element] / SOUND_SPEED
    t_peak = float(np.asarray(parameters.t_peak)[tx])
    expected_sample = (transmit_time + receive_time + t_peak) * SAMPLING_FREQ

    envelope = np.abs(hilbert(channel_data[tx, :, element]))
    peak_sample = np.argmax(envelope)
    assert abs(peak_sample - expected_sample) <= 2


def test_simulator_geometry_reuse(simulator):
    """Precomputed geometry gives the same result as passing positions."""
    positions, magnitudes = _make_targets(seed=5, n_scatterers=4)
    direct = simulator(magnitudes, positions=positions)
    geometry = simulator.geometry(positions)
    reused = simulator(magnitudes, geometry=geometry)
    np.testing.assert_allclose(
        ops.convert_to_numpy(direct), ops.convert_to_numpy(reused), rtol=1e-6, atol=1e-6
    )


def test_simulator_beamformer_consistency():
    """Beamforming a simulated single-scatterer echo peaks at the scatterer.

    The simulator excludes the waveform peak offset from its travel times, so
    the beamformer (which adds ``t_peak``) must sample each echo at its peak:
    the image maximum should land on the scatterer position.
    """
    parameters = _make_parameters(grid_size_x=48, grid_size_z=64)
    operator = DASOperator(parameters)
    simulator = ScattererSimulator(parameters, chunk_size=64)

    position = np.array([[1e-3, 0.0, 3e-3]], dtype=np.float32)
    channel_data = simulator(np.ones(1, dtype=np.float32), positions=position)
    image = np.abs(ops.convert_to_numpy(operator.to_grid(operator.forward(channel_data))))

    grid = parameters.grid
    peak_z, peak_x = np.unravel_index(np.argmax(image), image.shape)
    distance = np.linalg.norm(grid[peak_z, peak_x] - position[0])
    wavelength = SOUND_SPEED / CENTER_FREQ
    assert distance < wavelength


def test_seed_scatterers_properties(operator, measurement):
    """Seeded scatterers are reproducible, in bounds, and follow brightness."""
    _, image = measurement
    image_grid = ops.convert_to_numpy(operator.to_grid(image))
    grid = operator.parameters.grid

    positions = seed_scatterers(image_grid, grid, 300, seed=0)
    assert positions.shape == (300, 3)
    assert positions.dtype == np.float32

    # Reproducible with the same seed.
    np.testing.assert_array_equal(positions, seed_scatterers(image_grid, grid, 300, seed=0))

    # Inside the field of view (up to half a pixel of jitter).
    dx = grid[0, 1, 0] - grid[0, 0, 0]
    dz = grid[1, 0, 2] - grid[0, 0, 2]
    assert positions[:, 0].min() >= grid[..., 0].min() - dx
    assert positions[:, 0].max() <= grid[..., 0].max() + dx
    assert positions[:, 2].min() >= grid[..., 2].min() - dz
    assert positions[:, 2].max() <= grid[..., 2].max() + dz

    # With no uniform floor, seeds concentrate in the bright half of the image.
    envelope = np.abs(image_grid)
    threshold = np.median(envelope)
    bright = envelope >= threshold
    seeded = seed_scatterers(image_grid, grid, 300, uniform_frac=0.0, envelope=False, seed=1)
    z_index = np.clip(np.searchsorted(grid[:, 0, 2], seeded[:, 2]), 0, grid.shape[0] - 1)
    x_index = np.clip(np.searchsorted(grid[0, :, 0], seeded[:, 0]), 0, grid.shape[1] - 1)
    fraction_bright = bright[z_index, x_index].mean()
    assert fraction_bright > 0.8


def test_seed_scatterers_shape_mismatch_raises(operator):
    """A shape mismatch between image and grid raises a ValueError."""
    grid = operator.parameters.grid
    with pytest.raises(ValueError, match="does not match"):
        seed_scatterers(np.zeros((4, 4)), grid, 10)


def test_simulator_rejects_invalid_chunk_size(parameters):
    """A non-positive chunk size raises a ValueError instead of dividing by zero."""
    with pytest.raises(ValueError, match="chunk_size"):
        ScattererSimulator(parameters, chunk_size=0)


def test_simulator_rejects_empty_magnitudes(simulator):
    """Simulating zero scatterers raises a ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        simulator(np.zeros(0, dtype=np.float32), positions=np.zeros((0, 3), dtype=np.float32))


def test_invert_scatterers_rejects_custom_grid(parameters, measurement):
    """Custom-flatgrid operators are rejected: seeding uses `parameters.grid`."""
    _, image = measurement
    flatgrid = np.asarray(parameters.grid, dtype=np.float32).reshape(-1, 3)[::2]
    operator = DASOperator(parameters, flatgrid=flatgrid)
    with pytest.raises(ValueError, match="custom `flatgrid`"):
        invert_scatterers(operator, image[: len(flatgrid)], n_scatterers=10, n_iter=1)


def test_invert_direct_fits_image(operator, measurement):
    """The pseudo-inverse reproduces the measured image almost exactly."""
    _, image = measurement
    result = invert_direct(operator, image, n_iter=25)
    assert _correlation(result.image, image) > 0.99
    assert tuple(result.channel_data.shape) == operator.input_shape


def test_invert_direct_twice_on_fresh_operator(parameters, measurement):
    """Repeated jitted inversions on a fresh operator must not leak tracers.

    Regression test: the adjoint used to be built lazily on first use, so a
    first ``invert_direct`` call (jitted) cached a closure over traced values
    and a second call raised ``UnexpectedTracerError`` on the jax backend.
    """
    _, image = measurement
    operator = DASOperator(parameters)
    first = invert_direct(operator, image, n_iter=2)
    second = invert_direct(operator, image, n_iter=2)
    np.testing.assert_allclose(
        ops.convert_to_numpy(first.channel_data),
        ops.convert_to_numpy(second.channel_data),
        rtol=1e-5,
        atol=1e-6,
    )


def test_invert_scatterers_recovers_channel_data(operator, measurement):
    """The scatterer prior recovers the underlying channel data.

    This is the core property of the scatterer-prior inversion: the direct
    pseudo-inverse fits the image but not the channel data (the minimum-norm
    solution lives in the nullspace-orthogonal complement), whereas the
    scatterer parameterization regularizes the inversion towards physically
    consistent echoes.
    """
    channel_data, image = measurement
    result = invert_scatterers(operator, image, n_scatterers=400, n_iter=25, seed=0)

    assert result.positions.shape == (400, 3)
    assert tuple(result.channel_data.shape) == operator.input_shape
    assert _correlation(result.image, image) > 0.9
    assert _correlation(result.channel_data, channel_data) > 0.75

    direct = invert_direct(operator, image, n_iter=25)
    assert _correlation(result.channel_data, channel_data) > _correlation(
        direct.channel_data, channel_data
    )


def test_invert_scatterers_refine(operator, measurement):
    """Joint Adam refinement runs and does not degrade the data fit."""
    _, image = measurement
    result = invert_scatterers(operator, image, n_scatterers=100, n_iter=10, refine_iters=5, seed=0)
    assert result.positions.shape == (100, 3)
    assert _correlation(result.image, image) > 0.5
