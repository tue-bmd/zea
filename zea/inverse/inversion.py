"""High-level drivers for inverting the DAS beamformer.

Recovers pre-beamformed channel data from a post-beamformed (DAS) image. Two
inversions are provided:

* :func:`invert_direct` — solve for the full channel-data cube directly with
  CGLS. The DAS operator is massively underdetermined, so this yields the
  minimum-norm least-squares (pseudo-inverse) solution: it reproduces the
  image almost exactly but recovers the physical channel data poorly.
* :func:`invert_scatterers` — fit point-scatterer magnitudes (positions
  seeded from the image, shared across transmits) whose simulated channel
  data reproduces the image. The scatterer parameterization regularizes the
  nullspace of the DAS operator and recovers channel data far better on
  point-target scans. Optionally refines positions and magnitudes jointly
  with Adam.
"""

from dataclasses import dataclass

import keras
import numpy as np
from keras import ops

from zea import log
from zea.backend import jit as backend_jit
from zea.backend.autograd import AutoGrad
from zea.backend.optimizer import adam
from zea.inverse.operators import ScattererSimulator
from zea.inverse.seeding import seed_scatterers
from zea.inverse.solvers import cgls, linear_adjoint


def _jit_through_scan_grad(fn):
    """JIT-compile a function that differentiates through ``ops.scan``.

    On the tensorflow backend XLA cannot size the gradient accumulators of the
    scan's while loop, so compile without XLA there.
    """
    if keras.backend.backend() == "tensorflow":
        return backend_jit(fn, jit_compile=False)
    return backend_jit(fn)


@dataclass
class InversionResult:
    """Result of a DAS inversion.

    Args:
        channel_data (Tensor): Recovered pre-beamformed channel data of shape
            ``(n_tx, n_ax, n_el)``.
        image (Tensor): Re-beamformed image of the recovered channel data,
            flattened to shape ``(n_pix,)``. Compare against the measured
            image to assess the data fit.
        positions (ndarray | None): Scatterer positions ``(n_scat, 3)``.
            Only set by :func:`invert_scatterers`.
        magnitudes (Tensor | None): Scatterer magnitudes ``(n_scat,)``.
            Only set by :func:`invert_scatterers`.
    """

    channel_data: object
    image: object
    positions: object = None
    magnitudes: object = None


def invert_direct(operator, image, n_iter=50, jit=True, verbose=False):
    """Recover channel data from a beamformed image by pseudo-inversion.

    Solves ``min ||operator(channel_data) - image||^2`` over the full channel
    data cube with CGLS starting from zero, which converges to the
    minimum-norm (Moore-Penrose) solution. This fits the image essentially
    perfectly but, because the DAS operator has a large nullspace, the
    minimum-norm solution is generally *not* the physical channel data — see
    :func:`invert_scatterers` for a regularized alternative.

    Args:
        operator (DASOperator): The beamforming operator to invert.
        image (Tensor): Measured beamformed image, flattened ``(n_pix,)`` or
            shaped ``(grid_size_z, grid_size_x)``.
        n_iter (int, optional): CGLS iterations. Defaults to ``50``.
        jit (bool, optional): JIT-compile the operator applications (JAX and
            TensorFlow backends). Defaults to ``True``.
        verbose (bool, optional): Log CGLS progress. Defaults to ``False``.

    Returns:
        InversionResult: Recovered channel data and its re-beamformed image.
    """
    image = ops.reshape(ops.cast(ops.convert_to_tensor(image), "float32"), (-1,))
    matvec = operator.forward
    rmatvec = operator.adjoint
    if jit:
        matvec = backend_jit(matvec)
        rmatvec = _jit_through_scan_grad(rmatvec)
    channel_data = cgls(
        matvec,
        rmatvec,
        image,
        ops.zeros(operator.input_shape, dtype="float32"),
        n_iter=n_iter,
        verbose=verbose,
    )
    return InversionResult(channel_data=channel_data, image=matvec(channel_data))


def invert_scatterers(
    operator,
    image,
    n_scatterers=5000,
    n_iter=50,
    prob_exponent=2.5,
    uniform_frac=0.3,
    refine_iters=0,
    refine_step_size=0.05,
    simulator=None,
    seed=None,
    jit=True,
    verbose=False,
):
    """Recover channel data from a beamformed image with a scatterer prior.

    Seeds point scatterers from the image envelope, then solves the convex
    subproblem for their magnitudes with CGLS (positions fixed):
    ``min ||operator(simulate(positions, magnitudes)) - image||^2``.
    Optionally refines positions and magnitudes jointly with Adam afterwards
    (``refine_iters > 0``); positions are optimized in units of wavelength so
    a single step size applies to both variables.

    The scatterer parameterization regularizes the nullspace of the DAS
    operator: unlike :func:`invert_direct`, the recovered channel data is
    constrained to physically consistent point-scatterer echoes.

    Args:
        operator (DASOperator): The beamforming operator to invert.
        image (Tensor): Measured beamformed image, flattened ``(n_pix,)`` or
            shaped ``(grid_size_z, grid_size_x)``.
        n_scatterers (int, optional): Number of scatterers. Defaults to
            ``5000``.
        n_iter (int, optional): CGLS iterations for the magnitudes. Defaults
            to ``50``.
        prob_exponent (float, optional): Seeding sharpness, see
            :func:`zea.inverse.seed_scatterers`. Defaults to ``2.5``.
        uniform_frac (float, optional): Fraction of uniformly seeded
            scatterers, see :func:`zea.inverse.seed_scatterers`. Defaults to
            ``0.3``.
        refine_iters (int, optional): Adam iterations jointly refining
            positions and magnitudes. Defaults to ``0`` (disabled).
        refine_step_size (float, optional): Adam step size (wavelengths for
            positions). Defaults to ``0.05``.
        simulator (ScattererSimulator, optional): Custom simulator. Defaults
            to ``ScattererSimulator(operator.parameters)``.
        seed (int, optional): Seed for reproducible scatterer placement.
        jit (bool, optional): JIT-compile the operator applications (JAX and
            TensorFlow backends). Defaults to ``True``.
        verbose (bool, optional): Log progress. Defaults to ``False``.

    Returns:
        InversionResult: Recovered channel data, its re-beamformed image, and
        the scatterer positions and magnitudes.
    """
    parameters = operator.parameters
    if simulator is None:
        simulator = ScattererSimulator(parameters)

    # Seeding samples positions from `parameters.grid`, so an operator built
    # on a custom flatgrid would be seeded at unrelated coordinates. For
    # custom grids, seed positions manually and use ScattererSimulator + cgls.
    parameters_flatgrid = np.asarray(ops.convert_to_numpy(parameters.flatgrid), dtype=np.float32)
    operator_flatgrid = np.asarray(ops.convert_to_numpy(operator.flatgrid), dtype=np.float32)
    if operator_flatgrid.shape != parameters_flatgrid.shape or not np.allclose(
        operator_flatgrid, parameters_flatgrid
    ):
        raise ValueError(
            "`invert_scatterers` seeds scatterers from `operator.parameters.grid`, "
            "which does not match the operator's custom `flatgrid`. Build the "
            "operator on `parameters.flatgrid`, or seed positions manually and "
            "solve with `ScattererSimulator` and `cgls` directly."
        )

    image = ops.reshape(ops.cast(ops.convert_to_tensor(image), "float32"), (-1,))
    positions = seed_scatterers(
        ops.convert_to_numpy(operator.to_grid(image)),
        parameters.grid,
        n_scatterers,
        prob_exponent=prob_exponent,
        uniform_frac=uniform_frac,
        seed=seed,
    )

    geometry = simulator.geometry(positions)

    def matvec(magnitudes):
        return operator.forward(simulator(magnitudes, geometry=geometry))

    rmatvec = linear_adjoint(matvec, ops.zeros((n_scatterers,), dtype="float32"))
    if jit:
        matvec = backend_jit(matvec)
        rmatvec = _jit_through_scan_grad(rmatvec)

    magnitudes = cgls(
        matvec,
        rmatvec,
        image,
        ops.zeros((n_scatterers,), dtype="float32"),
        n_iter=n_iter,
        verbose=verbose,
    )

    if refine_iters > 0:
        positions, magnitudes = _refine_scatterers(
            operator,
            simulator,
            image,
            positions,
            magnitudes,
            n_iter=refine_iters,
            step_size=refine_step_size,
            jit=jit,
            verbose=verbose,
        )
        geometry = simulator.geometry(positions)

    channel_data = simulator(magnitudes, geometry=geometry)
    return InversionResult(
        channel_data=channel_data,
        image=operator.forward(channel_data),
        positions=positions,
        magnitudes=magnitudes,
    )


def _refine_scatterers(
    operator,
    simulator,
    image,
    positions,
    magnitudes,
    n_iter,
    step_size,
    jit=True,
    verbose=False,
):
    """Jointly refine scatterer positions and magnitudes with Adam.

    Positions are optimized in units of wavelength so that a single step size
    is meaningful for both positions and magnitudes. Returns the refined
    ``(positions, magnitudes)``.
    """
    parameters = operator.parameters
    wavelength = parameters.sound_speed / np.mean(ops.convert_to_numpy(parameters.center_frequency))

    def loss(variable):
        positions = variable[:, :3] * wavelength
        magnitudes = variable[:, 3]
        residual = operator.forward(simulator(magnitudes, positions=positions)) - image
        return ops.sum(residual**2)

    autograd = AutoGrad()
    autograd.set_function(loss)
    gradient_fn = autograd.gradient
    if jit:
        gradient_fn = _jit_through_scan_grad(gradient_fn)

    init, update, get_params = adam(step_size)
    variable = ops.concatenate(
        [ops.convert_to_tensor(positions) / wavelength, magnitudes[:, None]], axis=1
    )
    state = init(variable)
    log_every = max(1, n_iter // 8)
    for iteration in range(1, n_iter + 1):
        gradient = gradient_fn(get_params(state))
        state = update(gradient, state)
        if verbose and (iteration % log_every == 0 or iteration == n_iter):
            loss_value = float(ops.convert_to_numpy(loss(get_params(state))))
            log.info(f"refine iteration {iteration:3d} | loss {loss_value:.4e}")

    variable = get_params(state)
    return ops.convert_to_numpy(variable[:, :3] * wavelength), variable[:, 3]
