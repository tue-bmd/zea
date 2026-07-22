"""Functional core for ultrasound computed tomography (USCT) reconstruction.

This module holds the pure, standalone functions behind
:class:`zea.ops.USCTReflectivityDAS`. They can be used directly (e.g. from a
notebook or a custom script) without constructing a :class:`~zea.Pipeline`.
The operation in :mod:`zea.ops.usct` is a thin wrapper around
:func:`usct_reflectivity_das`.

See :class:`zea.ops.USCTReflectivityDAS` for the physical model (round-trip
time-of-flight Delay-And-Sum reflectivity, transmission rejection, backscatter
apodization and optional straight-ray speed-of-sound correction).

All geometry is expressed in the **2-D imaging plane**: ``transmit_origins``,
``receive_positions`` and ``pixels`` are ``(..., 2)`` in-plane coordinates.
"""

from keras import ops

from zea.func.tensor import vmap

__all__ = [
    "straight_ray_times",
    "usct_reflectivity_das",
]


def distance_and_unit(src, pixels):
    """``src`` ``(M, D)``, ``pixels`` ``(P, D)`` -> ``(dist (M, P), unit (M, P, D))``
    where ``unit`` points from each pixel toward the source."""
    from zea.beamform.beamformer import compute_receive_distances

    diff = src[:, None, :] - pixels[None, :, :]
    dist = compute_receive_distances(src, pixels)
    unit = diff / (dist[..., None] + 1e-9)
    return dist, unit


def _pairwise_batched_direct(tx_pos, rx_batch):
    """Direct source->element distance for per-transmit apertures.

    ``tx_pos`` ``(c, 2)``, ``rx_batch`` ``(c, n_el, 2)`` -> ``(c, n_el)``.
    """
    diff = rx_batch - tx_pos[:, None, :]
    return ops.sqrt(ops.sum(ops.square(diff), axis=-1))


def _sample_grid(grid, x_axis, z_axis, xq, zq):
    """Bilinearly sample a ``(nz, nx)`` grid at world-frame query points.

    Thin coordinate conversion around :func:`keras.ops.image.map_coordinates`;
    ``fill_mode="nearest"`` clamps queries outside the footprint to the edge
    values (the caller masks those out separately).
    """
    xit = (xq - x_axis[0]) / (x_axis[1] - x_axis[0])
    zit = (zq - z_axis[0]) / (z_axis[1] - z_axis[0])
    return ops.image.map_coordinates(
        grid, ops.stack([zit, xit], axis=0), order=1, fill_mode="nearest"
    )


def straight_ray_times(positions, pixels, sos_map, x_axis, z_axis, background_c, n_samples=16):
    """Straight-ray travel times through a heterogeneous sound-speed map.

    For each source/element position the local slowness (``1 / c``) is integrated
    along the straight line to every pixel, sampling ``sos_map`` where the ray is
    inside the map footprint and falling back to ``background_c`` outside it. The
    loop over ``positions`` keeps peak memory at ``O(n_samples * n_pixels)``.

    Args:
        positions: ``(M, 2)`` source or element positions, in-plane.
        pixels: ``(P, 2)`` reconstruction grid points, in-plane.
        sos_map: ``(nz, nx)`` sound-speed values.
        x_axis: ``(nx,)`` horizontal coordinates of ``sos_map`` (world frame).
        z_axis: ``(nz,)`` vertical coordinates of ``sos_map`` (world frame).
        background_c: sound speed used outside the map footprint.
        n_samples: number of midpoint samples along each ray.

    Returns:
        ``(M, P)`` travel times [s]. The map's two axes are taken to be the two
        in-plane coordinate components (columns 0 and 1 of ``positions``/``pixels``).
    """
    t_mid = (ops.arange(n_samples, dtype="float32") + 0.5) / n_samples
    x_lo, x_hi = x_axis[0], x_axis[-1]
    z_lo, z_hi = z_axis[0], z_axis[-1]

    seg = pixels[None, :, :] - positions[:, None, :]  # (M, P, 2)
    dist = ops.sqrt(ops.sum(ops.square(seg), axis=-1))  # (M, P)
    pts = positions[:, None, None, :] + t_mid[None, :, None, None] * seg[:, None, :, :]  # (M,S,P,2)
    xq, zq = pts[..., 0], pts[..., 1]  # (M, S, P)
    c_samp = _sample_grid(sos_map, x_axis, z_axis, xq, zq)
    inside = ops.logical_and(
        ops.logical_and(xq >= x_lo, xq <= x_hi),
        ops.logical_and(zq >= z_lo, zq <= z_hi),
    )
    c_eff = ops.where(inside, c_samp, background_c)
    return dist * ops.mean(1.0 / c_eff, axis=1)  # (M, P)


def _gather_time(trace, sample_pos, n_ax, interpolation):
    """Sample ``trace`` ``(..., n_ax)`` at fractional ``sample_pos`` (same leading
    shape as the output) along the last axis.

    ``nearest`` rounds to the closest sample; ``linear`` does a two-tap lerp
    between the floor and ceil samples. Positions are clamped into range; the
    caller is responsible for masking out-of-range positions via ``valid``.
    """
    if interpolation == "nearest":
        idx = ops.cast(ops.clip(ops.round(sample_pos), 0, n_ax - 1), "int32")
        return ops.take_along_axis(trace, idx, axis=-1)
    if interpolation == "linear":
        i0 = ops.floor(sample_pos)
        frac = ops.cast(sample_pos - i0, "complex64")
        i0c = ops.cast(ops.clip(i0, 0, n_ax - 1), "int32")
        i1c = ops.cast(ops.clip(i0 + 1.0, 0, n_ax - 1), "int32")
        a0 = ops.take_along_axis(trace, i0c, axis=-1)
        a1 = ops.take_along_axis(trace, i1c, axis=-1)
        return a0 * (1.0 - frac) + a1 * frac
    raise ValueError(f"interpolation must be 'linear' or 'nearest', got {interpolation!r}.")


def usct_reflectivity_das(
    analytic,
    transmit_origins,
    receive_positions,
    pixels,
    sampling_frequency,
    initial_times,
    sound_speed,
    *,
    tx_chunk=4,
    reject_transmission=True,
    transmission_guard_s=2.5e-6,
    backscatter_apodization=True,
    interpolation="linear",
    compounding="coherent",
    sos_map=None,
    sos_grid_x=None,
    sos_grid_z=None,
    n_sos_ray_samples=16,
):
    """Round-trip TOF DAS reflectivity for a single Ultrasound Computed Tomography frame.

    .. seealso::

        See :class:`zea.ops.USCTReflectivityDAS` for the physical model and
        more detailed documentation.

    Args:
        analytic: complex tensor ``(n_tx, n_ax, n_el)`` — the analytic channel
            signal (Hilbert of RF, or I/Q recombined to complex).
        transmit_origins: ``(n_tx, 2)`` in-plane transmit point-source positions.
        receive_positions: ``(n_el, 2)`` shared receive-element positions, or
            ``(n_tx, n_el, 2)`` per-transmit positions (e.g. a sliding sub-aperture).
        pixels: ``(P, 2)`` in-plane reconstruction grid points, ``P = nz * nx``.
        sampling_frequency: sampling rate [Hz].
        initial_times: ``(n_tx,)`` per-transmit time-zero [s].
        sound_speed: background sound speed [m/s].
        tx_chunk: transmits processed per vectorized batch (memory/speed knob).
        reject_transmission: drop the direct through-transmission arrival.
        transmission_guard_s: guard interval [s] added to the direct-path time
            when ``reject_transmission`` is set.
        backscatter_apodization: weight pairs by the pixel-referred tx/rx cosine.
        interpolation: ``"linear"`` (two-tap fractional-sample lerp, default) or
            ``"nearest"`` (round to the closest sample). Linear preserves carrier
            phase far better when the acquisition is only marginally oversampled
            (e.g. ring probes at ~4 samples/wavelength).
        compounding: ``"coherent"`` (default) sums the analytic signal across
            every transmit/receive pair as one complex sum and takes the
            magnitude once at the end — the highest-resolution option when the
            delay model is accurate everywhere. ``"incoherent"`` instead takes
            the magnitude per transmit (coherent across the receive aperture
            only) and averages those magnitudes across transmits; it trades
            some resolution for robustness to phase decorrelation between
            transmits caused by sound-speed mismatch or calibration error,
            which grows with the size of the aperture spanned by a full ring.
        sos_map, sos_grid_x, sos_grid_z: optional SoS map and its in-plane axes,
            enabling straight-ray SoS-corrected delays.
        n_sos_ray_samples: ray samples for the SoS integral.

    Returns:
        ``(P,)`` float32 magnitude reflectivity (linear scale), one value per pixel
        of ``pixels``. Pixels are independent, so callers can reconstruct the grid
        in patches (see :class:`zea.ops.PatchedGrid`) and reshape afterwards (see
        :class:`zea.ops.ReshapeGrid`).
    """
    if compounding not in ("coherent", "incoherent"):
        raise ValueError(f"compounding must be 'coherent' or 'incoherent', got {compounding!r}.")
    from zea.beamform.beamformer import compute_receive_distances

    n_ax = int(analytic.shape[1])
    fs = sampling_frequency
    per_tx_rx = len(receive_positions.shape) == 3

    sos_args = (sos_map, sos_grid_x, sos_grid_z)
    if any(a is not None for a in sos_args) and not all(a is not None for a in sos_args):
        raise ValueError(
            "sos_map, sos_grid_x, and sos_grid_z must all be provided together, or all omitted."
        )
    use_sos = sos_map is not None
    px = ops.convert_to_tensor(pixels)

    # Precompute receive-leg geometry when the aperture is shared across transmits.
    # This is the single biggest beneficiary of `straight_ray_times` now using
    # `vmap` internally instead of a Python loop over `n_el` elements (see below).
    if not per_tx_rx:
        if use_sos:
            rx_time = straight_ray_times(
                receive_positions,
                px,
                sos_map,
                sos_grid_x,
                sos_grid_z,
                sound_speed,
                n_sos_ray_samples,
            )
            _, rx_unit = distance_and_unit(receive_positions, px)
        else:
            rx_dist, rx_unit = distance_and_unit(receive_positions, px)
            rx_time = rx_dist / sound_speed

    n_tx = int(analytic.shape[0])
    P = int(px.shape[0])
    accum = ops.zeros((P,), dtype="complex64" if compounding == "coherent" else "float32")
    hits = ops.zeros((P,), dtype="float32")

    for i in range(0, n_tx, tx_chunk):
        j = min(i + tx_chunk, n_tx)
        tx_pos = transmit_origins[i:j]

        if use_sos:
            tx_time = straight_ray_times(
                tx_pos,
                px,
                sos_map,
                sos_grid_x,
                sos_grid_z,
                sound_speed,
                n_sos_ray_samples,
            )
            _, tx_unit = distance_and_unit(tx_pos, px)
        else:
            tx_dist, tx_unit = distance_and_unit(tx_pos, px)
            tx_time = tx_dist / sound_speed

        if per_tx_rx:
            rx_here = receive_positions[i:j]  # (c, n_el, 2)

            def _rx_time_one(rx_one):
                if use_sos:
                    return straight_ray_times(
                        rx_one, px, sos_map, sos_grid_x, sos_grid_z, sound_speed, n_sos_ray_samples
                    )
                rd, _ = distance_and_unit(rx_one, px)
                return rd / sound_speed

            def _rx_unit_one(rx_one):
                return distance_and_unit(rx_one, px)[1]

            rx_time_c = vmap(_rx_time_one)(rx_here)  # (c, n_el, P)
            rx_unit_c = vmap(_rx_unit_one)(rx_here)  # (c, n_el, P, 2)

            if use_sos:

                def _direct_one(tx_one, rx_one):
                    return straight_ray_times(
                        tx_one[None],
                        rx_one,
                        sos_map,
                        sos_grid_x,
                        sos_grid_z,
                        sound_speed,
                        n_sos_ray_samples,
                    )[0]

                direct = vmap(_direct_one, in_axes=(0, 0))(tx_pos, rx_here)  # (c, n_el)
            else:
                direct = _pairwise_batched_direct(tx_pos, rx_here) / sound_speed  # (c, n_el)
            t_round = tx_time[:, None, :] + rx_time_c
            direct = direct[:, :, None]
            cos = ops.sum(tx_unit[:, None, :, :] * rx_unit_c, axis=-1)
            trace = ops.transpose(analytic[i:j], (0, 2, 1))  # (c, n_el, n_ax)
        else:
            t_round = tx_time[:, None, :] + rx_time[None, :, :]  # (c, n_el, P)
            if use_sos:
                direct = straight_ray_times(
                    tx_pos,
                    receive_positions,
                    sos_map,
                    sos_grid_x,
                    sos_grid_z,
                    sound_speed,
                    n_sos_ray_samples,
                )[:, :, None]
            else:
                direct = (
                    compute_receive_distances(tx_pos, receive_positions)[:, :, None] / sound_speed
                )
            cos = ops.sum(tx_unit[:, None, :, :] * rx_unit[None, :, :, :], axis=-1)
            trace = ops.transpose(analytic[i:j], (0, 2, 1))  # (c, n_el, n_ax)

        sample_pos = (t_round - initial_times[i:j][:, None, None]) * fs
        valid = ops.logical_and(sample_pos >= 0, sample_pos < n_ax)
        if reject_transmission:
            valid = ops.logical_and(valid, t_round > (direct + transmission_guard_s))

        if backscatter_apodization:
            # Keep only backscatter geometries (cos > 0); the clamp also zeroes
            # the weight there, so no separate masking is needed.
            weight = ops.cast(valid, "float32") * ops.where(cos > 0.0, cos, 0.0)
        else:
            weight = ops.cast(valid, "float32")

        amp = _gather_time(trace, sample_pos, n_ax, interpolation)  # (c, n_el, P)
        weight_c = ops.cast(weight, "complex64")

        if compounding == "coherent":
            accum = accum + ops.sum(amp * weight_c, axis=(0, 1))
            hits = hits + ops.sum(weight, axis=(0, 1))
        else:
            # Coherent within one transmit's receive aperture, then averaged
            # (incoherently) across transmits: robust to phase decorrelation
            # between transmits, at the cost of some resolution.
            per_tx = ops.sum(amp * weight_c, axis=1)  # (c, P)
            per_tx_hits = ops.sum(weight, axis=1)  # (c, P)
            per_tx_amp = ops.abs(per_tx) / (per_tx_hits + 1e-6)
            accum = accum + ops.sum(per_tx_amp, axis=0)
            hits = hits + ops.sum(ops.cast(per_tx_hits > 0, "float32"), axis=0)

    if compounding == "coherent":
        return ops.abs(accum) / (hits + 1e-6)
    return accum / (hits + 1e-6)
