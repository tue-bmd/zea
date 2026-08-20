"""Reflectivity reconstruction for ultrasound computed tomography (USCT)."""

from keras import ops

from zea.func.ultrasound import channels_to_analytic
from zea.func.usct import usct_reflectivity_das
from zea.internal.core import DataTypes
from zea.internal.registry import ops_registry
from zea.ops.base import Operation

__all__ = ["USCTReflectivityDAS"]

# Columns of a zea (x, y, z) coordinate array that span the XZ imaging plane.
_IN_PLANE = [0, 2]


@ops_registry("usct_reflectivity_das")
class USCTReflectivityDAS(Operation):
    """Round-trip TOF DAS reflectivity for Ultrasound Computed Tomography acquisitions.

    USCT acquisitions do not fit zea's standard linear/phased-array B-mode pipeline
    (:class:`~zea.ops.Beamform` with :class:`~zea.ops.TOFCorrection`). In USCT the
    transmit events originate from **individual point sources**, either single ring
    elements firing in turn (full-ring tomographs) or dedicated emitters placed
    around the medium, rather than a wavefront steered *from* the receive aperture
    with per-element ``t0_delays`` and ``focus_distances``. The standard
    :class:`~zea.ops.TOFCorrection` derives its transmit time-of-flight from that
    steered-wavefront model and cannot represent an off-aperture point source, so a
    dedicated operation is required.

    This operation implements a round-trip time-of-flight Delay-And-Sum (DAS)
    reflectivity image that is the natural common denominator of these geometries.
    For every pixel it coherently sums, over all transmit/receive pairs, the
    analytic channel signal sampled at the round-trip delay

    .. math::

        \\tau_{t,r}(\\mathbf{p}) =
            \\frac{\\lVert \\mathbf{p} - \\mathbf{s}_t \\rVert}{c}
            + \\frac{\\lVert \\mathbf{p} - \\mathbf{e}_r \\rVert}{c}
            - t_{0,t},

    where :math:`\\mathbf{s}_t` is the transmit point-source position,
    :math:`\\mathbf{e}_r` the receive-element position, :math:`c` the sound speed and
    :math:`t_{0,t}` the per-transmit time-zero. This implementation is loosely
    based on the reflection ultrasound computed tomography (RUCT) approach for
    ring-array systems described below.

    .. admonition:: Reference

       B. Lafci, J. Robin, X. L. Deán-Ben and D. Razansky.
       *Expediting Image Acquisition in Reflection Ultrasound Computed Tomography.*
       IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
       69(10):2837-2848, 2022.
       `DOI: 10.1109/TUFFC.2022.3172713 <https://doi.org/10.1109/TUFFC.2022.3172713>`_

    .. seealso::

        See the `pyruct <https://github.com/berkanlafci/pyruct>`_
        repository for a reference implementation.

    A few options make it usable on the strongly-transmissive ring / dual-panel
    geometries that USCT uses:

    - **Transmission rejection** (``reject_transmission``): discard the direct
      through-transmission arrival (which dwarfs the backscatter) by keeping only
      round-trip delays that exceed the straight-line source→element time by a guard
      interval.
    - **Backscatter apodization** (``backscatter_apodization``): weight each pair by
      the cosine of the angle between the pixel→source and pixel→element directions,
      keeping only geometries where the receiver looks back toward the illumination
      (``cos > 0``).
    - **Compounding** (``compounding``): ``"coherent"`` (default) sums the analytic
      signal across every transmit/receive pair as one complex sum before taking
      the magnitude once, which gives the best resolution when the delay model is
      accurate everywhere. ``"incoherent"`` instead takes the magnitude per
      transmit (coherent only across that transmit's receive aperture) and
      averages those magnitudes across transmits, trading some resolution for
      robustness to phase decorrelation between transmits caused by sound-speed
      mismatch or calibration error — decorrelation that grows with the aperture
      spanned by a full ring, where transmit/receive pairs can be far apart and
      see very different propagation paths.
    - Optionally, a spatial **speed-of-sound map** can be supplied
      (``sos_map``/``sos_grid_x``/``sos_grid_z``) to replace the constant-``c`` delays
      with a straight-ray integral of the local slowness — useful when a ground-truth
      or estimated SoS map is available and the medium has large sound-speed contrast.

    Like the rest of zea's beamforming stack, the operation images the **XZ plane**,
    with ``y`` as the elevation (out-of-plane) axis. It consumes the standard zea
    parameters: ``flatgrid``, ``probe_geometry``, ``transmit_origins``,
    ``sampling_frequency``, ``initial_times`` and ``sound_speed``, and projects them
    onto the imaging plane internally, so a :class:`~zea.ops.Pipeline` can be driven
    straight from a file's parameters. Ring tomographs should therefore store their
    ring in the XZ plane (``y == 0``), the same convention a linear array uses.

    Each pixel is reconstructed independently, so, exactly like
    :class:`~zea.ops.Beamform`, the grid can be processed in patches to bound peak
    memory, and reshaped to an image afterwards::

        Cast -> PatchedGrid([USCTReflectivityDAS]) -> ReshapeGrid -> Normalize -> LogCompress

    Without patching, the receive-leg geometry alone costs ``O(n_el * n_pix)``, which
    is what makes a full-resolution grid blow up.

    Accepts raw RF (``n_ch == 1``, where it is demodulated with Hilbert internally),
    or two-channel I/Q (``n_ch == 2``).
    """

    def __init__(
        self,
        tx_chunk=4,
        reject_transmission=True,
        transmission_guard_s=2.5e-6,
        backscatter_apodization=True,
        interpolation="linear",
        compounding="coherent",
        n_sos_ray_samples=16,
        axial_axis=1,
        **kwargs,
    ):
        # Processes one frame at a time (n_tx as the leading axis, not a batch of frames).
        kwargs.setdefault("with_batch_dim", False)
        super().__init__(output_data_type=DataTypes.ENVELOPE_DATA, **kwargs)
        self.tx_chunk = tx_chunk
        self.reject_transmission = reject_transmission
        self.transmission_guard_s = transmission_guard_s
        self.backscatter_apodization = backscatter_apodization
        self.interpolation = interpolation
        self.compounding = compounding
        self.n_sos_ray_samples = n_sos_ray_samples
        self.axial_axis = axial_axis

    def call(
        self,
        flatgrid=None,
        probe_geometry=None,
        transmit_origins=None,
        sampling_frequency=None,
        initial_times=None,
        sound_speed=None,
        sos_map=None,
        sos_grid_x=None,
        sos_grid_z=None,
        **kwargs,
    ):
        data = kwargs[self.key]

        das_kwargs = dict(
            transmit_origins=ops.take(transmit_origins, _IN_PLANE, axis=-1),
            receive_positions=ops.take(probe_geometry, _IN_PLANE, axis=-1),
            # flatgrid is (n_pix, 3) in (x, y, z); image the XZ plane.
            pixels=ops.take(flatgrid, _IN_PLANE, axis=-1),
            sampling_frequency=sampling_frequency,
            initial_times=initial_times,
            sound_speed=sound_speed,
            tx_chunk=self.tx_chunk,
            reject_transmission=self.reject_transmission,
            transmission_guard_s=self.transmission_guard_s,
            backscatter_apodization=self.backscatter_apodization,
            interpolation=self.interpolation,
            compounding=self.compounding,
            sos_map=sos_map,
            sos_grid_x=sos_grid_x,
            sos_grid_z=sos_grid_z,
            n_sos_ray_samples=self.n_sos_ray_samples,
        )

        def _reconstruct_one(data_one):
            analytic = channels_to_analytic(data_one, axis=self.axial_axis)  # (n_tx, n_ax, n_el)
            return usct_reflectivity_das(analytic, **das_kwargs)

        if not self.with_batch_dim:
            img = _reconstruct_one(data)
        else:
            num_frames = ops.shape(data)[0]
            img = ops.stack([_reconstruct_one(data[i]) for i in range(num_frames)], axis=0)

        return {self.output_key: img}
