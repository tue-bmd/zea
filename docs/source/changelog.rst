:tocdepth: 2

.. _changelog:

Changelog
=========

This changelog is for the ``main`` branch. It lists the changes in each release, with the pull request and author of each change. Changes merged since the last release are listed under **Current**; to add one, see :ref:`changelog-entry` in the contributing guide. Older releases are also on `GitHub Releases <https://github.com/tue-bmd/zea/releases>`_.

..
   To add an entry, follow the "Log your changes" step in contributing.rst: one bullet
   under the matching subsection of "Current", ending with (:gh:`681` by :user:`your-handle`).

   The "Release PR" workflow moves "Current" under a new version heading. It uses
   .github/scripts/changelog.py, which relies on the heading layout below: keep the
   "Current" and "vX.Y.Z (YYYY-MM-DD)" headings underlined with "-", and the
   subsections underlined with "^".


Current
-------

New Features
^^^^^^^^^^^^
- ADMIRE (Aperture Domain Model Image REconstruction), a model-based beamformer that suppresses off-axis and multipath clutter, as :class:`zea.ops.ADMIRE` or ``zea.ops.Beamform(beamformer="admire")``. See :mod:`zea.beamform.admire` (:gh:`650` by :user:`tristan-deep`)
- UltraPIPS, a perceptual metric for B-mode ultrasound built on an ultrasound foundation model: :class:`zea.models.ultrapips.UltraPIPS`, or the ``ultrapips`` metric in :mod:`zea.metrics`, used like ``lpips`` (:gh:`649` by :user:`tristan-deep`)
- Transparent GIFs from RGBA frames with ``transparent=True`` in :func:`zea.io_lib.save_video` and :func:`zea.io_lib.save_to_gif` (:gh:`663` by :user:`tristan-deep`)
- GIFs that play once, with ``loop=None`` in :func:`zea.io_lib.save_to_gif` (:gh:`680` by :user:`wesselvannierop`)
- ``zea app`` can serve over HTTPS with ``--ssl-certfile`` and ``--ssl-keyfile`` (:gh:`648` by :user:`Louisvh`)
- Frequency-dependent attenuation in the pressure field, with ``attenuation_coef`` in dB/(cm·MHz) in :func:`zea.beamform.pfield.compute_pfield`. It defaults to 0 (lossless) (:gh:`682` by :user:`wesselvannierop`)
- :class:`zea.ops.BandPassFilter` and :class:`zea.ops.LowPassFilterIQ` take their ``bandwidth`` from the new :attr:`zea.Parameters.bandwidth`, derived as ``center_frequency * probe_bandwidth_percent / 100`` unless set or ``center_frequency`` varies across transmits (:gh:`644` by :user:`tristan-deep`)

Changed
^^^^^^^
- Streaming ``hf://`` datasets no longer runs into the Hugging Face rate limit, so you can train on many batches with many workers without downloading the data. The API is unchanged; two new environment variables, ``ZEA_MAX_CONCURRENT_REQUESTS`` and ``ZEA_FIND_H5_SHAPES_WORKERS``, tune it (see :doc:`environment`) (:gh:`671` by :user:`tristan-deep`)
- Conversion scripts for EchoNet-Dynamic, CAMUS and PICMUS accept the ``.zip`` file itself. EchoNet-Dynamic reproduces the published split by default, and its polar conversion is about 15 times faster (:gh:`676` by :user:`wesselvannierop`)
- (Breaking) In ``compute_pfield``, ``alpha`` and ``percentile`` also shape the field when ``norm=False``, the unused ``n_el`` argument is removed, and probes other than linear and phased arrays raise ``NotImplementedError`` (:gh:`657` by :user:`wesselvannierop`)
- ``Parameters.copy()`` and ``set_transmits`` are much faster (:gh:`656` by :user:`wesselvannierop`)
- The default ``probe_bandwidth_percent`` is 100 instead of 200, which also narrows the band used by ``compute_pfield`` (:gh:`644` by :user:`tristan-deep`)

Fixed
^^^^^
- The pressure field uses the one-way transmit spectrum, averages over frequencies instead of summing, and takes the element positions from ``probe_geometry`` (:gh:`657` by :user:`wesselvannierop`)
- ``t_peak`` estimated from ``waveforms_two_way`` uses the envelope peak; values change where the carrier peak differs from it (:gh:`656` by :user:`wesselvannierop`)
- Conversion of Verasonics ``BS100BW`` and ``BS50BW`` data samples I and Q at the same time instants. Data converted with earlier versions, such as some Open-H-RF data, may be affected (:gh:`672` by :user:`vincentvdschaft`)
- :class:`zea.ops.Normalize` clips its output instead of its input, so results stay within ``output_range`` (:gh:`681` by :user:`wesselvannierop`)
- ``waveforms_one_way`` and ``waveforms_two_way`` follow ``selected_transmits``, like the other per-transmit parameters (:gh:`683` by :user:`wesselvannierop`)
- The shared GIF palette keeps exact colors, so pure black stays ``(0, 0, 0)`` (:gh:`680` by :user:`wesselvannierop`)
- Deriving ``xlims`` on a GPU with the torch backend (:gh:`661` by :user:`tristan-deep`)
- Jitted ops on the torch backend in the Docker images, which lacked a C++ compiler (:gh:`658` by :user:`wesselvannierop`)


v0.1.8 (2026-10-01)
-------------------

New Features
^^^^^^^^^^^^
- Pipeline configs can declare the modules that define their custom operations, under ``pipeline: imports:`` or with ``zea process --import`` (:gh:`633` by :user:`tristan-deep`)

Changed
^^^^^^^
- ``zea app`` is more robust and easier to use, with a reworked UI. Mistyped paths are normalized, configs are auto-detected and file metadata loads with a progress bar (:gh:`642` by :user:`tristan-deep`)

Fixed
^^^^^
- Slow compilation of the beamforming pipeline on GPU in some settings, by disabling XLA's fusion autotuner only for :class:`zea.ops.TOFCorrection` (:gh:`638` by :user:`tristan-deep`)
- :class:`zea.ops.TOFCorrection`, and therefore :class:`zea.ops.Beamform`, applies lens correction (``apply_lens_correction=True``) correctly for curved probes (:gh:`634` by :user:`Louisvh`)
- ``zea.Dataloader(sort_files=True)`` no longer sorts on the wrong number in filenames like ``prefix_X_Y.hdf5`` (:gh:`637` by :user:`tristan-deep`)


v0.1.7 (2026-09-21)
-------------------

New Features
^^^^^^^^^^^^
- Conversion script for us4us (ARRUS + gui4us) data. Try it with ``zea convert us4us --help`` (:gh:`611` by :user:`tristan-deep`)
- ``zea process`` supports ``--save-as png`` (:gh:`626` by :user:`tristan-deep`)
- :class:`zea.File` accepts receive sub-apertures (fewer receive channels than probe elements), defined in ``scan.rx_aperture_indices``, with the matching ``parameters.rx_aperture_indices`` and ``parameters.n_rx`` (:gh:`615` by :user:`tristan-deep`)

Changed
^^^^^^^
- ``zea data`` file operations stream large files in batches instead of loading the whole file into memory (:gh:`620` by :user:`tristan-deep`)

Fixed
^^^^^
- Demodulation pads the signal, so the analytic-signal transform no longer wraps echoes around to the first depth samples (:gh:`629` by :user:`Felixduelmer`)
- Streaming ``hf://`` files no longer requires Hugging Face authentication, and uses your token when you are logged in (:gh:`625` by :user:`Felixduelmer`)


v0.1.6 (2026-09-11)
-------------------

New Features
^^^^^^^^^^^^
- :class:`zea.Dataloader` can return metadata alongside the data: scan and probe parameters, annotations and more (:gh:`578` by :user:`wesselvannierop`)
- InversionNet speed-of-sound reconstruction model, :class:`zea.models.inversionnet.InversionNet` (:gh:`607` by :user:`tristan-deep`)
- ABLE (Adaptive Beamforming by Deep Learning) model, :class:`zea.models.able.ABLE` (:gh:`357` by :user:`benluijten`)
- :class:`zea.ops.Refocus` supports IQ data (:gh:`570` by :user:`RobinVV`)
- Histogram matching with :func:`zea.display.histogram_match` (:gh:`532` by :user:`wesselvannierop`)
- :mod:`zea.simulator` has an elevation lens option (:gh:`566` by :user:`Louisvh`), noise and TGC options (:gh:`573` by :user:`Louisvh`) and a configurable scattering exponent (:gh:`587` by :user:`Louisvh`)
- ``zea tools`` CLI, which for now holds a selection tool for manual ROI annotation. Try it with ``zea tools --help`` (:gh:`579` by :user:`tristan-deep`)
- ``zea process`` supports ``--track`` (:gh:`592` by :user:`OisinNolan`)

Changed
^^^^^^^
- (Breaking) Some :class:`zea.Dataloader` arguments were renamed, deprecated or got new defaults; see :gh:`578` for the full list (:gh:`578` by :user:`wesselvannierop`)
- :mod:`zea.simulator` is 30 to 600 times faster, by factoring out terms shared by the transmit and receive paths (:gh:`550` by :user:`Louisvh`)
- ``zea`` automatically uses any installed backend when none is set (:gh:`585` by :user:`wesselvannierop`)
- ``zea.log`` no longer writes a log file (:gh:`602` by :user:`wesselvannierop`)

Fixed
^^^^^
- Slow JIT compilation of :class:`zea.Pipeline` for large grids, with the new ``patch_size`` argument, now preferred over ``num_patches`` (:gh:`596` by :user:`wesselvannierop`)
- The spread model of the simulator (:gh:`566` by :user:`Louisvh`)
- Automatic ``xlims`` for plane and diverging waves (:gh:`495` by :user:`tristan-deep`)
- Verasonics conversion with a single image buffer (:gh:`598` by :user:`OisinNolan`), or without ``ImgDataP`` in the workspace (:gh:`563` by :user:`vincentvdschaft`)
- Off-by-one error in ``prepare_schedule`` of the diffusion models (:gh:`601` by :user:`wesselvannierop`)
- Pipeline warnings no longer flag valid parameters (:gh:`603` by :user:`tristan-deep`)


v0.1.5 (2026-08-16)
-------------------

New Features
^^^^^^^^^^^^
- :class:`zea.Pipeline` can run in mixed precision, which speeds up a DAS B-mode pipeline another 1.5 times (:gh:`518` by :user:`wesselvannierop`)
- :class:`zea.Pipeline` supports JIT compilation on the PyTorch backend, through ``torch.compile()`` (:gh:`556` by :user:`swpenninga`)
- Synthetic aperture functions :func:`zea.func.decode_hadamard` and :func:`zea.func.construct_acquisition_from_synthetic_aperture` (:gh:`500` by :user:`vincentvdschaft`)
- ``t_peak`` in :mod:`zea.simulator` (:gh:`544` by :user:`vincentvdschaft`)

Changed
^^^^^^^
- (Breaking) Python 3.10 is no longer supported; ``zea`` requires Python 3.11 or newer. Users on Python 3.10 should stay on ``zea<=0.1.4`` (:gh:`534` by :user:`tristan-deep`)
- (Breaking) ``sample()`` and ``posterior_sample()`` of the :mod:`zea.models.generative` models both work on ``(n_samples, *input_shape)``. ``posterior_sample()`` used to work on ``(batch_size, n_samples, *input_shape)``; use :func:`zea.func.vmap` for batches (:gh:`555` by :user:`wesselvannierop`)
- The DAS beamformer is up to 2 times faster and DMAS up to 10 times (:gh:`489` by :user:`Louisvh`)
- Efficient chunking, introduced for ``raw_data`` in v0.1.3, also applies to images, maps and small metadata arrays (:gh:`535` by :user:`wesselvannierop`)
- The Docker images are fully installed with uv (:gh:`546` by :user:`wesselvannierop`)

Fixed
^^^^^
- Definitions of ``zlims`` and ``distance_to_apex`` (:gh:`542` by :user:`tristan-deep`)
- Stochastic sampling in :class:`zea.models.flow_matching.FlowMatchingModel` preserves the ODE marginals (:gh:`551` by :user:`wesselvannierop`)
- ``flat_aligned_apodization`` can be set explicitly (:gh:`552` by :user:`wesselvannierop`)
- ``init_device`` no longer writes the visible GPU ids to ``CUDA_VISIBLE_DEVICES`` (:gh:`554` by :user:`wesselvannierop`)
- Non-unit gain in the upmix/downmix roundtrip (:gh:`565` by :user:`Louisvh`)


v0.1.4 (2026-07-23)
-------------------

New Features
^^^^^^^^^^^^
- Minimum variance beamformer, :class:`zea.ops.MinimumVariance` or ``zea.ops.Beamform(beamformer="minimum_variance")`` (:gh:`490` by :user:`tristan-deep`)
- New probes :class:`zea.probes.Verasonics_c5_2v` and :class:`zea.probes.Verasonics_p4_2v`, and ``f_number`` supports curved probes, computed with respect to the element normals (:gh:`516` by :user:`tristan-deep`)

Changed
^^^^^^^
- The CLI parses its arguments with tyro (:gh:`521` by :user:`tristan-deep`)

Fixed
^^^^^
- With ``h5py>=3.16`` installed, v0.1.3 wrote ``.hdf5`` files in the HDF5 2.0 format, which older h5py/HDF5 installs cannot open. The write format is now capped, so files open with old and new installs alike. Fix files saved with v0.1.3 with ``zea data resave old_file.hdf5 new_file.hdf5``, or read them with ``h5py>=3.16`` (:gh:`520` by :user:`tristan-deep`)


v0.1.3 (2026-07-17)
-------------------

New Features
^^^^^^^^^^^^
- Stream zea files from Hugging Face with ``zea.File("hf://...", stream=True)``: indexing or slicing a dataset downloads only that part. Resave datasets with ``zea data resave`` for the best performance (:gh:`504` by :user:`wesselvannierop`)
- Ultrasound computed tomography with :class:`zea.ops.USCTReflectivityDAS` and :func:`zea.func.usct_reflectivity_das` (:gh:`506` by :user:`tristan-deep`)
- New file fields ``metadata.subject.weight`` and ``metadata.subject.genetic_strain`` (:gh:`513` by :user:`OisinNolan`)
- Custom fields of a :class:`zea.File` can be indexed by name and with dots, like standard fields (:gh:`507` by :user:`tristan-deep`)
- ``Parameters.t_peak`` is derived from ``waveforms_two_way`` when available (:gh:`501` by :user:`tristan-deep`)
- ``zea data`` file operations accept ``hf://`` paths (:gh:`502` by :user:`tristan-deep`)

Changed
^^^^^^^
- HDF5 reads are up to 10 times faster and storage is about 2 times smaller, through new chunking and compression choices (:gh:`504` by :user:`wesselvannierop`)
- Field names in a zea file must be snake_case (:gh:`507` by :user:`tristan-deep`)
- ``zea data summary`` and ``File.summary()`` show the actual values, not just the field descriptions (:gh:`503` by :user:`tristan-deep`)
- :class:`zea.Pipeline` gives clearer warnings and errors (:gh:`494` by :user:`tristan-deep`), as does reading HDF5 files (:gh:`512` by :user:`tristan-deep`)
- Log files are stored in a fixed location, and logging no longer interferes with progress bars (:gh:`488` by :user:`wesselvannierop`)

Fixed
^^^^^
- Division by zero at ``z=0`` in the lens-corrected delay computation (:gh:`497` by :user:`tristan-deep`)
- :class:`zea.ops.TissueSuppression` uses the complex-valued transpose (:gh:`514` by :user:`OisinNolan`)


v0.1.2 (2026-07-09)
-------------------

New Features
^^^^^^^^^^^^
- Scanline beamforming, with the new :class:`zea.ops.AlignedApodization` and :class:`zea.ops.ReceiveApodization` ops and the ``enable_scanline``, ``flat_receive_apodization`` and ``flat_aligned_apodization`` parameters (:gh:`482` by :user:`tristan-deep`)
- New file fields ``metadata.subject.bmi`` (:gh:`469` by :user:`tristan-deep`) and ``data.attenuation_map`` (:gh:`472` by :user:`tristan-deep`)
- Transmit-only tracks with ``TrackSpec(transmit_only=True)``, which hold only ``scan`` parameters and no ``data``, e.g. for shear-wave push pulses or therapeutic exposures (:gh:`475` by :user:`OisinNolan`)
- :class:`zea.Dataset` and :class:`zea.Dataloader` take a ``file_filter`` argument to keep only files whose content matches a predicate (:gh:`470` by :user:`tristan-deep`)

Changed
^^^^^^^
- The ``zea`` CLI is split into ``zea data``, ``zea process`` and ``zea app``. Try it with ``zea --help`` (:gh:`477` by :user:`wesselvannierop`)

Fixed
^^^^^
- Discontinuity of the transmit delays at the focus, with the new ``focal_region_length`` parameter that smooths the transition (:gh:`481` by :user:`tristan-deep`)
- JIT compilation of nested pipelines (:gh:`465` by :user:`wesselvannierop`)
- Shape consistency across data types and tracks (:gh:`483` by :user:`wesselvannierop`)
- ``None`` values in the spec and parameters are no longer cast (:gh:`484` by :user:`wesselvannierop`)
- No warning about missing ``azimuth_angles`` for 2D grids (:gh:`476` by :user:`tristan-deep`)


v0.1.1 (2026-06-25)
-------------------

Changed
^^^^^^^
- The Docker images ship with up-to-date machine learning libraries (:gh:`458` by :user:`wesselvannierop`)
- ``set_transmits`` is replaced by ``find_transmits`` (:gh:`459` by :user:`wesselvannierop`)
- Masks accept arbitrary leading dimensions (:gh:`460` by :user:`wesselvannierop`)
- Improvements to saving Verasonics data (:gh:`457` by :user:`tristan-deep`)


v0.1.0 (2026-06-22)
-------------------

The first release with the new zea file format, designed for the `OpenH-RF initiative <https://github.com/tue-bmd/zea/issues/318>`_. This section condenses the pre-releases v0.1.0a0 to v0.1.0a5; their `release notes <https://github.com/tue-bmd/zea/releases>`_ have the details.

New Features
^^^^^^^^^^^^
- New zea file format, with many more standardized (meta)data fields. Files saved with older versions are still read (:gh:`307` by :user:`tristan-deep`, :gh:`368` by :user:`swpenninga`)
- A file stores a ``probe`` group (:class:`zea.data.spec.ProbeSpec`) next to the ``scan`` group (:gh:`379` by :user:`tristan-deep`)
- Multiple acquisition tracks in one file, e.g. B-mode and Doppler transmits that share the probe (:gh:`359` by :user:`OisinNolan`)
- ``data/aligned_data``, ``data/beamformed_data``, ``data/envelope_data`` and ``data/image`` store ``values`` and ``coordinates``, so their extent is known (:gh:`362` and :gh:`380` by :user:`OisinNolan`, :gh:`383` by :user:`swpenninga`)
- Custom elements in zea files with :class:`zea.data.file.CustomElement` (:gh:`450` by :user:`wesselvannierop`)
- Optional ``acquisition_time`` file attribute (:gh:`431` by :user:`tristan-deep`) and ``zea.File.timestamps`` for the global timestamps of all transmit events (:gh:`437` by :user:`OisinNolan`)
- 1D signals such as ``metadata.ecg`` can be stored with irregular ``timestamps`` (:gh:`427` by :user:`Felixduelmer`), and ``scan.time_to_next_transmit`` can omit its last value (:gh:`426` by :user:`Felixduelmer`)
- New ``zea`` CLI, with ``zea app`` (a Gradio app) and ``zea process`` (:gh:`416` by :user:`tristan-deep`)
- Custom operations in pipelines (:gh:`415` by :user:`tristan-deep`)
- Load large datasets lazily with ``zea.Dataset(..., lazy=True)`` (:gh:`419` by :user:`tristan-deep`)
- New ops :class:`zea.ops.Refocus` (:gh:`336` by :user:`swpenninga`) and :class:`zea.ops.TissueSuppression` (:gh:`417` by :user:`vincentvdschaft`)
- New models :class:`zea.models.speckle2self.Speckle2Self` (:gh:`377` by :user:`tristan-deep`), :class:`zea.models.flow_matching.FlowMatchingModel` (:gh:`378` by :user:`swpenninga`) and the Diffusion Transformer backbone :class:`zea.models.dit.DiTTimeConditional` (:gh:`413` by :user:`swpenninga`)
- Percentile-based normalization in :class:`zea.ops.Normalize` (:gh:`353` by :user:`wesselvannierop`)
- ``passband`` argument for :class:`zea.ops.BandPassFilter` (:gh:`356` by :user:`benluijten`)
- :class:`zea.Dataloader` has a ``shape`` attribute and seeds its pipeline (:gh:`339` by :user:`wesselvannierop`), can disable re-shuffling (:gh:`352` by :user:`wesselvannierop`) and can zero-pad short videos instead of skipping them (:gh:`447` by :user:`Louisvh`)
- ``zea data`` file operations work on folders (:gh:`354` by :user:`wesselvannierop`)
- Conversion scripts for PICMUS, CAMUS and CETUS store all metadata (:gh:`385` by :user:`tristan-deep`), a new EchoXFlow conversion script (:gh:`424` by :user:`swpenninga`), an improved EchoNet-LVH script (:gh:`435` by :user:`wesselvannierop`) and ``--upload`` to Hugging Face for the Verasonics conversion (:gh:`404` by :user:`OisinNolan`)

Changed
^^^^^^^
- (Breaking) ``zea.Scan`` is renamed to ``zea.Parameters``. Load all parameters with ``zea.File.load_parameters()`` and pass them to a pipeline with ``pipeline.prepare_parameters(parameters)``. ``probe.center_frequency`` is renamed to ``probe.probe_center_frequency``, and names in ``scan`` and ``probe`` must not overlap (:gh:`392` by :user:`wesselvannierop`)
- (Breaking) ``zea.Interface`` is removed in favor of the ``zea process`` CLI (:gh:`416` by :user:`tristan-deep`)
- (Breaking) ``zea.File.create`` no longer returns an open file, and ``probe.probe_geometry`` is required to save ``data/raw_data`` (:gh:`411` by :user:`wesselvannierop`)
- (Breaking) ``data/image_sc`` is no longer a standard field, as images carry their coordinates (:gh:`439` by :user:`swpenninga`)
- (Breaking) ``generate_zea_dataset`` is removed (:gh:`443` by :user:`swpenninga`)
- ``data/raw_data`` can be stored as ``int16``; add ``zea.ops.Cast(dtype="float32")`` at the start of a pipeline to avoid dtype errors (:gh:`307` by :user:`tristan-deep`)
- The default compression is ``lzf`` instead of ``gzip`` (:gh:`399` by :user:`tristan-deep`)
- Deprecate ``probe_name`` in favor of the ``probe`` group (:gh:`379` by :user:`tristan-deep`)
- Fewer and more relevant warnings when saving and loading files (:gh:`395` and :gh:`444` by :user:`tristan-deep`, :gh:`443` by :user:`swpenninga`)
- Config validation is redesigned (:gh:`412` by :user:`swpenninga`)
- Faster pressure field computation (:gh:`267` by :user:`wesselvannierop`)
- ``zea`` is installed and developed with uv instead of Poetry (:gh:`425` by :user:`tristan-deep`)

Fixed
^^^^^
- Near-field artifact in the pressure field (:gh:`267` by :user:`wesselvannierop`)
- Importing ``zea`` without tkinter installed (:gh:`432` by :user:`OisinNolan`)
