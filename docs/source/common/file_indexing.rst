The indices parameter can be used to load a subset of the data. This can be

- ``'all'`` or ``None`` to load all data

- an ``int`` to load a single frame

- a ``List[int]`` to load specific frames

- a ``Tuple[Union[list, slice, int], ...]`` to index multiple axes (i.e. frames and transmits). Note that
    indexing with lists of indices for multiple axes is not supported. In that case,
    try to define one of the axes with a slice for optimal performance. Alternatively,
    slice the data after loading.

For more information on the indexing options,
see `indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>`_ and
`fancy indexing in h5py <https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing>`_.