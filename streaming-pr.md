# HDF5 chunks, compression and streaming

I noticed loading a compressed hdf5 file was quite slow. I found out that HDF5 cannot natively do
concurrent decompression of chunks, and that the default chunk size was not really suitable for
how we load channel data generally.

This PR introduces various fixes to improve that.

- It switches compression algotrithm to a modern one (...)
- It decompresses chunks in parallel using a thread pool
- It changes the default chunk size to be more suitable for our use case
- It adds a streaming interface to `zea.File` for huggingface datasets

## Speed-ups in common use cases

Just checking the summary of a cloud file:

```python
import zea
with zea.File("...", stream=True) as f:
    print(f.summary()) # 300 ms, previously would have to download entire file!
```

Loading a couple of frames from a cloud file:

```python
import zea
with zea.File("...", stream=True) as f:
    raw_data = f.data.raw_data[:3] # streams chunks from cloud, decompresses in parallel, returns numpy array

Loading a local file:

```python
import zea
with zea.File("...") as f:
    raw_data = f.data.raw_data[:3] # concurrently reads and decompress chunks from disk, returns numpy array
```

## Benchmark
