# Notebooks
The notebooks in this folder are automatically tested using the `test_run_notebooks` unit test.

Warning: For the test to succeed there must be a cell in the notebook that is tagged with `parameters`.
The unit test will then execute the notebook and inject the line `quick_mode = True` in the parameters cell.
This enables you to make a quick version of the notebook where data loading is skipped or the number of iterations is reduced to be more suitable for testing.