"""
This file searches all notebooks in the folder examples/notebooks and then runs
a test for each notebook that executes it using papermill. The test fails if
if any of the cells in the notebook raise an exception.
"""
from pathlib import Path
import shutil
import pytest
import papermill as pm

# Find all notebooks in the folder examples
notebook_paths = set(Path('examples').rglob('*.ipynb'))

# Specify notebook names to be removed
notebooks_to_remove = set([])

# Filter out the undesired notebook names
notebook_paths = notebook_paths - notebooks_to_remove

print(f'Found {len(notebook_paths)} notebooks to test.')

@pytest.mark.parametrize("notebook_path", notebook_paths)
def test_notebook_run(notebook_path):
    """Runs the notebook at notebook path and fails if any of the cells raise
    an exception."""

    output_dir = Path('temp', 'notebooks_run_by_pytest')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / Path(notebook_path).name

    # Run notebook using the papermill module
    # Papermill will run the notebook and raise an exception if any of the
    # cells raise an exception.
    #
    # The parameters keyword argument argument causes papermill to look for a
    # cell with the tag "parameters" and inserts a cell below it setting the
    # variables in the dictionary as local variables in the notebook.
    # This means that the variable quick_mode will be overwritten to be True
    # when running the test. This is useful for notebooks that take a long time
    # to run, but we want to run them quickly for testing purposes.
    # Note that for this to work we need to add the tag "parameters" to the
    # cell that we want to use as the parameters cell.
    try:
        pm.execute_notebook(
           notebook_path,
           output_path,
           parameters={'quick_mode':True}
        )
    except pm.exceptions.PapermillExecutionError:
        assert False, 'Error executing the notebook with papermill.'

    # Remove temporary directory
    shutil.rmtree(output_dir)
