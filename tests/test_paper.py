"""Test module for validating Python code snippets in paper.md."""

import re
from pathlib import Path

import pytest


def extract_python_snippets(markdown_file):
    """Extract Python code snippets from a markdown file.

    Args:
        markdown_file: Path to the markdown file

    Returns:
        List of tuples (snippet_name, snippet_code, line_number)
    """
    content = Path(markdown_file).read_text()

    # Find all Python code blocks
    pattern = r"```python\n(.*?)```"
    matches = re.finditer(pattern, content, re.DOTALL)

    snippets = []
    for idx, match in enumerate(matches):
        code = match.group(1)
        # Calculate approximate line number
        line_num = content[: match.start()].count("\n") + 1

        # Infer snippet name from content
        name = _infer_snippet_name(code, idx)
        snippets.append((name, code, line_num))

    return snippets


def _infer_snippet_name(code, idx):
    """Infer a descriptive name for a code snippet based on its content."""
    code_lower = code.lower()

    # Check for key patterns to identify the snippet
    if "zea.file" in code_lower and "load_data" in code_lower:
        return "data_file_loading"
    elif "make_dataloader" in code_lower:
        return "data_dataloader"
    elif "pipeline" in code_lower and any(
        op in code_lower for op in ["demodulate", "delayandsum", "beamform"]
    ):
        return "pipeline_beamforming"
    elif "diffusionmodel" in code_lower:
        return "models_diffusion"
    elif "greedy" in code_lower or "agent" in code_lower:
        return "agent_selection"
    else:
        return f"snippet_{idx}"


def prepare_snippet_for_testing(code):
    """Prepare a code snippet for testing by making necessary modifications.

    Args:
        code: The original code snippet

    Returns:
        Modified code that can be executed in tests
    """
    # Replace placeholder path with actual working path
    code = code.replace(
        'path = "hf://zeahub/..."',
        'path = "hf://zeahub/picmus/database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"',
    )

    # Break out of dataloader loop after first iteration
    code = code.replace(
        "for batch in dataloader:\n    ... # your training loop here",
        "for batch in dataloader:\n    break  # just test one iteration",
    )

    # Reduce model sampling parameters for faster testing
    code = code.replace(
        "model.sample(n_samples=16, n_steps=90, verbose=True)",
        "model.sample(n_samples=1, n_steps=2, verbose=False)",
    )

    # Replace placeholder data with actual fake data
    # Look for the pattern where data = ... is used
    if "data = ..." in code:
        # Need to inject fake data before the agent creation
        # Extract dimensions from the code if possible
        lines = code.split("\n")
        new_lines = []
        for line in lines:
            if "data = ..." in line:
                # Add fake data generation
                new_lines.append("# Generate fake data for testing")
                new_lines.append("width = 128")
                new_lines.append("height = 128")
                new_lines.append("batch_size = 4")
                new_lines.append("min_val = -60")
                new_lines.append("dynamic_range = (-60, 0)")
                new_lines.append(
                    "data = keras.random.uniform((batch_size, height, width), minval=-60, maxval=0)"
                )
            else:
                new_lines.append(line)
        code = "\n".join(new_lines)

    # Add warning filter at the beginning to suppress escape sequence warnings
    # These can come from docstrings or comments in the paper snippets
    code = "import warnings\nwarnings.filterwarnings('ignore', category=SyntaxWarning)\n" + code

    return code


@pytest.fixture
def paper_snippets():
    """Extract all Python snippets from paper.md."""
    paper_path = Path(__file__).parent.parent / "paper" / "paper.md"
    return extract_python_snippets(paper_path)


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on actual number of snippets."""
    if "snippet_name" in metafunc.fixturenames:
        paper_path = Path(__file__).parent.parent / "paper" / "paper.md"
        snippets = extract_python_snippets(paper_path)
        # Use snippet names as test IDs for better readability
        metafunc.parametrize(
            "snippet_name,snippet_code,snippet_line",
            [(name, code, line) for name, code, line in snippets],
            ids=[name for name, _, _ in snippets],
        )


@pytest.mark.heavy
def test_paper_snippet(snippet_name, snippet_code, snippet_line):
    """Test that each code snippet from paper.md runs without errors."""
    # Prepare the snippet for testing
    code = prepare_snippet_for_testing(snippet_code)

    # Execute the code and check for runtime errors
    try:
        exec(code)
    except Exception as e:
        pytest.fail(
            f"Snippet '{snippet_name}' (starting at line {snippet_line}) failed with error: {e}\n"
            f"Modified code:\n{code}"
        )
