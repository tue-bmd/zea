""" Check that all Python files in the project can be compiled, and that no import errors occur, for
example due to missing dependencies in the requirements.txt file. """

import glob
import traceback
from pathlib import Path


def check_imports_errors(directory):
    """ Check all Python files in a directory for import errors. """
    python_files = glob.glob(f"{directory}/**/*.py", recursive=True)

    for python_file in python_files:
        print(python_file)

    success = True
    for python_file in python_files:
        try:
            # Attempt to compile the Python file (checks for import errors)
            with open(python_file, 'rb') as file:
                compile(file.read(), python_file, 'exec')
        except SyntaxError as e:
            print(f"Syntax error in {python_file}:\n{e}")
            success = False
        except ImportError as e:
            print(f"Import error in {python_file}:\n{e}")
            success = False
        except Exception as e:
            print(f"Error in {python_file}:\n{e}")
            traceback.print_exc()
            success = False

    assert success, "Import errors found in one or more Python files."

if __name__ == '__main__':
    directory_to_check = Path(__file__).parent.parent
    check_imports_errors(directory_to_check)
