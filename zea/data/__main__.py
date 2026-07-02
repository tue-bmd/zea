"""Entry point for ``python -m zea.data``.

Dispatches to the zea data file manipulation CLI defined in
:mod:`zea.data.file_operations`.
"""

from zea.data.file_operations import main

if __name__ == "__main__":
    main()
