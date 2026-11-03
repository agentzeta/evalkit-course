import sys

from .cli import main

try:
    code = main()
except BrokenPipeError:
    # Output pipe closed early (e.g. `| head`). Not an error in the tool:
    # exit quietly rather than printing a traceback over the user's terminal.
    try:
        sys.stdout.close()
    except BrokenPipeError:
        pass
    code = 0

raise SystemExit(code)
