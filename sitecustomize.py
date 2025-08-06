import sys

try:
    # On Linux/Azure this will succeed and override sqlite3
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    # On Windows (where the wheel isn't available), just fall back
    pass

import collections
import collections.abc

# Patch Sequence + MutableSequence for Python 3.11 compatibility
for name in ("Sequence", "MutableSequence"):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
