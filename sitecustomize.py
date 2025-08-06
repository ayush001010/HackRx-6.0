
import collections
import collections.abc

# Patch Sequence + MutableSequence for Python 3.11 compatibility
for name in ("Sequence", "MutableSequence"):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
