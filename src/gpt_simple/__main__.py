"""Entry point for ``python -m gpt_simple``.

Kept separate from the library modules on purpose: ``gpt_simple/__init__.py``
re-exports the public API (including ``train``), so launching a re-exported
submodule with ``-m`` would make runpy import it twice and emit a
RuntimeWarning.  ``__main__`` is never imported by ``__init__``, so running
``-m gpt_simple`` loads the package once and dispatches cleanly.

The distributed launcher (``cli/train_cmd.py``) shells out to this entry point.
"""

from gpt_simple.train import _module_main

if __name__ == "__main__":
    _module_main()
