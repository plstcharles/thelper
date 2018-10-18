=============
Documentation
=============

The sphinx documentation is generated automatically via `readthedocs.io <https://readthedocs.org/projects/thelper/>`_, but it might
still be incomplete due to buggy apidoc usage/platform limitations. To build it yourself, use the makefile::

  $ cd <THELPER_ROOT>
  $ make docs

The HTML documentation should then be generated inside ``<THELPER_ROOT>/docs/build/html``. To browse it, simply open the
``index.html`` file there.
