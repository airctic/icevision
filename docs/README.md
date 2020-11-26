# IceVision Documentation

The source for IceVision documentation is in the `docs/` folder.
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- Locally install the package as described [here](https://airctic.com/install/#option-2-installing-an-editable-package-locally-for-developers)
- From the root directory, `cd` into the `docs/` folder and run:
    - `python autogen.py`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](http://localhost:8000)
