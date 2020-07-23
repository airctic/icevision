# Mantisshrimp Documentation

The source for Mantisshrimp documentation is in the `docs/` folder.
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- Install dependencies: `pip install -r docs/requirements.txt`
- `pip install -e .` to make sure that Python will import your modified version of Mantisshrimp.
- From the root directory, `cd` into the `docs/` folder and run:
    - `python autogen.py` # Generate md files, and copy assets into the `docs_dir/` folder specified in `mkdocs.yml` file
    - `mkdocs serve`      # Starts a local webserver:  [localhost:8000](http://localhost:8000)
    
 - To build the `site/`, from the root directory, `cd` into the `docs/` folder and run:  
    - `mkdocs build`    # Builds a static site in the `site/` directory

- To locally test the `site/`, from the `docs/` directory, `cd` into the `site/` folder and run:
    - `python -m http.server`
