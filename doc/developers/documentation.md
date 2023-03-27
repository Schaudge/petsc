# Developing PETSc Documentation

```{toctree}
:maxdepth: 2
```

## General Guidelines

- Good documentation should be like a bonsai tree: alive, on display, frequently tended, and as small as possible (adapted from [these best practices](https://github.com/google/styleguide/blob/gh-pages/docguide/best_practices.md)).
- Wrong, irrelevant, or confusing documentation is worse than no documentation.

(sphinx-documentation)=

## Documentation with Sphinx

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to build our web page and documentation.  Most content is written using [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), a simple markup language.

[These slides](https://gitlab.com/psanan/petsc-sphinx-slides) contain an overview of Sphinx and how we use(d) it, as of October, 2020.

(sec-local-html-docs)=

### Building the HTML docs locally

:::{admonition} Note
The documentation build with Sphinx involves configuring a minimal build
of PETSc and building some of the {any}`classic docs <classic_docs_build>`,
which requires local working `flex`, `gcc`, and  `g++` before
you follow the instructions below.
:::

We suggest using a [Python 3 virtual environment](https://docs.python.org/3/tutorial/venv.html)  [^venv-footnote].

```console
$ cd $PETSC_DIR
$ python3 -m venv petsc-doc-env
$ . petsc-doc-env/bin/activate
$ python3 -m pip install -r doc/requirements.txt
```

Then,

```console
$ cd doc
$ make html                      # may take several minutes
$ browse _build/html/index.html  # or otherwise open in browser
```

to turn off the Python virtual environment once you have built the documentation use

```console
$ deactivate
```

(sec-local-docs-latex)=

### Building the manual locally as a PDF via LaTeX

:::{admonition} Note
Before following these instructions, you should have a working
local LaTeX installation and the ability to install additional packages,
if need be, to resolve LaTeX errors.
:::

Set up your local Python environment (e.g. {ref}`as above <sec_local_html_docs>`), then

```console
$ cd doc
$ make latexpdf
$ open _build/latex/manual.pdf  # or otherwise open in PDF viewer
```

(sphinx-guidelines)=

### Sphinx Documentation Guidelines

Refer to Sphinx's [own documentation](https://https://www.sphinx-doc.org) for general information on how to use Sphinx, and note the following additional guidelines.

- Use the [literalinclude directive](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-literalinclude) to directly include pieces of source code. Use an "absolute" path, beginning with `/`, which means relative to the root for the Sphinx docs (where `conf.py` is found).

  ```rst
  .. literalinclude:: /../src/sys/error/err.c
     :start-at: PetscErrorCode PetscError(
     :end-at: PetscFunctionReturn(PETSC_SUCCESS)
     :append: }
  ```

  For robustness to changes in the source files, Use `:start-at:` and related options when possible, noting that you can also use (positive) values of `:lines:` relative to this. For languages other than C, use the `:language:` option to appropriately highlight.

- Any invocable command line statements longer than a few words should be in
  `.. code-block::` sections. Any such statements not in code-block statements must be
  enclosed by double backticks "\`\`". For example `make all` is acceptable but

  ```console
  $ make PETSC_DIR=/my/path/to/petsc PETSC_ARCH=my-petsc-arch all
  ```

  should be in a block.

- All code blocks showing invocation of command line must use the "console" block
  directive. E.g.

  ```rst
  .. code-block:: console

     $ cd $PETSC_DIR/src/snes/interface
     $ ./someprog
     output1
     output2
  ```

  The only exception of this is when displaying raw output, i.e. with no preceding
  commands. Then one may use just the "::" directive to improve visibility E.g.

  ```rst
  ::

     output1
     output2
  ```

- Any code blocks that show command line invocations must be preceded by `$`, e.g.

  ```rst
  .. code-block:: console

     $ ./configure --some-args
     $ make libs
     $ make ./ex1
     $ ./ex1 --some-args
  ```

- Environment variables such as `$PETSC_DIR` or `$PATH` must be preceded by
  `$` and be enclosed in double backticks, e.g.

  ```rst
  Set ``$PETSC_DIR`` and ``$PETSC_ARCH``
  ```

- For internal links, use explicit labels, e.g

  ```rst
  .. _sec_short_name:

  Section name
  ============
  ```

  and elsewhere (in any document),

  ```rst
  See :ref:`link text <sec_short_name>`
  ```

- For internal links in the manual with targets outside the manual, always provide alt text
  so that the text will be  properly formatted in the {ref}`standalone PDF manual <sec_local_docs_latex>`, e.g.

  > ```rst
  > PETSc has :doc:`mailing lists </community/mailing>`.
  > ```

- We use the [sphinxcontrib-bibtex extension](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/)
  to include citations from BibTeX files.
  You must include `.. bibliography::` blocks at the bottom of a page including citations ([example](https://gitlab.com/petsc/petsc/-/raw/main/doc/manual/ksp.rst)).
  To cite the same reference in more than one page, use [this workaround](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#key-prefixing) on one of them ([example](https://gitlab.com/petsc/petsc/-/raw/main/doc/developers/articles.rst)) [^bibtex-footnote].

- See special instructions on {any}`docs_images`.

- Prefer formatting styles that are easy to modify and maintain.  In particular, use of [list-table](https://docutils.sourceforge.io/docs/ref/rst/directives.html#list-table) is recommended.

- When using external links with inline URLs, prefer to use [anonymous hyperlink references](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#anonymous-hyperlinks) with two trailing underscores, e.g.

  ```rst
  `link text <https://external.org>`__
  ```

- To pluralize something with inline markup, e.g. `DM`s, escape the trailing character to avoid `WARNING: Inline literal start-string without end-string`.

  ```rst
  ``DM``\s
  ```

- Use restraint in adding new Sphinx extensions, in particular those which aren't
  widely-used and well-supported, or those with hidden system dependencies.

(docs-images)=

## Images

PETSc's documentation is tightly coupled to the source code and tests, and
is tracked in the primary PETSc Git repository. However, image files are
too large to directly track this way (especially because they persist in the integration branches' histories).

Therefore, we store image files in a separate git repository and clone it when
needed. Any new images required must be added to the currently-used branch of this repository.

### Image Guidelines

- Whenever possible, use SVG files.  SVG is a web-friendly vector format and will be automatically converted to PDF using `rsvg-convert` [^svg-footnote]
- Avoid large files and large numbers of images.
- Do not add movies or other non-image files.

### Adding new images

- Note the URL and currently-used branch (after `-b`) for the upstream images repository, as used by the documentation build:

```{literalinclude} /../doc/makefile
:language: makefile
:lines: 2
:start-at: 'images:'
```

- Decide where in `doc/images` a new image should go. Use the structure of the `doc/` tree itself as a guide.
- Create a Merge Request to the currently-used branch of the upstream images repository, adding this image [^maintainer-fast-image-footnote].
- Once this Merge Request is merged, you may make a {doc}`Merge Request to the primary PETSc repository </developers/integration>`, relying on the new image(s).

It may be helpful to place working copies of new image(s) in your local `doc/images`
while iterating on documentation; just don't forget to update the upstream images repository.

### Removing, renaming, moving or updating images

Do not directly move, rename, or update images in the images repository.
Simply add a logically-numbered new version of the image.

If an image is not used in *any* {any}`integration branch <sec_integration_branches>` (`main` or `release`),
add it to the the top-level list of files to delete, in the images repository.

(docs-images-cleanup)=

### Cleaning up the images repository (maintainers only)

If the size of the image repository grows too large,

- Create a new branch `main-X`, where `X` increments the current value
- Create a new commit deleting all files in the to-delete list and clearing the list
- Reset the new `main-X` to a single commit with this new, cleaned-up state
- Set `main-X` as the "default" branch on GitLab (or wherever it is hosted).
- Update both `release` and `main` in the primary PETSc repository to clone this new branch

(classic-docs-build)=

## Building Classic Documentation

Some of the documentation is built by a "classic" process as described below.

The documentation tools listed below can be
automatically downloaded and installed by `configure`.

- [Sowing](http://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz): a text processing tool developed by Bill Gropp.  This produces the PETSc manual pages; see the [Sowing documentation](http://wgropp.cs.illinois.edu/projects/software/sowing/doctext/doctext.htm) and {ref}`manual_page_format`.
- [C2html](http://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz): A text processing package. This generates the HTML versions of all the source code.

Note that Sowing and C2html are build tools that do not use the compilers specified to PETSc's `configure`, as they
need to work in cross-compilation environments. Thus, they default to using `gcc`, `g++`, and `flex` from
the user's environment (or `configure` options like `--download-sowing-cxx`). Microsoft Windows users should install `gcc`
etc. from Cygwin as these tools don't build with MS compilers.

One can run this process in-tree with

```console
$ make alldoc12 LOC=${PETSC_DIR}
```

For debugging, a quick preview of manual pages from a single source directory can be obtained, e.g.

```console
$ cd $PETSC_DIR/src/snes/interface
$ make LOC=$PETSC_DIR manualpages_buildcite
$ browse $PETSC_DIR/manualpages/SNES/SNESCreate.html  # or otherwise open in browser
```

```{rubric} Footnotes
```

[^venv-footnote]: This requires Python 3.3 or later, and you maybe need to install a package like `python3-venv`.

[^bibtex-footnote]: The extensions's [development branch](https://github.com/mcmtroffaes/sphinxcontrib-bibtex) [supports our use case better](https://github.com/mcmtroffaes/sphinxcontrib-bibtex/pull/185) (`:footcite:`), which can be investigated if a release is ever made.

[^svg-footnote]: `rsvg-convert` is installable with your package manager, e.g., `librsvg2-bin` on Debian/Ubuntu systems).

[^maintainer-fast-image-footnote]: Maintainers may directly push commits.
