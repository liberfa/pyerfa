# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# PyERFA documentation build configuration file.
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

from datetime import datetime
from pkg_resources import get_distribution

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'PyERFA'
author = 'The Astropy Developers'
copyright = '2011–{0}, {1}'.format(datetime.utcnow().year, author)

# The full version, including alpha/beta/rc tags.
try:
    release = get_distribution(project).version
except Exception:
    import configparser
    metadata = configparser.ConfigParser()
    metadata.read('../setup.cfg')
    release = metadata['metadata']['version']

# The short X.Y version.
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.7'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_automodapi.automodapi']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store']

# Enable nitpicky mode - which ensures that all references in the docs
# resolve.
nitpicky = True

# Misc.
highlight_language = 'none'
numpydoc_show_class_members = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# A dictionary of values to pass into the template engine’s context for all pages.
html_context = {
    'to_be_indexed': ['stable', 'latest']
}

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
html_extra_path = ['robots.txt']


# -- Options for LaTeX output ------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

# Setting this URL is requited by sphinx-astropy
github_issues_url = 'https://github.com/liberfa/pyerfa/issues/'
edit_on_github_branch = 'master'


# -- Options for linkcheck output --------------------------------------------
linkcheck_retry = 5
linkcheck_ignore = [
    r'https://github\.com/liberfa/pyerfa/(?:issues|pull)/\d+',
]
linkcheck_timeout = 180
linkcheck_anchors = False
