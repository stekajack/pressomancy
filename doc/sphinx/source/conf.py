# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../pressomancy'))

project = 'pressomancy'
copyright = '2024, Deniz Mostarac'
author = 'Deniz Mostarac'
release = '0.1.0'
packages = [
    'pressomancy',
]
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
sans_serif_fonts = "'Lucida Grande', 'Arial', sans-serif"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme_options = {
    'sidebar_width': '260px',  # this value must be specified in custom.css too
    'page_width': '1120px',
    'body_max_width': 'auto',
    'font_family': sans_serif_fonts,
    'font_size': '17px',
    'caption_font_family': sans_serif_fonts,
    'caption_font_size': '17px',
    'head_font_family': sans_serif_fonts,
}
html_static_path = ['_static']
# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': ['localtoc.html', 'searchbox.html']
}
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright (C) 2022 The ESPResSo project" is shown
# in the HTML footer. Default is True.
html_show_copyright = True

# -- reStructuredText options ---------------------------------------------

autodoc_default_options = {
    'members': None,
    'show-inheritance': None,
    'undoc-members': None,
}
# For sidebar toc depth
rst_prolog = """
:tocdepth: 3
"""
# Style ESPResSo
rst_epilog = """
.. |es| replace:: *ESPResSo*
"""
# Re-enable grouping of arguments in numpydoc
# https://github.com/sphinx-doc/sphinx/issues/2227
napoleon_use_param = False


# -- Other options --------------------------------------------------------

# -- Options for Cython ---------------------------------------------------

# see https://opendreamkit.org/2017/06/09/CythonSphinx/ why this is needed
def isfunction(obj):
    return hasattr(type(obj), "__code__")

import inspect
inspect.isfunction = isfunction
# In your conf.py

import sys
import types

def setup(app):
    import pressomancy.helper_functions as hf
    original_managed = hf.ManagedSimulation

    # Mock ManagedSimulation for autosummary step
    hf.ManagedSimulation = lambda cls: cls

    def restore_managed_simulation(app, env, docnames):
        # Restore the original ManagedSimulation
        hf.ManagedSimulation = original_managed

    # Connect the restore function
    app.connect('env-before-read-docs', restore_managed_simulation)