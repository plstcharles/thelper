import os


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]
if os.getenv("SPELLCHECK"):
    extensions += "sphinxcontrib.spelling",
    spelling_show_suggestions = True
    spelling_lang = "en_US"

source_suffix = ".rst"
master_doc = "index"
project = "thelper"
year = "2018"
author = "Pierre-Luc St-Charles"
copyright = "{0}, {1}".format(year, author)
version = release = "0.0.1"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/plstcharles/thelper/issues/%s", "#"),
    "pr": ("https://github.com/plstcharles/thelper/pull/%s", "PR #"),
}
import sphinx_py3doc_enhanced_theme
html_theme = "sphinx_py3doc_enhanced_theme"
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]
html_theme_options = {
    "githuburl": "https://github.com/plstcharles/thelper/"
}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = "%s-%s" % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

def skip(app, what, name, obj, skip, options):
    if name == "__init__" or name == "__call__" or name == "__getitem__":
        return False
    return skip

def run_apidoc(_):
    argv = ["-o", "./src/", "../src/"]
    try:
        # Sphinx 1.7+
        from sphinx.ext import apidoc
        apidoc.main(argv)
    except ImportError:
        # Sphinx 1.6 (and earlier)
        from sphinx import apidoc
        argv.insert(0, apidoc.__file__)
        apidoc.main(argv)

def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect("builder-inited", run_apidoc)
