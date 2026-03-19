"""field_site_sfm – process Insta360 video and satellite imagery into a
georeferenced 3-D reconstruction of a field site."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("field-site-sfm")
except PackageNotFoundError:
    __version__ = "0.0.0"
