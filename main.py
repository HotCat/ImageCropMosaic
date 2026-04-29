import os
import sys
from importlib.util import spec_from_file_location

# Fix: the local sam2/ repo directory shadows the installed sam2 Python package.
# Insert a custom finder that resolves 'sam2' to the real package (sam2/sam2/)
# before PathFinder finds the repo root as a namespace package.
_project_dir = os.path.dirname(os.path.abspath(__file__))
_sam2_pkg_dir = os.path.join(_project_dir, "sam2", "sam2")
_sam2_init = os.path.join(_sam2_pkg_dir, "__init__.py")


class _Sam2Finder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "sam2" and os.path.isfile(_sam2_init):
            return spec_from_file_location(
                "sam2", _sam2_init,
                submodule_search_locations=[_sam2_pkg_dir],
            )
        return None


sys.meta_path.insert(0, _Sam2Finder)

import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(
        description="Image Crop Tool with SAM2 Selection"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to image file to open"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="SAM2 config file path"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="SAM2 checkpoint path"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Image Crop Tool")

    window = MainWindow(initial_image=args.image)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()