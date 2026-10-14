"""
PRISM GUI entry point: python -m prism (or the `prism` command), or python prism/main.py.
"""
import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file (python prism/main.py): import PRISM as the `prism` package from the folder above,
    # never this folder's modules as loose top-level names (core, gui, ...), which would load a second copy.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from prism.gui.app import PrismApp
    PrismApp().mainloop()


if __name__ == "__main__":
    main()
