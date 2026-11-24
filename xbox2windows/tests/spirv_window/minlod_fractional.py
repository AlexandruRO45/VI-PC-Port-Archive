"""Run the strict fractional point-filter MinLod cases."""
import sys

from minlod_levels import main


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:], fractional=True))
