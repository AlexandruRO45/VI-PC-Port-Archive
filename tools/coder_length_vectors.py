"""Find complete 129..140-symbol raw Huffman length vectors in a local file."""
import argparse
import re
from pathlib import Path


def candidates(data):
    for run in re.finditer(rb"[\x07\x08]{129,}", data):
        for start in range(run.start(), run.end() - 128):
            sevens = data.count(b"\x07", start, start + 128)
            for length in range(129, min(140, run.end() - start) + 1):
                sevens += data[start + length - 1] == 7
                if length + sevens == 256:
                    yield start, length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    count = 0
    for offset, length in candidates(args.image.read_bytes()):
        print(f"{offset:#x} {length}")
        count += 1
    print(f"candidate windows: {count}")


if __name__ == "__main__":
    main()
