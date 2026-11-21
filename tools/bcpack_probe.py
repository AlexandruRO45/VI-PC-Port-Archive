"""Test the published fixed-symbol rANS model against an eight-lane BCPack prefix."""
import argparse
import json
from pathlib import Path


LOWER = 1 << 24
SCALE = 1 << 16
EXCEPTIONS = {0xdc: 4, 0xdd: 8, 0xde: 32}


def probe(prefix, blocks, *, prefetch_mask=0, prefetch=b"", next_mask=0, next_bytes=b""):
    if len(prefix) != 48 or not 1 <= blocks <= 1 << 20:
        raise ValueError("Expected a 48-byte prefix and a positive bounded block count")
    if not 0 <= prefetch_mask <= 255 or len(prefetch) != prefetch_mask.bit_count():
        raise ValueError("Expected one first-round byte per participating lane")
    if not 0 <= next_mask <= 255 or len(next_bytes) != next_mask.bit_count():
        raise ValueError("Expected one second-round byte per participating lane")
    matches = []
    for lane in range(8):
        mode, budget, *state_bytes = prefix[lane::8]
        if mode not in EXCEPTIONS:
            continue
        freq = SCALE - EXCEPTIONS[mode]
        state = LOWER
        refills = 0
        first_refill = second_refill = None
        for _ in range(blocks):
            if state >= (LOWER // SCALE) * 256 * freq:
                # Encoding runs backwards. The last emitted byte is consumed first.
                second_refill, first_refill = first_refill, state & 255
                state >>= 8
                refills += 1
            quotient, remainder = divmod(state, freq)
            state = quotient * SCALE + remainder + 1
        if budget != refills + 3 or state != int.from_bytes(state_bytes, "big"):
            continue
        status = "header_match"
        if refills == 0:
            for _ in range(blocks):
                quotient, slot = divmod(state, SCALE)
                if not 1 <= slot <= freq:
                    raise ValueError("Predicted constant lane contains an escape")
                state = freq * quotient + slot - 1
                if state < LOWER:
                    raise ValueError("Decoder requires an unbudgeted input byte")
            if state != LOWER:
                raise ValueError("Incorrect rANS terminal state")
            status = "verified_constant_lane"
        match = {"lane": lane, "mode": mode, "refill_bytes": refills, "status": status}
        if refills and prefetch_mask & (1 << lane):
            index = (prefetch_mask & ((1 << lane) - 1)).bit_count()
            match["first_refill_offset"] = 48 + index
            match["first_refill_matches"] = prefetch[index] == first_refill
        if refills >= 2 and (prefetch_mask & next_mask) & (1 << lane):
            index = (next_mask & ((1 << lane) - 1)).bit_count()
            match["second_refill_offset"] = 48 + len(prefetch) + index
            match["second_refill_matches"] = next_bytes[index] == second_refill
        matches.append(match)
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--blocks", required=True, type=int)
    parser.add_argument("--prefetch-mask", type=lambda s: int(s, 0), default=0, choices=range(256),
                        metavar="MASK", help="Bit i means lane i reads a byte in the first round after the header")
    parser.add_argument("--next-mask", type=lambda s: int(s, 0), default=0, choices=range(256),
                        metavar="MASK", help="Assumed lane mask for the immediately following refill round")
    args = parser.parse_args()
    with args.packet.open("rb") as stream:
        prefix = stream.read(48)
        prefetch = stream.read(args.prefetch_mask.bit_count())
        next_bytes = stream.read(args.next_mask.bit_count())
        result = probe(prefix, args.blocks, prefetch_mask=args.prefetch_mask, prefetch=prefetch,
                       next_mask=args.next_mask, next_bytes=next_bytes)
        print(json.dumps(result, indent=2))
