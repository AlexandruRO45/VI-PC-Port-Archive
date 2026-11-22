"""Convert one non-XOR GFX10 4 KiB standard-swizzle BC7 surface to BC block rows."""

# Block index bits from low to high: y0 y1 x0 x1 y2 x2 y3 x3.
_X_BITS = tuple(((x & 3) << 2) | ((x & 4) << 3) | ((x & 8) << 4) for x in range(16))
_Y_BITS = tuple((y & 3) | ((y & 4) << 2) | ((y & 8) << 3) for y in range(16))


def linearize_4k_s(payload, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Expected positive pixel dimensions")
    columns, rows = (width + 3) // 4, (height + 3) // 4
    tile_pitch = (columns + 15) // 16
    if len(payload) != tile_pitch * ((rows + 15) // 16) * 4096:
        raise ValueError("Expected complete, padded 4 KiB tiles for one surface")
    output = bytearray(columns * rows * 16)
    for y in range(rows):
        tile_row = (y >> 4) * tile_pitch
        y_bits = _Y_BITS[y & 15]
        row_dst = y * columns * 16
        for x in range(columns):
            tile = tile_row + (x >> 4)
            src = tile * 4096 + (y_bits | _X_BITS[x & 15]) * 16
            dst = row_dst + x * 16
            output[dst:dst + 16] = payload[src:src + 16]
    return bytes(output)


def linearize_4k_s_tail(tile: bytes | bytearray | memoryview, width: int, height: int, tail_level: int) -> bytes:
    """Read a BC7 mip from a shared 4 KiB tile; level zero is the largest packed mip."""
    if not 0 <= tail_level < 8 or width <= 0 or height <= 0:
        raise ValueError("Invalid BC7 mip tail dimensions or level")
    columns, rows = (width + 3) // 4, (height + 3) // 4
    if columns > max(1, 8 >> tail_level) or rows > max(1, 16 >> tail_level):
        raise ValueError("Mip exceeds its slot in the 4 KiB tail")
    if len(tile) != 4096:
        raise ValueError("Expected one complete 4 KiB mip-tail tile")
    # GFX10 thin, 16-byte elements: positions in the 16-by-16 block tile.
    origins = ((8, 0), (4, 8), (0, 12), (0, 8), (4, 4), (4, 0), (0, 4), (0, 0))
    x, y = origins[tail_level]
    output = bytearray()
    for row in range(y, y + rows):
        y_bits = _Y_BITS[row]
        for column in range(x, x + columns):
            start = (y_bits | _X_BITS[column]) * 16
            output.extend(tile[start:start + 16])
    return bytes(output)
