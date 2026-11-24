extern "archive_directory_gpu" fn directory_input() callconv(.C) ?*const Inputs;
extern "archive_directory_gpu" fn enqueue_page(offset: u64, size: u32) callconv(.C) u32;
extern "archive_directory_gpu" fn draw_pages() callconv(.C) u32;

// Three adjacent, unmodified RDS5 boundary words, then an explicit first-offset control.
const Inputs = extern struct { first: u64, second: u64, end: u64, first_bias: u64 };

fn units(word: u64) u64 {
    return word & 0x3ffffff;
}

noinline fn queueRecord(word: u64, next: u64, bias: u64) bool {
    const start = units(word) << 8;
    const end = units(next) << 8;
    const padding = word >> 52;
    const size: u32 = @truncate(end -% start -% padding);
    return enqueue_page(start +% bias, size) == size;
}

pub fn main() u8 {
    const input = directory_input() orelse return 2;
    if (!queueRecord(input.first, input.second, input.first_bias)) return 3;
    if (!queueRecord(input.second, input.end, 0)) return 3;
    return @truncate(draw_pages());
}
