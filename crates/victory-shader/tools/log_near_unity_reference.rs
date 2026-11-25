// Regenerator for `LOG_NEAR_UNITY_ROWS`, the in-band reference table of
// `crates/victory-shader/src/emit/valu_tests.rs` (the near-unity accuracy fixture for `v_log_f32`).
//
// A maintainer tool, deliberately kept outside `src/`, `tests/`, `examples/` and `benches/`: no
// Cargo target refers to it, so nothing about building, linting, formatting or testing the crate
// changes because of its presence, while the checked-in fixture stays machine-checkable. Its only
// runtime requirement is this repository's own pinned Rust toolchain plus the C linker that Cargo
// already needs; there are no third-party crates, no console tools, and no GPU work involved.
//
//   rustc --edition 2024 -O crates/victory-shader/tools/log_near_unity_reference.rs \
//         -o log-near-unity-reference
//   ./log-near-unity-reference                  -> the fixture's table rows, on stdout only
//   ./log-near-unity-reference --check FILE     -> comparison summary against a valu_tests.rs
//
// On Windows, run `rustc` from the same x64 MSVC developer environment (`vcvars64.bat`) that Cargo
// uses. Output is deterministic: no clock, no randomness, no locale or argument-order dependence.
// The exit status is 0 when everything agrees and non-zero when `--check` reports a difference or one
// of the internal precision guards trips, so it can gate any future change to the table.
//
// Independence of the expected values. Every reference bit pattern and acceptance window here comes
// from mathematics alone: `log2(x)` is derived as `ln(x) / ln(2)` with both logarithms summed from
// their series in integer arithmetic. This file reads no constant, coefficient or intermediate of the
// lowering under test - it never looks at `lower.rs`. The input list below is fixture bookkeeping
// (which operands to cover), never an expectation: reference values are computed from scratch.
//
// How the arithmetic works and why it is precise enough for a table that must be right bit for bit.
// `ln(x)` uses ln(x) = 2 * atanh((x - 1) / (x + 1)), summed over odd powers. The binary32 operand is
// an exact dyadic rational, so `(x - 1) / (x + 1)` is exactly a ratio `numerator / denominator` of
// integers below 2^25, and each power of it is obtained by one multiply and one divide. Everything
// lives on a grid: logarithms on a grid of 2^-60, the result `log2(x)` on the finer grid of 2^-64,
// both far below a `u128` overflow (the widest intermediate here is about 2^124).
//
// One grid unit of 2^-64 is about 5 * 10^-20. The quantities this table compares are coarser than
// that: the finest is one reference-binade ULP of a result as small as 6 * 10^-8, about 1.3 * 10^-14,
// so each comparison sits thousands of grid units away from the point where it would flip. What
// bounds the error is truncation, and truncation is countable: every divide floors, costing under one
// unit of the 2^-60 logarithm grid; the atanh tail stops only once the running power has shifted away
// whole, costing under one further unit; and the series runs fewer than 60 terms. Carried through to
// the output grid that is well under a hundred units near unity and a few hundred for the largest
// results at the interval edges, where the fixed point has proportionally fewer spare bits.
//
// The generator records the distance of each decision from its boundary and requires at least 2^12
// output-grid units of separation. It also recomputes each row on finer grids (2^-62 series,
// 2^-66 output). Agreement is a consistency check: both evaluations can share truncation bias.
// The fixed-point error bound, measured decision margins and independent reference verification
// support the checked-in fixture.

use std::process::exit;

/// Grid for the logarithm series and for the derived `log2` value, in the primary pass.
const SERIES_BITS: usize = 60;
const OUTPUT_BITS: usize = 64;
/// Finer grids for a consistency check. Both grids use the same derivation and can share truncation
/// bias. The finer grids provide more resolution around decision boundaries.
const CONFIRM_SERIES_BITS: usize = 62;
const CONFIRM_OUTPUT_BITS: usize = 66;

/// Least allowed distance, in grid units of the output grid, between a decision and the point where
/// it would flip. Comfortably above the roughly hundred units of accumulated truncation, so a
/// borderline row trips an assertion instead of quietly becoming a wrong expectation.
const MIN_MARGIN_UNITS: u128 = 1 << 12;

// The corrected interval |x - 1| <= 2^-6. Counted in bit-pattern steps its half-widths differ across
// unity, because binary32 spacing does: a step below one is 2^-24, above it 2^-23.
const LOWER_BOUNDARY: u32 = 0x3f7c_0000;
const UNITY: u32 = 0x3f80_0000;
const UPPER_BOUNDARY: u32 = 0x3f82_0000;
const TWO: u32 = 0x4000_0000; // binary32 for exactly 2, the anchor for ln(2)

// Which operands the fixture covers - coverage choices only, never an expected value. `--check`
// reports any drift between this list and what is checked in.
/// The interval endpoints plus their two immediate inside neighbours on each side.
const BOUNDARY_STEPS: [u32; 3] = [0, 1, 2];
/// A deterministic stride across each half of the interval: `GRID_DIVISOR` equal value steps per
/// side, so a point sits at nearest(`k * half-width / GRID_DIVISOR`) in pattern terms. Points up to
/// `GRID_POINTS_PER_SIDE` land strictly inside; one more lands exactly on an endpoint already
/// covered above, which is why duplicates are dropped rather than counted twice.
const GRID_DIVISOR: u64 = 30;
const GRID_POINTS_PER_SIDE: u64 = 29;
/// The closest representable operands to unity on each side, in pattern steps.
const NEAR_UNITY_STEPS: u32 = 34;
/// Power-of-two bit-pattern offsets from unity, which land on the round values a shader writer
/// reaches for (1 +- 2^-9, 1 + 2^-7 and so on). Above unity the list runs one step further because
/// that half of the interval is half as many pattern steps wide.
const POWERS_BELOW_UNITY: [u32; 10] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const POWERS_ABOVE_UNITY: [u32; 11] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// The number of in-band operands the fixture holds.
const EXPECTED_INPUT_COUNT: usize = 152;

/// One fixture row exactly as the test states it: the operand, the correctly rounded reference, and
/// the acceptance window as whole-pattern steps either side of that reference (never positive below,
/// never negative above), where for one sign the low 31 bits run monotonically outward from zero. A
/// result is accepted when its step lies in `steps_below..=steps_above`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    input: u32,
    reference: u32,
    steps_below: i64,
    steps_above: i64,
}

fn bit_length(value: u128) -> usize {
    (128 - value.leading_zeros()) as usize
}

/// The exact value of a finite binary32 magnitude on a grid of 2^-`grid_bits`, with no rounding: a
/// binary32 is `significand * 2^(biased - 150)`, so scaling by 2^grid_bits is a shift.
fn scaled_magnitude(pattern: u32, grid_bits: usize) -> u128 {
    let biased = ((pattern >> 23) & 0xff) as i64;
    assert!(biased < 255, "the fixture holds only finite inputs");
    let significand = (pattern & 0x7f_ff_ff) + if biased == 0 { 0 } else { 1 << 23 };
    let shift = grid_bits as i64 + biased.max(1) - 150;
    assert!(
        shift >= 0,
        "the grid 2^-{grid_bits} is not finer than the magnitude of {pattern:08x}"
    );
    u128::from(significand) << shift as u32
}

/// `x - 1` and `x + 1` as an exact integer pair with `x = significand / 2^shift`: over this band and
/// at x = 2 both are far below 2^25, which is what keeps every later multiply inside `u128`.
fn sides_of_unity(pattern: u32) -> (u128, u128) {
    let biased = ((pattern >> 23) & 0xff) as i64;
    assert!(
        biased < 255 && biased >= 1,
        "expected a normal, finite operand"
    );
    let significand = u128::from((pattern & 0x7f_ff_ff) + (1 << 23));
    // x - 1 and x + 1 share the denominator 2^shift, so only the numerators are needed. The
    // difference is taken as a magnitude: below one it is negative, and atanh is odd, so the same
    // sum of magnitudes gives |ln(x)| on either side of one.
    let half = 1u128 << (150 - biased) as u32;
    assert!(significand != half, "expected an operand away from one");
    (significand.abs_diff(half), significand + half)
}

/// |ln(x)| on a grid of 2^-`grid_bits`, from ln(x) = 2 * sum_k z^(2k+1) / (2k + 1) with
/// z = (x - 1) / (x + 1). Because atanh is odd, the same sum over magnitudes gives |ln(x)| on either
/// side of one. Each power is one multiply by the exact ratio followed by a divide, so it floors:
/// under one grid unit per step, and the loop stops once the power has shifted away whole, which
/// leaves an omitted tail under one further unit for arguments this small.
fn ln_magnitude(pattern: u32, grid_bits: usize) -> u128 {
    let (numerator, denominator) = sides_of_unity(pattern);
    // The series runs over odd powers, so one step advances the running value by z^2. Both squares
    // fit comfortably: numerator and denominator are below 2^25 here, so the widest intermediate is
    // about 2^110 inside a u128.
    let square_numerator = numerator * numerator;
    let square_denominator = denominator * denominator;
    assert!(
        bit_length(square_denominator) < 127,
        "the squared ratio escaped u128"
    );
    let mut power = (1u128 << grid_bits) * numerator / denominator;
    let mut sum = 0u128;
    let mut odd_divisor: u128 = 1;
    let mut steps = 0usize;
    while power != 0 {
        sum += power / odd_divisor;
        let previous = power;
        power = power * square_numerator / square_denominator;
        // Successive powers must shrink strictly: that is what both terminates the sum and bounds the
        // omitted tail, and it means |z| < 1 for every operand used here.
        if power != 0 {
            assert!(power < previous, "the atanh series stopped shrinking");
        }
        odd_divisor += 2;
        steps += 1;
        assert!(steps < 1_000, "the atanh series did not terminate");
    }
    sum * 2
}

/// |log2(x)| on a grid of 2^-`output_bits`, from `ln(x)` and a precomputed `ln(2)`. The series grid
/// cancels in the ratio, so one division - which floors, costing under one grid unit - carries the
/// result onto the output grid.
fn log2_magnitude(pattern: u32, series_grid: usize, output_grid: usize, ln_two: u128) -> u128 {
    let ln_of_input = ln_magnitude(pattern, series_grid);
    assert!(
        ln_of_input.leading_zeros() >= output_grid as u32,
        "the logarithm is too large to rescale inside u128"
    );
    (ln_of_input << output_grid) / ln_two
}

/// A starting pattern for the reference value, sign excluded: the binade read off this magnitude's
/// bit length plus an exact floor of its 23 fraction bits. `derive_row` decides the rounding by
/// comparing grid distances, so no tie rule or guard/sticky bookkeeping is relied on here.
fn estimated_pattern(magnitude: u128, output_grid: usize) -> u32 {
    assert!(magnitude != 0, "log2 of a fixture operand cannot be zero");
    let unbiased_exponent = bit_length(magnitude) as i64 - 1 - output_grid as i64;
    assert!(
        (-126..=127).contains(&unbiased_exponent),
        "expected an ordinary binary32 reference value"
    );
    let leading = 1u128 << (unbiased_exponent + output_grid as i64) as u32;
    assert!(magnitude >= leading, "bit length disagrees with the binade");
    // floor(fraction * 2^23): a dimensionless count of fraction steps. Adding rather than OR-ing lets
    // a floor at exactly 2^23 carry into the exponent field, which is what an overflowing significand
    // means; the neighbour comparison in `derive_row` settles the rounding either way.
    let fraction_floor = ((magnitude - leading) << 23) / leading;
    assert!(
        fraction_floor <= (1 << 25),
        "the estimate ran past its binade"
    );
    (((unbiased_exponent + 127) as u32) << 23).wrapping_add(fraction_floor as u32)
}

/// How close the decisions of a run came to their boundaries, so precision is audited rather than
/// assumed: every note records two candidates the run had to tell apart.
struct MarginGuard {
    decisions: usize,
    smallest: u128,
}

impl MarginGuard {
    fn new() -> Self {
        MarginGuard {
            decisions: 0,
            smallest: u128::MAX,
        }
    }

    /// Record one separation. A zero separation means an exact tie, which this fixed point cannot
    /// decide, so it asserts rather than reporting a guess.
    fn note(&mut self, first: u128, second: u128) {
        let gap = if first > second {
            first - second
        } else {
            second - first
        };
        assert!(gap != 0, "two candidates tied exactly on the grid");
        self.decisions += 1;
        self.smallest = self.smallest.min(gap);
    }

    /// Enforce the floor and describe the run for the summary line.
    fn report(&self, output_grid: usize) -> String {
        assert!(
            self.smallest != u128::MAX,
            "no decisions were recorded; the derivation did not run"
        );
        assert!(
            self.smallest >= MIN_MARGIN_UNITS,
            "a decision sat only {} grid units (2^-{} of a unit) from a boundary on the grid of \
             2^-{output_grid}, below the required {MIN_MARGIN_UNITS}: the fixed point is no longer \
             comfortably finer than what it judges",
            self.smallest,
            output_grid as i64 - bit_length(self.smallest) as i64 + 1
        );
        format!(
            "decisions={} closest-boundary-separation={} grid units (floor {MIN_MARGIN_UNITS})",
            self.decisions, self.smallest
        )
    }
}

/// One fixture row's expectation: round the mathematical value to binary32 by exact comparison of
/// grid distances, then walk outward in pattern steps while each candidate stays within one ULP of
/// the *unrounded* result, which is exactly what a checked-in window encodes.
fn derive_row(
    input: u32,
    ln_two: u128,
    series_grid: usize,
    output_grid: usize,
    guard: &mut MarginGuard,
) -> Row {
    assert!(input != UNITY && (LOWER_BOUNDARY..=UPPER_BOUNDARY).contains(&input));
    let magnitude = log2_magnitude(input, series_grid, output_grid, ln_two);

    // Reference-binade ULP convention: spacing 2^(floor(log2|y|) - 23), counted from the unrounded
    // value y rather than from any rounded output. On this grid it is exactly a power of two.
    let magnitude_bits = bit_length(magnitude);
    assert!(
        magnitude_bits <= output_grid,
        "|log2({input:08x})| reached {magnitude_bits} bits on a grid of 2^-{output_grid}; it stays \
         under one over this band"
    );
    let one_ulp = 1u128 << (magnitude_bits - 24);

    // log2 of anything below unity is negative, so the sign bit follows the input side; everything
    // below works on magnitudes and the asserts confirm it never crosses zero.
    let sign_bit = if input < UNITY { 0x8000_0000 } else { 0 };

    // Nearest pattern by exact grid comparison around an estimate that is at most one fraction step
    // off. Distance grows monotonically away from the optimum for a fixed sign, so a candidate that
    // beats both immediate neighbours of its own sign is the global nearest.
    let start = estimated_pattern(magnitude, output_grid) | sign_bit;
    let mut best_pattern = start;
    let mut best_gap = magnitude.abs_diff(scaled_magnitude(start & 0x7f_ff_ff_ff, output_grid));
    for delta in -3i64..=3 {
        let candidate = (start as i64 + delta) as u32;
        assert_eq!(candidate >> 31, sign_bit >> 31, "the scan crossed zero");
        let gap = magnitude.abs_diff(scaled_magnitude(candidate & 0x7f_ff_ff_ff, output_grid));
        if gap < best_gap {
            best_pattern = candidate;
            best_gap = gap;
        }
    }
    for delta in [-1i64, 1] {
        let neighbour = (best_pattern as i64 + delta) as u32;
        assert_eq!(neighbour >> 31, sign_bit >> 31);
        let gap = magnitude.abs_diff(scaled_magnitude(neighbour & 0x7f_ff_ff_ff, output_grid));
        guard.note(gap, best_gap);
    }

    // The acceptance window: whole-pattern steps either side of the reference, kept while the
    // candidate's value lies within one ULP of the unrounded result. Both tests are inclusive, and
    // each accept-or-reject is recorded so a borderline row would show up in `guard`.
    let mut steps_below = 0i64;
    for step in 1..=8 {
        let candidate = (best_pattern as i64 - step) as u32;
        assert_eq!(candidate >> 31, sign_bit >> 31);
        let gap = magnitude.abs_diff(scaled_magnitude(candidate & 0x7f_ff_ff_ff, output_grid));
        guard.note(gap, one_ulp);
        if gap > one_ulp {
            break;
        }
        steps_below = -step;
    }
    let mut steps_above = 0i64;
    for step in 1..=8 {
        let candidate = (best_pattern as i64 + step) as u32;
        assert_eq!(candidate >> 31, sign_bit >> 31);
        let gap = magnitude.abs_diff(scaled_magnitude(candidate & 0x7f_ff_ff_ff, output_grid));
        guard.note(gap, one_ulp);
        if gap > one_ulp {
            break;
        }
        steps_above = step;
    }

    Row {
        input,
        reference: best_pattern,
        steps_below,
        steps_above,
    }
}

/// Every row at one pair of grids, with `ln(2)` derived once and reused.
fn derive_rows(
    inputs: &[u32],
    series_grid: usize,
    output_grid: usize,
    guard: &mut MarginGuard,
) -> Vec<Row> {
    // A standing check on the grids themselves: 1.0 has to read as exactly one grid unit and 2.0 as
    // two. Getting that wrong skews every logarithm downstream without any other visible symptom.
    assert_eq!(scaled_magnitude(UNITY, series_grid), 1u128 << series_grid);
    assert_eq!(scaled_magnitude(TWO, series_grid), 2u128 << series_grid);
    assert_eq!(scaled_magnitude(UNITY, output_grid), 1u128 << output_grid);
    let ln_two = ln_magnitude(TWO, series_grid);
    inputs
        .iter()
        .map(|input| derive_row(*input, ln_two, series_grid, output_grid, guard))
        .collect()
}

/// Nearest pattern-step count of grid point `k`, from equal value spacing across each half of the
/// interval. A halfway landing cannot happen at this divisor and is asserted rather than absorbed by
/// picking a tie rule.
fn grid_step(half_width_steps: u64, k: u64) -> u32 {
    let scaled = k * half_width_steps;
    let whole = scaled / GRID_DIVISOR;
    let remainder = scaled % GRID_DIVISOR;
    assert!(
        remainder * 2 != GRID_DIVISOR,
        "a grid point landed exactly halfway"
    );
    if remainder * 2 > GRID_DIVISOR {
        (whole + 1) as u32
    } else {
        whole as u32
    }
}

/// The operands the fixture covers: interval endpoints and their inside neighbours, a deterministic
/// stride across each half of the interval, the closest representable values to unity on both sides,
/// and power-of-two bit-pattern offsets from unity. Points two families share (a grid point landing
/// on an endpoint, say) are counted once.
fn fixture_inputs() -> Vec<u32> {
    let mut inputs: Vec<u32> = Vec::new();
    for step in BOUNDARY_STEPS {
        inputs.push(LOWER_BOUNDARY + step);
        inputs.push(UPPER_BOUNDARY - step);
    }
    let half_below = u64::from(UNITY - LOWER_BOUNDARY);
    let half_above = u64::from(UPPER_BOUNDARY - UNITY);
    for k in 1..=GRID_POINTS_PER_SIDE {
        inputs.push(UNITY - grid_step(half_below, k));
        inputs.push(UNITY + grid_step(half_above, k));
    }
    for step in 1..=NEAR_UNITY_STEPS {
        inputs.push(UNITY - step);
        inputs.push(UNITY + step);
    }
    for exponent in POWERS_BELOW_UNITY {
        inputs.push(UNITY - (1 << exponent));
    }
    for exponent in POWERS_ABOVE_UNITY {
        inputs.push(UNITY + (1 << exponent));
    }

    inputs.retain(|input| *input != UNITY && (LOWER_BOUNDARY..=UPPER_BOUNDARY).contains(input));
    inputs.sort_unstable();
    inputs.dedup();
    assert_eq!(
        inputs.len(),
        EXPECTED_INPUT_COUNT,
        "fixture coverage drifted"
    );
    inputs
}

fn render_row(row: &Row) -> String {
    format!(
        "    ({:#010x}, {:#010x}, {}, {}),",
        row.input, row.reference, row.steps_below, row.steps_above
    )
}

/// Read the table as checked in, so a regeneration can be compared against it token by token.
fn read_checked_in_table(path: &str) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("reading {path}: {error}"))?;
    let marker = "const LOG_NEAR_UNITY_ROWS";
    let start = text
        .find(marker)
        .ok_or_else(|| format!("{path}: no {marker} declaration found"))?;
    let declared = &text[start..];
    let literal_at = declared
        .find("= &[")
        .ok_or_else(|| format!("no array literal after {marker}"))?;
    let rows_text = &declared[literal_at + 3..];

    let mut rows: Vec<Row> = Vec::new();
    let bytes = rows_text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b']' {
            break;
        }
        if bytes[index] != b'(' {
            index += 1;
            continue;
        }
        let Some(close) = rows_text[index..].find(')') else {
            return Err("unterminated table row".to_owned());
        };
        let inside = &rows_text[index + 1..index + close];
        let fields: Vec<String> = inside
            .split(',')
            .map(|field| field.trim().chars().filter(|c| *c != '_').collect())
            .collect();
        if fields.len() != 4 {
            return Err(format!(
                "table row `({inside})` has {} fields, expected four",
                fields.len()
            ));
        }
        let number = |field: &str| -> Result<u64, String> {
            if let Some(hex) = field.strip_prefix("0x").or(field.strip_prefix("0X")) {
                u64::from_str_radix(hex, 16).map_err(|e| format!("{field}: {e}"))
            } else {
                field
                    .parse::<i64>()
                    .map(|value| value as u64)
                    .map_err(|e| format!("{field}: {e}"))
            }
        };
        rows.push(Row {
            input: number(&fields[0])? as u32,
            reference: number(&fields[1])? as u32,
            steps_below: number(&fields[2])? as i64,
            steps_above: number(&fields[3])? as i64,
        });
        index += close + 1;
    }
    Ok(rows)
}

/// Compare two row lists pairwise. Row order is part of the fixture, so an equal set in a different
/// order still counts as drift.
fn compare_rows(label: &str, generated: &[Row], expected: &[Row]) -> bool {
    let mut differences = (generated.len() as i64 - expected.len() as i64).abs() as usize;
    for index in 0..generated.len().min(expected.len()) {
        if generated[index] != expected[index] {
            differences += 1;
            if differences <= 8 {
                println!(
                    "{label}: row {} differs, generated [{}] expected [{}]",
                    index + 1,
                    render_row(&generated[index]).trim(),
                    render_row(&expected[index]).trim()
                );
            }
        }
    }
    if differences == 0 {
        println!(
            "{label}: all {} rows match exactly, in order",
            generated.len()
        );
        true
    } else {
        println!(
            "{label}: {differences} difference(s) across {} generated and {} expected rows",
            generated.len(),
            expected.len()
        );
        false
    }
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let inputs = fixture_inputs();

    if !arguments.first().is_some_and(|first| first == "--check") {
        // Rows only, so the output can be pasted straight into the test. The precision guards run as
        // part of the derivation either way, and the audit line goes to stderr.
        let mut guard = MarginGuard::new();
        for row in derive_rows(&inputs, SERIES_BITS, OUTPUT_BITS, &mut guard) {
            println!("{}", render_row(&row));
        }
        eprintln!(
            "reference generated on a grid of 2^-{OUTPUT_BITS}; {}",
            guard.report(OUTPUT_BITS)
        );
        return;
    }

    let Some(path) = arguments.get(1) else {
        let name = std::env::args()
            .next()
            .unwrap_or_else(|| "log-near-unity-reference".to_owned());
        println!("usage: {name} --check <path to valu_tests.rs>");
        exit(2);
    };

    let mut primary_guard = MarginGuard::new();
    let rows = derive_rows(&inputs, SERIES_BITS, OUTPUT_BITS, &mut primary_guard);
    let mut confirm_guard = MarginGuard::new();
    let confirmed = derive_rows(
        &inputs,
        CONFIRM_SERIES_BITS,
        CONFIRM_OUTPUT_BITS,
        &mut confirm_guard,
    );

    println!(
        "reference generator: {} in-band operands over |x - 1| <= 2^-6",
        inputs.len()
    );
    println!(
        "primary grids (series 2^-{SERIES_BITS}, output 2^-{OUTPUT_BITS}): {}",
        primary_guard.report(OUTPUT_BITS)
    );
    println!(
        "confirming grids (series 2^-{CONFIRM_SERIES_BITS}, output 2^-{CONFIRM_OUTPUT_BITS}): {}",
        confirm_guard.report(CONFIRM_OUTPUT_BITS)
    );

    let grids_agree = compare_rows("grid agreement", &rows, &confirmed);
    match read_checked_in_table(path) {
        Ok(checked_in) => {
            println!("checked-in table: {path}");
            let matches_fixture = compare_rows("fixture comparison", &rows, &checked_in);
            exit(if grids_agree && matches_fixture { 0 } else { 1 });
        }
        Err(error) => {
            println!("cannot read the checked-in table: {error}");
            exit(2);
        }
    }
}
