//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); Grand Theft Auto VI sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    /// Compute units of the GPU.
    pub const fn compute_units(self) -> u32 {
        match self {
            Console::SeriesX => 52,
            Console::SeriesS => 20,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

/// Which console the title sees, by the adapter's dedicated video memory:
/// 12 GiB or less names the Series S profile — the Series X one sizes a
/// title's memory at 13.5 GiB (10 of them GPU-optimal) and paces 4K
/// targets, which such a card does not hold, and half the frame went to
/// the Series S mode on a 10 GB RTX 3080 (2,699 → 1,898 ms; #1317, H-1.1).
/// No answer names Series X, as before.
pub fn profile_for(dedicated_video: Option<usize>) -> Console {
    match dedicated_video {
        Some(d) if d <= 12 * GIB => Console::SeriesS,
        _ => Console::SeriesX,
    }
}

/// Which console the host runs: `profile_for` of the first adapter's
/// dedicated video memory, the adapter a device is created on when none
/// is named. `VICTORY_LOG=victory_gpu=info` names the adapter's own
/// account; the console can still be chosen by `--console`.
pub fn profile_by_adapter() -> (Console, Option<usize>) {
    let dedicated = dedicated_video_memory();
    (profile_for(dedicated), dedicated)
}

/// The first adapter's dedicated video memory, in bytes: the adapter the
/// device takes by default (`D3D12CreateDevice(None, …)`). None when DXGI
/// does not answer.
fn dedicated_video_memory() -> Option<usize> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory4,
    };
    // SAFETY: plain factory and adapter queries, with no side effects;
    // the factory and adapter are freed on drop.
    let desc = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter = factory.EnumAdapters1(0).ok()?;
        adapter.GetDesc1().ok()?
    };
    Some(desc.DedicatedVideoMemory as usize)
}

/// The mode the console's display lists (`dxgi`): size that matches it.
pub(crate) fn mode_size(console: Console) -> (u32, u32) {
    match console {
        Console::SeriesX => (3840, 2160),
        Console::SeriesS => (2560, 1440),
    }
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_S.store(console == Console::SeriesS, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dedicated video memory of 12 GiB or less names the Series S
    /// profile; more, or no answer, the Series X one.
    #[test]
    fn the_profile_follows_the_adapter_once_it_answers() {
        let gib = 1 << 30;
        assert_eq!(profile_for(None), Console::SeriesX, "no adapter answer");
        assert_eq!(profile_for(Some(0)), Console::SeriesS);
        assert_eq!(profile_for(Some(10 * gib)), Console::SeriesS);
        assert_eq!(
            profile_for(Some(12 * gib)),
            Console::SeriesS,
            "at most 12 GiB"
        );
        assert_eq!(
            profile_for(Some(12 * gib + 1)),
            Console::SeriesX,
            "past 12 GiB"
        );
        assert_eq!(profile_for(Some(16 * gib)), Console::SeriesX);
        assert_eq!(profile_for(Some(24 * gib)), Console::SeriesX);
    }

    /// The listed display mode matches the console: 4K on Series X,
    /// the Series S target of 2560×1440 on it.
    #[test]
    fn the_mode_matches_the_console() {
        assert_eq!(mode_size(Console::SeriesX), (3840, 2160));
        assert_eq!(mode_size(Console::SeriesS), (2560, 1440));
    }
}
