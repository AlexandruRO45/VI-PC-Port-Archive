//! The `windows` crate's `Win32` D3D12, DXGI and Direct3D bindings where
//! that crate does not build (Linux: its `windows-future` needs Windows).
//! Users name this crate `windows` (`extern crate victory_windows as
//! windows`), so the same D3D12 code builds on both. The COM methods are
//! `extern "system"`: SysV here, as vkd3d-proton's are.
#![cfg(not(windows))]
#![allow(
    non_snake_case,
    non_upper_case_globals,
    non_camel_case_types,
    dead_code,
    unused_imports,
    clippy::all
)]

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::Windows::Win32;

/// `windows_core`, plus what it only defines on Windows.
pub mod core {
    pub use ::windows_core::*;

    /// A NUL-terminated byte string.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PCSTR(pub *const u8);

    /// A NUL-terminated UTF-16 string.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PCWSTR(pub *const u16);

    /// A mutable NUL-terminated byte string.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PSTR(pub *mut u8);

    /// A mutable NUL-terminated UTF-16 string.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PWSTR(pub *mut u16);

    impl PCSTR {
        pub const fn null() -> Self {
            Self(std::ptr::null())
        }
        pub const fn from_raw(ptr: *const u8) -> Self {
            Self(ptr)
        }
        pub const fn as_ptr(&self) -> *const u8 {
            self.0
        }
        pub fn is_null(&self) -> bool {
            self.0.is_null()
        }
    }

    impl PCWSTR {
        pub const fn null() -> Self {
            Self(std::ptr::null())
        }
        pub const fn as_ptr(&self) -> *const u16 {
            self.0
        }
        pub fn is_null(&self) -> bool {
            self.0.is_null()
        }
    }

    impl TypeKind for PCSTR {
        type TypeKind = CopyType;
    }
    impl TypeKind for PCWSTR {
        type TypeKind = CopyType;
    }
    impl TypeKind for PSTR {
        type TypeKind = CopyType;
    }
    impl TypeKind for PWSTR {
        type TypeKind = CopyType;
    }

    /// Closes a handle (`windows_core::Free` on Windows).
    pub trait Free {
        /// # Safety
        ///
        /// The handle is live and owned.
        unsafe fn free(&mut self);
    }

    /// A NUL-terminated `PCSTR` literal.
    #[macro_export]
    macro_rules! s {
        ($s:literal) => {
            $crate::core::PCSTR::from_raw(::std::concat!($s, '\0').as_ptr())
        };
    }
    pub use crate::s;
}
