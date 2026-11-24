//! DXC fallback for our own HLSL when the system compiler refuses it.
//!
//! Under Wine/Proton, `D3DCompile` is Wine's `d3dcompiler_47` (vkd3d-shader),
//! which has no `RWByteAddressBuffer` methods (`E5005: Method 'InterlockedMax'
//! is not defined on type 'RWByteAddressBuffer'`). The indirect-argument
//! clamp then fails to build and every indirect draw and dispatch is dropped.
//! When `D3DCompile` refuses a shader of ours, the same source is compiled
//! with DXC (`dxcompiler.dll`, found as [`library`] describes) as SM 6.0 and
//! signed with `dxil.dll`. On Windows `D3DCompile` succeeds and nothing changes.
//!
//! Adapted from Marvisak/VIctory PR #178 by Ortomanic (`dxc_library`,
//! `dxc_compile`, `sign_dxc_bytecode`), generalized to any entry point and
//! SM 5.x target.

use windows::core::{HSTRING, PCSTR};

/// Where `dxcompiler.dll` is looked for: `VICTORY_DXC` (the DLL or its
/// folder), beside the `dxil.dll` shader validation uses, then the
/// executable's folder and System32.
fn library() -> windows::core::Result<windows::Win32::Foundation::HMODULE> {
    use windows::Win32::System::LibraryLoader::{
        LOAD_LIBRARY_SEARCH_APPLICATION_DIR, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };
    use windows::core::w;
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("VICTORY_DXC").map(std::path::PathBuf::from) {
        paths.push(if p.is_dir() { p.join("dxcompiler.dll") } else { p });
    }
    if let Some(dir) = victory_shader::emit::validator::Validator::default_path()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        paths.push(dir.join("dxcompiler.dll"));
    }
    for path in paths.iter().filter(|p| p.is_file()) {
        // SAFETY: a NUL-terminated path; its folder (dxil.dll) is searched first.
        if let Ok(m) = unsafe {
            LoadLibraryExW(
                &HSTRING::from(path.as_path()),
                None,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        } {
            return Ok(m);
        }
    }
    // SAFETY: a bare name, searched only in the executable's folder and System32.
    unsafe {
        LoadLibraryExW(
            w!("dxcompiler.dll"),
            None,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    }
}

/// `cs_5_1` and the like as DXC takes them (SM 6.0).
fn profile(target: &str) -> String {
    match target.split_once('_') {
        Some((stage, _)) => format!("{stage}_6_0"),
        None => target.to_owned(),
    }
}

/// Compiles function `entry` of `source` for `target` (an SM 5.x profile,
/// raised to 6.0) with DXC, signed, root signature inside if the source
/// declares one.
pub(crate) fn compile(source: &str, entry: &str, target: &str) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Dxc::*;
    use windows::Win32::System::LibraryLoader::GetProcAddress;
    use windows::core::{Interface, s};
    type Create = unsafe extern "system" fn(
        *const windows::core::GUID,
        *const windows::core::GUID,
        *mut *mut std::ffi::c_void,
    ) -> windows::core::HRESULT;
    let entry_w = HSTRING::from(entry);
    let profile_w = HSTRING::from(profile(target));
    // SAFETY: loading the compiler and calling its documented entry point with
    // out pointers of ours; the buffer and arguments outlive the calls.
    unsafe {
        let library = library()?;
        let create: Create = std::mem::transmute(
            GetProcAddress(library, s!("DxcCreateInstance"))
                .ok_or_else(windows::core::Error::from_thread)?,
        );
        let mut raw = std::ptr::null_mut();
        create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw).ok()?;
        let compiler = IDxcCompiler3::from_raw(raw);
        let buffer = DxcBuffer {
            Ptr: source.as_ptr().cast(),
            Size: source.len(),
            Encoding: DXC_CP_UTF8.0,
        };
        let arguments = [
            windows::core::w!("-T"),
            windows::core::PCWSTR(profile_w.as_ptr()),
            windows::core::w!("-E"),
            windows::core::PCWSTR(entry_w.as_ptr()),
            windows::core::w!("-O3"),
        ];
        let result: IDxcResult = compiler.Compile(&buffer, Some(&arguments), None)?;
        if let Err(e) = result.GetStatus()?.ok() {
            if let Ok(errors) = result.GetErrorBuffer() {
                let text = std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                );
                tracing::warn!("DXC fallback: {}", String::from_utf8_lossy(text).trim());
            }
            return Err(e);
        }
        let blob = result.GetResult()?;
        let mut bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec();
        sign(library, &mut bytes)?;
        Ok(bytes)
    }
}

/// Like [`compile`], for a `PCSTR` entry and target as `D3DCompile` takes them.
pub(crate) fn compile_pcstr(source: &str, entry: PCSTR, target: PCSTR) -> windows::core::Result<Vec<u8>> {
    // SAFETY: NUL-terminated literals from `s!`.
    let (entry, target) = unsafe { (entry.to_string(), target.to_string()) };
    let entry = entry.map_err(|_| windows::core::Error::empty())?;
    let target = target.map_err(|_| windows::core::Error::empty())?;
    compile(source, &entry, &target)
}

/// Older DXC DLLs can compile successfully without finding the signing DLL.
/// The resulting container has a zero digest and native D3D12 rejects it.
fn sign(compiler: windows::Win32::Foundation::HMODULE, bytes: &mut [u8]) -> windows::core::Result<()> {
    use std::os::windows::ffi::OsStringExt;
    use victory_shader::emit::validator::Validator;
    use windows::Win32::Foundation::E_FAIL;
    use windows::Win32::System::LibraryLoader::GetModuleFileNameW;

    if bytes.get(4..20).is_some_and(|digest| digest.iter().any(|&b| b != 0)) {
        return Ok(());
    }
    let explicit = std::env::var_os("VICTORY_DXIL").map(std::path::PathBuf::from);
    let beside_compiler = || {
        let mut path = vec![0; 32_768];
        // SAFETY: the compiler module is loaded and the buffer is ours.
        let len = unsafe { GetModuleFileNameW(Some(compiler), &mut path) } as usize;
        if len == 0 || len >= path.len() {
            return None;
        }
        let compiler = std::path::PathBuf::from(std::ffi::OsString::from_wide(&path[..len]));
        let validator = compiler.with_file_name("dxil.dll");
        validator.is_file().then_some(validator)
    };
    let path = explicit.or_else(beside_compiler).or_else(Validator::default_path);
    Validator::load(path.as_deref())
        .and_then(|mut validator| validator.validate(bytes))
        .map_err(|error| windows::core::Error::new(E_FAIL, format!("cannot sign DXC shader: {error}")))
}
