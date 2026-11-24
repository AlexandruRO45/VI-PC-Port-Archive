use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn full_scan_and_streaming_probe_agree_on_header_evidence() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("xcl-rpf-{}-{nonce}.rpf", std::process::id()));
    let mut data = vec![0; 0x860];
    data[..4].copy_from_slice(b"9FPR");
    data[0x800..0x804].copy_from_slice(b"9FPR");
    std::fs::write(&archive, data).unwrap();

    let run = |mode| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .arg(mode)
            .output()
    };
    let full = run("--full");
    let probe = run("--probe");
    std::fs::remove_file(&archive).unwrap();

    let summary = |output: std::process::Output| {
        assert!(output.status.success(), "{:?}", output);
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .skip(1) // The banner describes the selected mode.
            .collect::<Vec<_>>()
            .join("\n")
    };
    let full = summary(full.unwrap());
    let probe = summary(probe.unwrap());
    assert_eq!(full, probe);
    assert!(full.contains("RPF9 archives (headers every 2 KB): 2"));
    assert!(full.contains("TOC encryption: unknown"));
}

#[test]
fn explicit_probe_rejects_conflicting_options() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!(
        "xcl-rpf-options-{}-{nonce}.rpf",
        std::process::id()
    ));
    std::fs::write(&archive, b"9FPR").unwrap();

    let options: [&[&str]; 4] = [
        &["--full"],
        &["--find", "record"],
        &["--extract", "record"],
        &["--out", "unused"],
    ];
    let outputs = options.map(|args| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .arg("--probe")
            .args(args)
            .output()
    });
    std::fs::remove_file(&archive).unwrap();

    for (args, output) in options.into_iter().zip(outputs) {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(String::from_utf8(output.stderr).unwrap().contains(args[0]));
    }
}

#[test]
fn signature_sampling_is_bounded_without_skipping_requested_records() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = std::env::temp_dir().join(format!("xcl-rpf-sample-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let archive = root.join("sample.rpf");
    let out = root.join("out");
    let mut data = vec![0; 0x40];
    data[..4].copy_from_slice(b"9FPR");
    for index in 0..128 {
        let name = format!("CS_Record_{index:03}");
        let payload = if index == 127 { b"LATE" } else { b"FSSM" };
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        data.push((name.len() + 1) as u8);
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&[0, 50]);
        for value in [packed.len() as u32, 0xddce7, packed.len() as u32, payload.len() as u32] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        data.extend_from_slice(&packed);
    }
    std::fs::write(&archive, data).unwrap();
    let command = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xcl"));
        command.arg("rpf").arg(&archive).arg("--full");
        command
    };
    let sample = command().args(["--top", "0"]).output();
    let found = command().args(["--top", "1", "--find", "CS_Record_127"]).output();
    let extracted = command().args(["--top", "0", "--extract", "CS_Record_127", "--out"]).arg(&out).output();
    let payload = std::fs::read(out.join("CS_Record_127"));
    std::fs::remove_dir_all(&root).unwrap();

    let sample = sample.unwrap();
    assert!(sample.status.success(), "{sample:?}");
    let sample = String::from_utf8(sample.stdout).unwrap();
    assert!(sample.contains("FSSM 64"), "{sample}");
    assert!(!sample.contains("LATE"), "{sample}");
    let found = found.unwrap();
    assert!(found.status.success(), "{found:?}");
    let found = String::from_utf8(found.stdout).unwrap();
    assert!(found.contains("CS_Record_127"), "{found}");
    assert!(found.contains("LATE 1"), "{found}");
    assert!(found.contains("FSSM 64"), "{found}");
    let extracted = extracted.unwrap();
    assert!(extracted.status.success(), "{extracted:?}");
    assert_eq!(payload.unwrap(), b"LATE");
}
