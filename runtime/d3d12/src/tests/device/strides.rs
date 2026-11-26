use super::*;

#[test]
fn set_all_and_collect_within_a_range() {
    let strides = Strides::default();
    strides.set(0x1000, 4);
    strides.set(0x1040, 32);
    strides.set(0x1080, 16);
    let found = strides.collect(0x1000, 0x1080);
    assert_eq!(found, vec![(0x1000, 4), (0x1040, 32)]);
    assert_eq!(strides.collect(0x1100, 0x1200), Vec::<(usize, u32)>::new());
    assert_eq!(strides.all(), vec![(0x1000, 4), (0x1040, 32), (0x1080, 16)]);
}
