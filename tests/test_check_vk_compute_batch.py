"""runtime/test/check_vk_compute_batch.py compares each batch job with a single run of the same program. A driver
notice printed once per device creation (RADV, lavapipe, ANV) is in the single run only, so it must not count as a
difference; anything else in the output still must."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "test"))
import check_vk_compute_batch as check  # noqa: E402

RADV = b"WARNING: radv is not a conformant Vulkan implementation, testing use only.\n"


def test_a_driver_notice_from_device_creation_is_not_a_difference():
    single = RADV + b"device: AMD Radeon RX 9060 XT (RADV GFX1200)\r\nPASS\r\n"
    batch_job = b"device: AMD Radeon RX 9060 XT (RADV GFX1200)\nPASS\n"
    assert check.plain(single) == check.plain(batch_job)


def test_other_output_still_differs():
    assert check.plain(RADV + b"PASS\n") != check.plain(b"FAIL\n")
    assert check.plain(b"WARNING: something else\nPASS\n") != check.plain(b"PASS\n")
