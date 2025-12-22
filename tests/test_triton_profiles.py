import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.optimizer.tuner.profiles import get_triton_kernel_profiles


class TestTritonKernelProfiles(unittest.TestCase):
    def test_profiles_exist_and_differ_for_auto(self) -> None:
        dev = "cuda:0:FakeGPU:cc80"
        for fused in ("triton1pass", "triton2pass"):
            profs = get_triton_kernel_profiles(mode="auto", device_sig=dev, fused=fused, decode_block=1024)
            self.assertGreaterEqual(len(profs), 2)
            self.assertEqual(profs[0].name, "latency")
            self.assertEqual(profs[1].name, "throughput")
            # For cc80, throughput should differ from latency.
            self.assertNotEqual(profs[0], profs[1])

    def test_profiles_small_mode_is_singleton(self) -> None:
        profs = get_triton_kernel_profiles(
            mode="small", device_sig="cuda:0:FakeGPU:cc80", fused="triton1pass", decode_block=256
        )
        self.assertEqual(len(profs), 1)
        self.assertEqual(profs[0].name, "latency")

    def test_profiles_off_mode_is_empty(self) -> None:
        profs = get_triton_kernel_profiles(mode="off", device_sig="cuda:0:FakeGPU:cc80", fused="triton1pass", decode_block=256)
        self.assertEqual(profs, [])


if __name__ == "__main__":
    unittest.main()


