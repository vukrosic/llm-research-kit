import unittest

from training.device import DEVICE_CHOICES, resolve_device


class DeviceSelectionTest(unittest.TestCase):
    def test_cpu_can_be_forced(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_auto_resolves_to_supported_device(self):
        device = resolve_device("auto")
        self.assertIn(device.type, {"cuda", "mps", "cpu"})

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_choices_match_cli_contract(self):
        self.assertEqual(DEVICE_CHOICES, ("auto", "cuda", "mps", "cpu"))


if __name__ == "__main__":
    unittest.main()
