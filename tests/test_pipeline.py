import io
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from PIL import Image
from fashion_demo.pipeline import overlay_mask, read_image, Result


class ImageHandlingTests(unittest.TestCase):
    def test_mask_changes_only_selected_pixels(self):
        image = Image.new("RGB", (12, 8), "white")
        mask = np.zeros((8, 12), dtype=bool)
        mask[2:4, 3:7] = True
        output = np.array(overlay_mask(image, mask, .45))
        self.assertTrue(np.all(output[~mask] == 255))
        self.assertTrue(np.any(output[mask] != 255))
        self.assertEqual(overlay_mask(image, mask, 0).tobytes(), image.tobytes())

    def test_bad_image_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, "无法读取图片"):
            read_image(io.BytesIO(b"not an image"))

    def test_unicode_path_and_gray_image(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "中文图片.png"
            Image.new("L", (13, 21), 127).save(path)
            image = read_image(path)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (13, 21))

    def test_misaligned_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            overlay_mask(Image.new("RGB", (12, 8)), np.zeros((12, 8), bool), .4)


if __name__ == "__main__":
    unittest.main()
