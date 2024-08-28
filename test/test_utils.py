from packages import *
from utils import *

class TestGetDevice(TestCase):

    def test_get_cpu(self):
        self.assertEqual(get_cpu(), torch.device('cpu'))

    def test_get_cuda(self):
        if torch.cuda.is_available():
            self.assertEqual(get_cuda(), torch.device('cuda:0'))

    def test_get_mps(self):
        if torch.backends.mps.is_available():
            self.assertEqual(get_mps(), torch.device('mps'))
        else:
            self.assertRaises(ValueError, get_mps)

    def test_get_cuda_cpu(self):
        device = None
        if torch.cuda.is_available():
            device = get_cuda_cpu()
            self.assertEqual(device, torch.device('cuda:0'))
        else:
            device = get_cuda_cpu()
            self.assertEqual(device, torch.device('cpu'))
        logger.info(device)

class TestList(TestCase):

    def test_get_2d_list(self):
        rows, cols = 3, 3
        val = 0
        l = get_2d_list(rows, cols, val)
        self.assertEqual(len(l), rows)
        self.assertEqual(len(l[0]), cols)
        for i in range(rows):
            for j in range(cols):
                self.assertEqual(l[i][j], val)

class Test_imshow():

    @classmethod
    def test_imshow_np(self):
        size = (10, 32, 32)
        imgs = np.random.rand(*size)
        imgs = [imgs[i] for i in range(size[0])]
        imshow(imgs, rows=2, cols=5)

    @classmethod
    def test_imshow_image(self):
        size = (12, 32, 32)
        imgs = np.random.rand(*size)
        imgs = [imgs[i] for i in range(size[0])]
        for i in range(size[0]):
            imgs[i] = Image.fromarray((imgs[i] * 255).astype(np.uint8))
        imshow(imgs, rows=3, cols=4)


#%%
if __name__ == '__main__':
    Test_imshow.test_imshow_np()
    Test_imshow.test_imshow_image()
    # unittest.main()


# %%
