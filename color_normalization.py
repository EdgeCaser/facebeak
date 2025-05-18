def create_normalizer(method='adaptive', use_gpu=False):
    class DummyNormalizer:
        def normalize(self, img):
            return img
    return DummyNormalizer() 