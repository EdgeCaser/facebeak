def create_multi_view_extractor():
    class DummyMultiViewExtractor:
        def extract(self, img):
            return [img]
    return DummyMultiViewExtractor() 