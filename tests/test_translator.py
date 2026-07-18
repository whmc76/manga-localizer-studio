from manga_localizer.translator import pytorch_device


def test_shared_gpu_name_is_normalized_for_pytorch():
    assert pytorch_device("gpu:0") == "cuda:0"
    assert pytorch_device("gpu:2") == "cuda:2"
    assert pytorch_device("cpu") == "cpu"
    assert pytorch_device("auto") == "auto"
