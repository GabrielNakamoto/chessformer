from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from datasets import load_dataset_builder


ds_builder = load_dataset_builder("gRa1ne/decorrelated-chess-3.8m")
N = int(ds_builder.info.splits['train'].num_examples)
x = Tensor.empty(N, 8, 64, dtype=dtypes.uint8, device="DISK:tensors/x.bin")
yz = Tensor.empty(N, 3, dtype=dtypes.float32, device="DISK:tensors/yz.bin")
yp = Tensor.empty(N, 1858, dtype=dtypes.int8, device="DISK:tensors/yp.bin")
print(x[0].numpy())
