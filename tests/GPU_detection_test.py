# file used to test if pytorch gpu is properly installed
import torch
import cuml
import cudf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Device : {device}")

devNumber = torch.cuda.current_device()

print(f"Curent device : {devNumber}")

devName = torch.cuda.get_device_name()

print(f"Name device : {devName}")

print(cudf.Series([1, 2, 3]))

