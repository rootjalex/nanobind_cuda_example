import nanobind_cuda_example
import torch

def test_add():
    assert nanobind_cuda_example.add(1, 2) == 3

def test_cuda_add_helper(n : int):
    x = torch.randn(n, dtype=torch.float32).to(device="cuda")
    y = torch.randn(n, dtype=torch.float32).to(device="cuda")

    iters = 10

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True, blocking=True)

    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        r = x + y
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print(f"Torch: {start.elapsed_time(end)}ms")
    torch_gpu_ms = start.elapsed_time(end)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True, blocking=True)

    start.record()
    for _ in range(iters):
        mr = nanobind_cuda_example.gpu_add_f32(x, y)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(f"nanobind_cuda_example: {start.elapsed_time(end)}ms")
    nanobind_cuda_example_gpu_ms = start.elapsed_time(end)

    assert(torch.equal(r, mr))

if __name__ == "__main__":
    test_add()
    test_cuda_add_helper(10)
    test_cuda_add_helper(1000)
    test_cuda_add_helper(10000)
