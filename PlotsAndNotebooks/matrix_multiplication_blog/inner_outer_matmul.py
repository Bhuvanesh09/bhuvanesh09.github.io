import triton
import torch
import pickle
from datetime import datetime
import triton.language as tl

DEVICE = torch.device("cuda:0")
BLOCK_POWER = 7
BLOCK_SIZE = 2**BLOCK_POWER
print("The Current device is:", DEVICE)


@triton.jit
def inner_kernel(A, B, C, M, N, O, BLOCK_SIZE: tl.constexpr, MIN_BATCH: tl.constexpr):
    _ = tl.program_id(0)  # Not really parallel since we will use only one kernel
    # Doing it serially since we only need to check the cache efficiency.
    # We are not interested in the parallelizing the kernel itself at this moment.

    offsets_1d = tl.arange(0, BLOCK_SIZE)[None, :]  # Makes it a 1xBLOCK_SIZE vector
    batch_offsets = tl.arange(0, MIN_BATCH)[
        :, None
    ]  # Makes it a MIN_BATCH x 1 vector since we can't have a batchsize of 1
    offsets_2d = offsets_1d + batch_offsets  # Makes it a MIN_BATCH x BLOCK_SIZE vector
    # Of the MIN_BATCH x BLOCK_SIZE vector, only the first row should be used and the first M/N/O cols should be used.
    # We must create appropriately sized masks to ensure that only the first M/N/O cols are used.

    # We are not doing operations in parallel at the moment.
    for i in range(M):
        for j in range(O):
            # Load row i of A and column j of B into shared memory
            # A is M x N
            # a should be of the size 1 x N but actually would be MIN_BATCH x BLOCK_SIZE with appropriate masking
            mask_a = (offsets_1d < N) * (
                batch_offsets < 1
            )  # mask_a is a MIN_BATCH x BLOCK_SIZE matrix by rules of broadcasting
            a = tl.load(A + i * N + offsets_2d, mask=mask_a)
            # B is N x O
            # b should be of the size N x 1 but actually would be BLOCK_SIZE x MIN_BATCH with appropriate masking
            mask_b = (tl.trans(offsets_1d) < N) * (
                tl.trans(batch_offsets) < 1
            )  # mask_b is a BLOCK_SIZE x MIN_BATCH matrix by rules of broadcasting
            b = tl.load(B + j + (tl.trans(offsets_2d) * O), mask=mask_b)

            # Both of them should be of the size N
            dot_product = tl.dot(a, b)
            mask_for_c = (batch_offsets * tl.trans(batch_offsets)) < 1

            # C is M x O
            tl.store(
                C + i * O + j + (batch_offsets * tl.trans(batch_offsets)),
                dot_product,
                mask=mask_for_c,
            )


@triton.jit
def inner_kernel_triple_loop(A, B, C, M, N, O, BLOCK_SIZE: tl.constexpr):
    _ = tl.program_id(0)  # Not really parallel since we will use only one kernel

    # We are not doing operations in parallel at the moment.
    offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(M):
        for j in range(O):
            a = tl.load(A + i * N + offsets, mask=offsets < N)
            b = tl.load(B + offsets * O + j, mask=offsets < N)
            c = tl.sum(a * b)

            tl.store(C + i * O + j, c)


@triton.jit
def outer_kernel(A, B, C, M, N, O, BLOCK_SIZE: tl.constexpr, MIN_BATCH: tl.constexpr):
    _ = tl.program_id(0)
    batch_offsets = tl.arange(0, MIN_BATCH)
    offsets_a = tl.arange(0, BLOCK_SIZE)[:, None]  # col vector
    mask_a = offsets_a < M
    offsets_b = tl.arange(0, BLOCK_SIZE)[None, :]  # row vector
    mask_b = offsets_b < O
    c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    mask_c = (offsets_a < M) * (offsets_b < O)
    for k in range(0, N):
        # load kth column of A:
        a = tl.load(
            (A + offsets_a * N + k) + batch_offsets[None, :],
            mask=mask_a * (batch_offsets[None, :] < 1),
        )
        # load kth row of B:
        b = tl.load(
            (B + k * O + offsets_b) + batch_offsets[:, None],
            mask=mask_b * (batch_offsets[:, None] < 1),
        )
        c += tl.dot(a, b)

    tl.store(C + offsets_a * O + offsets_b, c, mask=mask_c)


def inner_product_wrapper(x: torch.Tensor, y: torch.Tensor):
    # print(".", end="")
    output = torch.zeros((x.shape[0], y.shape[1]), device=x.device)
    assert x.shape[1] == y.shape[0]
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    M, N, O = x.shape[0], x.shape[1], y.shape[1]

    grid = lambda meta: (1,)

    inner_kernel_triple_loop[grid](x, y, output, M, N, O, BLOCK_SIZE=BLOCK_SIZE)
    return output


def outer_product_wrapper(x: torch.Tensor, y: torch.Tensor):
    # print("*", end="")
    output = torch.zeros((x.shape[0], y.shape[1]), device=x.device)
    assert x.shape[1] == y.shape[0]
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    M, N, O = x.shape[0], x.shape[1], y.shape[1]

    grid = lambda meta: (1,)

    outer_kernel[grid](x, y, output, M, N, O, BLOCK_SIZE=BLOCK_SIZE, MIN_BATCH=16)
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(4, BLOCK_POWER + 1, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["inner_product", "outer_product"],  # Possible values for `line_arg`.
        line_names=["inner_product", "outer_product"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="ms",  # Label name for the y-axis.
        plot_name="Matrix Multiplication Performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand((size, size), device=DEVICE, dtype=torch.float32)
    y = torch.rand((size, size), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "inner_product":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: inner_product_wrapper(x, y), quantiles=quantiles, rep=2000
        )
    if provider == "outer_product":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: outer_product_wrapper(x, y), quantiles=quantiles, rep=2000
        )
    return ms, min_ms, max_ms


if __name__ == "__main__":

    output = benchmark.run(print_data=True, return_df=True)
    results = {"block_size": BLOCK_SIZE, "output": output}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./Data/log_file_{timestamp}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(output)
