+++
title = "A small realisation about matrix multiplications"
date = "2025-01-06"
category = "treatise"
math = true
plotly = true
type = "posts"
tags = [
  "CS", "CUDA", "GPU", 
]
+++

Recently, I realized something interesting about matrix multiplication. There's a less intuitive but more efficient way to think about it. Here's a technical breakdown of both perspectives and why the order of operations matters for cache efficiency - an insight that clicked while reading about how FlashAttention optimizes the attention mechanism by changing the order of operations.

## Matrix Mult. as Inner Product (The Human Way):
<!-- ![Inner Product Animation](inner_product_matrix_mult.gif) -->
<!-- ![](inner_product_original.gif) -->

{{< image src="inner_product_original.gif" width="60%" height="70%" >}}
When we first learn matrix multiplication, we're taught to think about it as inner products. For matrices $A_{m \times n}$ and $B_{n \times o}$, their product $C = A @ B$ results in a matrix $C_{m \times o}$ where:
- For each element in $C$, we need to do an inner product of **each** row of A with **each** column of B.

In terms of operations:
- We need to compute $O(m \times o)$ different pairwise vector inner products
- Each inner product involves $n$ multiplications and additions
- For now, let's assume these vector operations are instantaneous (we'll discuss the reality later)

This is the natural way most people think about matrix multiplication - take a row from A, a column from B, multiply corresponding elements, and sum them up.

<!-- ![Matrix Mult as inner product](/images/inner_product_matrix_mult.gif) -->

## Matrix Mult. as Outer Product (The Machine Way):

{{< image src="outer_product_original.gif" width="60%" height="70%" >}}
Matrix $A @ B = C$ can also be thought of as the respective **outer** products of the **columns** of A with the **rows** of B.
There are $n$ columns of A and $n$ rows of B. For C, we need the outer product (aka **cross product**) of each column ($m\times 1$) of A with ~~each~~ **respective** row ($1 \times o$)  of B (Not every row of B!) and simply add them ( Adding $n$ number of $m\times o$ shaped matrices). This means, under the hypothetical assumption that vectorwise operations are instantaneous, we only need to do $O(n)$ pairwise vector operations and not $O(n^2)$! 
[Note: The outer product of two vectors is a matrix, and the inner product of two vectors is a scalar.]


The code for the above two animations in manim has been linked in the [references](#references) section below!
## Real life implications of the above:

In reality of course, pairwise operations between vectors aren't instantaneous. Each of the the $n$ outer product in second case takes $O(m \times o)$ FLOPs on its own. And in the first case, each of the $m \times o$ inner product takes $O(n)$ FLOPs on its own. Hence the number of multiplications that are being done is the same i.e. $O(m \times n \times o)$. 

However, the outer product approach is more cache friendly and therefore faster. Why? The key is reducing how often data must be transferred between main (or GPU) memory and the processor. In the outer-product variant, each time we load two vectors (a column of A and a row of B) into fast memory, we perform O(m × o) multiplications, updating a large region of the output matrix. By contrast, the inner-product method only yields O(n) multiplications per pair of loaded vectors—so it has to reload data more frequently to process all the required row-by-column products. Since memory transfers are often the bottleneck, performing more work each time data is loaded translates into better cache utilization and higher overall performance. It happens to be that the overhead of bringing the vectors to the working memory from GPU's VRAM is significant to the actual computation! Reiterating, we get to do $O(m \times o)$ operations in outerproduct method everytime we load two vectors while on the first inner product case, we would be only doing $O(n)$ operations(dot product) for the two vectors loaded before having to load new vectors. This makes the outer product approach more cache friendly and faster.

## What did we actually change?

To come to think of it, all we did was change the order of the loop of normal matrix multiplication. Instead of doing the normal $i,j,k$ loop, we did $k, i, j$ loop.  The loop order and index changes are colour coded to illustrate how interpretation of inner/outer product boils down to a simple order change of loops. Example in python:

{{< notice tip >}}
It might be worthwhile to look at the code example below and then go back to our animation for a moment to really grasp what changed visually!
{{< /notice >}}
```python
def matrix_multiply_inner(A, B):
    m, n = A.shape
    n, o = B.shape
    C = np.zeros((m, o))
    
    # Using three nested loops in the order of i, j, k
    for i in range(m):
        for j in range(o):
            # load the row i of A and column j of B to 
            # shared memory (2 vectors loaded m * o times)
            for k in range(n): 
                C[i, j] += A[i, k] * B[k, j]  # Doing n Element-wise 
                # multiplication and summation for the two 
                # vectors loaded aka inner product
    
    # The above is equivalent to the following:
    # for i in range(m):
    #     for j in range(o):
    #         C[i, j] = np.dot(A[i], B[:, j])  # Inner product

    return C

def matrix_multiply_outer(A, B):
    m, n = A.shape
    n, o = B.shape
    C = np.zeros((m, o))
    
    # Using three nested loops in the order of k, i, j
    for k in range(n):
        # load the column k of A and row k of B to shared memory
        # (2 vectors loaded n times)
        for i in range(m): 
            for j in range(o):  # The two above loops make O(m * o)
              # multiplications for the two vectors loaded aka outer product
                C[i, j] += A[i, k] * B[k, j]  #Accumulate outer product
                # contributions to its right place

    # The above is equivalent to the following:
    # for k in range(n):
    #     C += np.outer(A[:, k], B[k])  # Outer product

    return C
```

To support the above claim and to see whether the outer product approach is indeed faster, Lets do a couple of experiment in Triton, a language which enables us get control over the multiplicaton kernels directly. To remove complexities of tiling and parallelization for now, we shall just compare the two approaches in the simplest way possible where we are running just 1 thread and therefore both comparisons are doing all the $M\times N$ computations. The experiments are done on an $A100$ with 40GB of VRAM. (For the full Triton experiment code, see the [Appendix](#appendix) section below.)

## Experiment Results

Here's a look at the execution times (in milliseconds) for different matrix sizes with BLOCKSIZE=128:

| Matrix Size | Inner Product | Outer Product | Speed Improvement |
| ----------- | ------------- | ------------- | ----------------- |
| 16×16       | 0.088064      | 0.025600      | 3.4×              |
| 32×32       | 0.349184      | 0.036864      | 9.5×              |
| 64×64       | 1.356800      | 0.062464      | 21.7×             |
| 128×128     | 6.077440      | 0.114688      | 53.0×             |

The data clearly shows that while both approaches scale with increasing matrix sizes, the outer product approach maintains significantly better performance - up to 53× faster for 128×128 matrices! Even more interestingly, our data with larger block sizes (256 and 512) shows that this performance gap widens as matrices get larger, precisely because the cache efficiency benefits become more pronounced with larger datasets.

This experiment elegantly demonstrates our earlier theory: when we organize computation to maximize work done per memory access (as in the outer product approach), we achieve substantially better performance despite performing the same mathematical operations.

{{<plotly json="blockwise_128_graph.json" height="300px" width="50vw">}}

We also see that the slope of the outer product is much less steeper than the inner product. 
<!-- This is because the outer product is more cache friendly and hence the overhead of bringing the data to the working memory is less. -->

{{< notice note >}}
Since the plot's axis is logarithmic, you may get a better idea of comparison by hovering over the datapoints.
{{< /notice >}}
Peculiarly! We see that when playing around with smaller matrices, in larger block sizes; the inner product approach could be faster than the outer product (Block Size 512, Matrix Size 16x16 in the below graph). This is because the overhead of writing $BLOCK\\_SIZE^2$ matrix to the VRAM is more than the actual computation itself and hence an unfair comparison. Nevertheless, the rate at which the time increases is still lower in the outerproduct method, further reinforcing our belief of its better scalability and efficiency. This is a good example of how the hardware and the way we write our code can affect the performance.

{{<plotly json="inner_outer_graph.json" height="400px" width="50vw">}}

## Conclusion

Thinking about matrix multiplication as inner products (row-by-column) or outer products (column-by-row) gives us two valid perspectives, but they’re not equally efficient in practice.

The key takeaway? How we write our loops matters. By understanding the math and the hardware, we can squeeze out better performance in our matrix multiplication code. Often times, the "human" and intuitive way of thinking about a concept may not be ideal. This realisation also hit me while  reading FlashAttention. In Attention, we often think from the perspective of queries and getting the correct combination of value vectors for this query using the keys. Though intuitive, this is not the most efficient way and switching the loop order is one of the key optimizations that enabled FlashAttention to tile the operations nicely and make it faster than the original.
In this small write up, we didn't look at any parallelism and yet managed to find ways to save some time. Next, We shall look at how further tiling and parallelisation can be done to make the matrix multiplication even faster. 

{{< notice info >}}
The code and experiments presented in this article are simplified "toy" examples intended solely to highlight how loop ordering and cache efficiency affects the performance of fundamental operations, particularly matrix multiplication.

For large-scale real-world scenarios, matrix multiplication kernels are heavily optimized beyond simple loop reordering. Optimized libraries (like cuBLAS, cuDNN, CUTLASS or highly tuned CMSIS kernels for CPU) utilize advanced optimization techniques such as tiling, shared memory reuse, register blocking, vectorization, instruction-level parallelism, and carefully tuned parallel kernels. As a result, they typically outperform straightforward textbook implementations (including those demonstrated above) by orders of magnitude.
{{< /notice >}}

---



### References:

- [Animation for inner and outer product made using manim. Code for the animation can be found on this Gist](https://gist.github.com/Bhuvanesh09/d3ab95084a5e8d133a91f4d64ecc2639)
- [Matrix Multiplication as Inner and Outer Product written in Triton](https://gist.github.com/Bhuvanesh09/3e826a488bb3d087ad408d1119c8f5f1)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://github.com/Dao-AILab/flash-attention)

## Appendix 

### Triton code for comparing the matmul operations:

```python
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

```
