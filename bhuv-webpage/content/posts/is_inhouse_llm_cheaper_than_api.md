+++
title = "Are inhouse LLMs really cheaper than APIs?"
date = "2024-10-05"
math = true
tags = [
    "python", "CS", "LLM" 
]
+++

In the rapidly evolving landscape of Large Language Models (LLMs), organizations face a crucial decision: should they deploy their own in-house models or rely on third-party API services? This analysis delves into the cost and performance dynamics of both approaches, leveraging detailed examinations of token-based latency and throughput.

## Understanding Token Dynamics in LLMs

Tokens are the fundamental units processed by LLMs, encompassing words or subword units. The number of input tokens (prompt length) and output tokens (generated text) directly impact computational resources, latency, and overall costs. Understanding this relationship is essential for optimizing both performance and expenditure.

## Cost vs. Token Length: Analyzing Latency and Throughput
<!---->
<!-- ### Experimental Setup -->
<!---->
<!-- To investigate the relationship between token length and system performance, experiments were conducted using the Llama-7B model with int4 quantization, implemented in TensorRT. The study focused on varying both input and output token lengths to observe their effects on latency and throughput. -->
<!---->
<!-- **Key Parameters:** -->
<!-- - **Model:** Llama-7B (int4 quantization) -->
<!-- - **Framework:** TensorRT -->
<!-- - **Token Ranges:** 256 to 2048 tokens for both input and output, in increments of 256 -->
<!-- - **Trials:** 3 per token length combination -->
<!---->
<!-- ### Understanding Latency Dependencies in Causal LLMs -->
<!---->
<!-- When deploying Large Language Models (LLMs) for tasks such as text generation, understanding and predicting latency is crucial for optimizing performance and cost. Latency in LLMs is influenced by both the number of input tokens (prompt length) and the number of output tokens (generated text). Given the causal nature of these models, we can break down the latency dependencies as follows: -->

#### 1. Linear Dependence on Output Tokens

<iframe width="800" height="600" frameborder="0" scrolling="no" src="//plotly.com/~bhuvanesh09/17.embed"></iframe>

LLMs operate in a causal manner, where each generated token depends on the previously generated tokens. This sequential generation process implies that the latency increases linearly with the number of output tokens. Specifically, the total latency ($ \text{Latency} $) can be expressed as:

$$ \text{Latency} = k_1 \times \text{num\\_output\\_tokens} $$

<!-- Here, \( k_1 \) is a constant that represents the latency incurred per output token. -->
This linear relationship is intuitive because generating each additional token requires a consistent amount of computation, leading to a proportional increase in total latency.

**Note**: We also observe that the slope by which the latency increases with the output tokens is also increseasing with increase in number of input tokens.

#### 2. Linear Dependence on Input Tokens

In addition to output tokens, the latency for generating each output token is also dependent on the number of input tokens. This is due to the attention mechanism in LLMs, where each new token generation step involves attending to the entire input sequence. Consequently, the latency per output token ($ \text{Latency per token} $) scales linearly with the number of input tokens (assuming that the number of output tokens is relatively small compared to the input tokens):

$$
\text{Latency per token} = k_2 \times \text{num\\_input\\_tokens}
$$

Here,  $k_2$ is a constant that encapsulates the latency contribution from each input token per output token generated. As the input token count increases, each output token generation step becomes more computationally intensive, thereby increasing the overall latency.


<iframe width="800" height="600" frameborder="0" scrolling="no" src="//plotly.com/~bhuvanesh09/15.embed"></iframe>

#### Combining Dependencies: A Hyperbolic Relationship

The linear dependencies on both input and output token counts suggest that latency is a function of the product of these two variables. To capture this interaction, we try to fit a hyperbolic model:

$$
z = c \times x \times y
$$

Where:
$ z $ represents the total latency,
$ x $ is the number of input tokens,
$ y $ is the number of output tokens,
$ c $ is a constant that combines $ k_1 $ and $ k_2 $

This hyperbolic function effectively models the combined impact of input and output token lengths on latency. By fitting this model to empirical data, we can accurately predict latency under varying conditions of token lengths, facilitating better resource allocation and cost management in LLM deployments.

#### Predictive Modeling of Latency

A hyperbolic curve was fitted to model latency based on input and output token lengths. The model achieved a **Root Mean Squared Error (RMSE) of 0.01 seconds**!, demonstrating high accuracy in predicting latency across various token lengths.

**Model Details:**
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

poly = PolynomialFeatures(degree=2)
X = np.stack([df.input_size.to_numpy(), df.output_size.to_numpy()]).T
y = df.latency.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LinearRegression()
clf.fit(poly.fit_transform(X_train), y_train)
y_pred = clf.predict(poly.fit_transform(X_test))

rmse = mean_squared_error(y_test, y_pred, squared=False)
```

### Is this level of prediction sufficient?
Unfortunately, most modern deployment system don't operate with a single sequence at a time. Especially with the advent of modern methods like PagedAttention, we can't be certain of which batches of sequence end at what time. They use inflight batching to maximize the throughput of the system where a single sequence of a batch can be replaced with a new request while the rest of the sequences are still processing their respective requests. This means that we need to consider the number of concurrent requests the system can handle at a time. 

#### Extending the Model for Complex Systems

The initial latency model was expanded to include semaphore counts, accommodating more complex deployment scenarios. This extension ensures accurate latency predictions even when handling multiple concurrent requests.

For a bit of emperical validation that what we are doing is correct, lets try to plot the relation of latency against the number of semaphore. This is essential before we try to fit a curve like we did for the input and output tokens. 

<iframe width="800" height="600" frameborder="0" scrolling="no" src="//plotly.com/~bhuvanesh09/19.embed"></iframe>

**Enhanced Model Implementation:**
<!---->
<!-- ```python -->
<!-- poly = PolynomialFeatures(degree=2) -->
<!-- X = np.stack([df["Input Tokens"].to_numpy(), df["Output Tokens"].to_numpy(), df["Semaphore"].to_numpy()]).T -->
<!-- y = df["Mean Latency"].to_numpy() -->
<!---->
<!-- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) -->
<!-- clf = LinearRegression() -->
<!-- clf.fit(poly.fit_transform(X_train), y_train) -->
<!-- y_pred = clf.predict(poly.fit_transform(X_test)) -->
<!---->
<!-- rmse = mean_squared_error(y_test, y_pred, squared=False) -->
<!-- ``` -->
<!---->
The extended model now contains the num semaphore as well as part of the training data. For this regression, we have 3 variables which we raise to the degree of two. This implies that our linear regression is fitting $\binom{3}{2} + \binom{3}{1} + 1 = 10$ variables. The extended model achieved an **RMSE of 0.15 seconds** for predicting the latency given `<input>`, `<output>`, and `<semaphores>` demonstrating reliable performance in predicting latency within complex in-flight batching systems. The increase in RMSE compared to the simpler model is expected due to the additional complexity introduced by semaphore counts, and also because different frameworks have different ways in which the inflight scheduling is handled, ideally, all the sequences should go through the forward pass together but due to restricted number of SMs, etc they don't occur truly parallely.

## Conundrum of Latency vs Throughput: A tradeoff
Now we have a model that can predict the latency of the system given the input tokens, output tokens and the number of semaphore. But we need to understand how the throughput of the system is affected by these parameters. Throughput in tokens per second can easily be calculated by the formula: $$ \text{Throughput} = \text{Mean output tokens per call} \times \frac{\text{Semaphores}}{\text{Latency}} $$.

#### Handling In-Flight Batching

Managing multiple simultaneous requests (in-flight batching) introduces complexity in maintaining low latency and high throughput. Semaphore counts, which determine the number of concurrent requests, play a critical role in this balance. Having a higher semaphore could lead to higher throughput since more number of sequences are processed together albeit at the cost of lantecy. Going to the extremes and overloading the system with too many concurrent requests can lead to increased latency and also reduced throughput efficiency.

**Optimization Strategies:**
- **Semaphore Management:** Optimize the number of concurrent requests depending on the VRAM till the throughput plateaus. This is the max semaphore that we should use albeit it would have the among the slowest latency.

**Maximum Semaphore possible is a function of the VRAM, number of parameters, the number of tokens that would be there in KV Cache and so on.**

Lets take an example of Llama 3.1 8B:
- Number of layers in Llama 3.1 8B: $n_l = 32$
- Dimensionality of model: $d = 4096$
- Number of attention heads: $n_h = 32$
- Dimension per head: $\frac{d}{n_h} = 4096/32 = 128$
- Number of kv\_heads: $n_{kv} = 32/4 = 8$
- Each KV Cache is stored in 2bytes: $n_b = 2$ (FP16)

The amount of memory in GPU VRAM taken by a single token stored in kv_cache would be:
$$ 2 \times n_l \times \frac{d}{n_h} \times n_{kv} \times 2 = 32 * (4096/32) * 8 * 2 * 2 = 131072 \text{bytes} \approx 131 \text{KB}$$

So, considering an input length of 2048, a semaphore of 30 would need around $ 2048 \cdot 30\cdot131KB \approx 8GB$ of memory in VRAM. Which comes just under the budget since on A10G, we have $24GB - 16GB\text{(for the model weights)} = 8GB$ of memory left.

- **Latency Prediction:** Utilize predictive models to estimate and maintain acceptable latency levels under varying loads for the SLA promised.

Example: One may want to keep the latency under 3 second for 90% of the requests. This can be achieved by first seeing how long does it takes the system to process a single sequence and then increasing the number of semaphores till the latency is just under 3 seconds.


## From Token Dynamics to Cost Efficiency: In-House vs. API

### Cost Calculation Framework

Calculating the cost of deploying an in-house LLM involves understanding latency, throughput, and the concurrency level (semaphore). By estimating daily token expenditure based on these parameters, organizations can compare the costs against API-based solutions.

**Cost Estimation Steps:**
1. **Determine Daily Token Throughput:**
   $$
   \text{Input Tokens Served} = \text{Semaphore} \times \text{Input Tokens} \times \left(\frac{24\cdot60\cdot60}{\text{Latency}}\right)
   $$
   $$
   \text{Output Tokens Served} = \text{Semaphore} \times \text{Output Tokens} \times \left(\frac{24\cdot60\cdot60}{\text{Latency}}\right)
   $$

Where the latency and semaphore are the values predicted by the fitted curve while keeping the latency in check.

2. **Calculate API Equivalent Costs:**
   Compare the estimated input and output token counts against API pricing models to determine cost efficiency.

### Case Study: In-House Setup vs. GPT 4o mini

For the sake of argument, consider that we are able to finetune a llama 3.1 8B to match the performance of GPT 4o mini for a niche usecase. 

A comparative analysis between an in-house setup using an A10G GPU with Llama3 and the GPT-4o mini API revealed significant cost savings with the in-house deployment. 

**Example Configuration:**
- **GPU:** A10G
- **Model:** Llama3.1 8B INT8 Quantized
- **Input Tokens per Call:** 2048
- **Output Tokens per Call:** 130
- **Semaphore:** 21
- **Daily Cost:** $13.00 (Rent of an A10G machine reserved for 1 year) 

With our earlier curve fitting, we are able to determine that we would be able to serve the above config with the latency of $12.29$ seconds.
#### Cost for equivalent API: 

$$ \text{Input Tokens Served} = 21 \times 2048 \times \left(\frac{24\cdot60\cdot60}{12.29}\right) = 302,350,789 $$
$$ \text{Output Tokens Served} = 21 \times 130 \times \left(\frac{24\cdot60\cdot60}{12.29}\right) = 19,192,188 $$

The input cost rate is $15$ cents per million tokens and the output cost rate is $60$ cents per million tokens for GPT 4o-mini. Therefore:
$$ \text{Input Cost} = 0.15 \times \frac{302,350,789}{1,000,000} = \\$45.35261835 $$
$$ \text{Output Cost} = 0.60 \times \frac{19,192,188}{1,000,000} = \\$11.5153128 $$
$$ \text{Total Cost} = \\$45.35261835 + \\$11.5153128 = \\$56.86793115 $$

Therefore, the in-house setup proved to be over **four times cheaper** under the tested configurations.


## Conclusion

We are able to show wih very real and plausible scenarios that we can be much cheaper inspite of OpenAI slashing down their costs frequently! This is done without any specific optimization and just out of the box predictions with vLLM 0.6.1. With additional usecase specific optimizations like kv_cache prefix reuse, we can decrease the cost incured by a lot more! 
This analysis provides a comprehensive comparison of in-house LLM deployments versus API-based solutions, grounded in empirical data and robust modeling techniques. By leveraging detailed token and latency analyses, organizations can optimize their AI deployments for both performance and cost-efficiency.
Thanks a lot for sticking around till the end!


## References

- **PagedAttention:** [arxiv link](https://arxiv.org/abs/2309.06180)
- [**kipply's blog for inference arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)
- [vLLM Inference Library](https://github.com/vllm-project/vllm)
- [Llama 3.1 8B Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

