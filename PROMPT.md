Large Language Models (LLMs) performance on a specific system is characterized in two ways:

1. prompt processing speed (tokens/second) or Time to First Token (seconds for given prompt length);
2. token generation speed (tokens/second).

In the simple case, prompt processing is usually compute bound (because you read layer weight once and the process many tokens without using much memory reads, then read next layer(s) weights and so on), and token generation speed is usually memory bound (for each new token you need to read entire layer(s) weights).

For the simple case of small model that fully fits in a single GPU, there is analysis done in https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/:

<citation>
Method
A GPU does two things: loading data and computing FLOPs. To estimate time is to estimate the total memory and compute work in each stage of LLM inference and divide by the GPU's processing rate. For that, you need the following constants:

- `s`: sequence length
- `b`: batch size
- `h`: hidden dimension
- `L`: number of transformer layers
- `N`: model parameters
- GPU FLOPs rate. For A100, this is 312 TFLOPs.
- GPU Memory bandwidth rate. For A100, this is 1.5 TB/second.

We assume 16-bit precision (2 bytes per parameter), the formulae are:

<formula>Prefill Compute = 2 * N * b * s</formula>
<formula>Decode Compute = 2 * N * b * 1</formula>

Explanation: 
2*N is the approximated amount of compute needed to process each token. Consider the product
<formula>W*x = w_11*x_1 + w_12*x_2 + ...</formula>,
where W is the weight matrix. Each parameter w_ij is involved in one multiplication and one summation. <formula>b*s</formula> is the total number of tokens to process. In the decode stage, we produce one token at a time so
<formula>s = 1</formula>.

<formula>Prefill Memory = Decode Memory = 2 * N</formula>

Explanation: Every model parameter needs to be loaded on to the GPU for computation. `2` converts model parameter count to amount of bytes assuming 16-bit precision.

Time-To-First-Token (TTFT)
<formula>(Prefill Compute)/(FLOPs rate) + (Prefill Memory)/(Memory BW rate)</formula>

Time-Per-Output-Token (TPOT)
<formula>(Decode Compute)/(FLOPs rate) + (Decode Memory)/(Memory BW rate)</formula>

VRAM used
<formula>VRAM = (Model Parameters) * 2 + (KV Cache) * 2 = (Model Parameters) * 2 + (2 * h * L * b * s) * 2</formula>

Let's interpret the KV cache expression from left two right. Each token has one cached key vector and one cached value vector in each attention head (The first 2). The parameter dimension after aggregating all attention heads is the model dimension (* h). There are (b * s) tokens in token. The final *2 converts parameter to bytes assuming 16-bit precision.
</citation>

Note that
- this is simple model, which will not hold when layers are distributed between GPUs/their VRAM and CPU/RAM.
- 16-bit precision is assumed often in citation, this is not true if model is quantized.

I want to calculate estimates for prompt processing and token generation speeds for the much more complex case: Mixture-of-Experts LLM weights distrubuted accross multiple GPUs and RAM, and computation done in GPUs and CPU.

Usually, majority of MoE model weights are:
Number of big blocks of identical architecture (but different learned weights), each of this consists mainly of smaller layers:
- Attention layer, which passes its output to
- Routing layer, which decides to which experts to pass attention layer output
- Shared experts layers, which always process attention layer output
- Routed experts layers, of which fixed number is selected by router layer, for performance estimation purpose we can assume selection is random
Then shared and routed experts layers output is added together and passed to the next big blocks.
We can say that there are
- always active layers (attention, routing and shared expert layers in each of the big block)
- active layers (always active layers + routed expert layers selected by routing layer)
- all the layers (active + unselected routed expert layers) - still, every of them should be kept in VRAM/RAM

When model does not fully fit in VRAM, the optimal layers placement strategy is like this:

1. Try distribute always active layers across GPUs, as uniformly as possible; if not enough VRAM, offload always active layers that didn't fit and all the rest to the RAM
2. if there is VRAM left, then distribute shared experts layers across GPUs, if possible, (placed routed expert layers on the same GPU where shared expert layer of same big block is, if possible), in order to fill VRAM, until VRAM cannot be used anymore;
3. Then offload yet undistributed shared expert layers to RAM.

An example:
- e.g. for DeepSeek R1, there are about 60 layers; in those layers, there is 1 shared expert tensor group which is always active, and 256 routed expert tensor groups, for each token 8 routed expert tensor groups are selected based on router score; so, if we neglect other less computationally significant tensors
- router tensor group and shared expert tensor group are always active
- always active + 8 routed expert tensor groups are active parameters

For simplicity, assume that motherboard has
- `N` PCIe slots all occupied with identical GPUs, and
- `n` RAM slots are occupied with identical RAM modules.
Batch size is 1, and there is single request.

What parameters to use:

CPU has parameter
- `f` (maximum TFlops)

RAM module has parameters
- `m` (memory size in GB)
- `b` (memory speed in GB/s)

Assume model layers are uniormly distributed across memory modules, so total memory bandwidth adds up from each module memory bandwidth.

GPU has parameters:
- `F` (maximum TFlops)
- `B` (GPU memory bandwidth)
- `M` (memory size in GB)

In contrast to RAM, layers are not distributed accross GPU, as their interconnect (PCIe) bandwidth is not high and computations are done on each GPU. So operations are done sequentially if model is large enough not to fit in single GPU - result of computations of 1st GPU (some first layers) goes to 2nd (where next layers computations are performed) and so on.

PCIe slots parameters:
- `p` (bandwidth in GB/s)

Model parameters:
- `P` (number of parameters in billions)
- `A` (number of active parameters per token, including experts weights that are activated from time to time) 
- `S` (number of parameters in the experts which are always active)
- `L` (number of layers, each layer can have many tensors)
- `Q` (weights quantization level, in bits per parameter; this characterizes how much weights are compressed and could be fit and read faster from memory)
- `q` (KV cache quantization level, in bits per parameter)
- `H` (inter-layer hidden dimension)
- `h` (intra-layer, or tensor-to-tensor hidden dimension)

Also, I read and observed, but didn't quite understand why, that there is some factor proportional to context length squared (maybe it is related to K*V in attention layers, where K and V sizes are proportional to the context length). Hence, one more parameter:

- `C` (context length).

CPU has lots of RAM but not great bandwidth compared to GPUs, and regrettable TFlops.
KV Cache and always active parameters should be split between GPUs; now rest of active parameters that are activated from time to time should be split between GPUs and CPU - fill GPUs until the VRAM is exhausted, with some buffer left, and place rest on the CPU.

Help me to write
1. formulas for the prompt processing and token generation speed for MoE model split across GPUs and CPU
2. Python script to estimate those for some systems and LLMs.

P.S. Some data to use:

RTX 3090 GPU: 24GB VRAM, 930 GB/s VRAM bandwidth, 35 TFlops
Epyc 9543 CPU: 0.3 TFlops
Typical DDR5 ECC Server RAM module read speed is 35 GB/s
PCIe Gen 5 speed is 64 GB/s per 8 lanes

Estimate performance for two motherboards:
- 7 PCIe Gen 5 slots and 8 DDR5 slots
- 4 PCIe Gen 5 slots and 12 DDR5 slots

In Python script, get estimates for these models:

- Mixtral 8x22B
- Qwen3 235B-A22B
- DeepSeek R1 (680B)

Use dataclasses for Motherboard, CPU, GPU, RAM, LLM

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Buyable:
    name: str
    price: float


@dataclass
class Motherboard(Buyable):
    ram_slots: int
    pcie_slots: int
    pcie_gen: int = 5


@dataclass
class RAM(Buyable):
    capacity_gb: float
    memory_bw_per_module_gbps: float  # GB/s


@dataclass
class GPU(Buyable):
    vram_gb: float
    memory_bw_gbps: float  # GB/s
    tflops: float


@dataclass
class CPU(Buyable):
    tflops: float


@dataclass
class LLM:
    name: str
    total_parameters: float  # in billions
    active_parameters_per_token: float  # in billions
    always_active_parameters_per_token: float  # in billions
    activations_size_per_token: float  # GB
    kvcache_size_per_user_token: float  # GB
    hidden_dim: int
    bits_per_weight: int = 5
    efficiency_factor: float = 0.5
```

Implement `PCBuild` class with following methods, the main method - `.perfomance(...)`:

```python
class PCBuild:
    def __init__(
        self,
        motherboard: Optional[Motherboard] = None,
        cpu: Optional[CPU] = None,
    ):
        self.motherboard = motherboard
        self.cpu = cpu
        self.ram_sticks: List[RAM] = []
        self.gpus: List[GPU] = []

    def fill(self, ram: RAM, gpu: GPU):
        """Fill available slots with specified components"""
        if not self.motherboard:
            raise ValueError("Motherboard must be selected first")

        # Fill RAM slots
        self.ram_sticks = [ram for _ in range(self.motherboard.ram_slots)]

        # Fill GPU slots (practical limit of 8 GPUs)
        num_gpus = self.motherboard.pcie_slots
        self.gpus = [gpu for _ in range(num_gpus)]

        return self

    def price(self) -> float:
        """Calculate total build price"""
        if not self.motherboard or not self.cpu:
            return 0

        motherboard_cost = self.motherboard.price
        cpu_cost = self.cpu.price
        ram_cost = sum(ram.price for ram in self.ram_sticks)
        gpu_cost = sum(gpu.price for gpu in self.gpus)

        return motherboard_cost + cpu_cost + ram_cost + gpu_cost

    def performance(
        self,
        llm: LLM,
        context_length: int,
        kv_cache_bpw=1,
    ) -> dict:
        """Calculate LLM performance metrics based on PC build hardware, LLM model, context length and KV cache quantization"""

        # TODO
```