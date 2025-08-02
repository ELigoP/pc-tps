from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


# Data Classes (as provided in the prompt)
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
    memory_bw_per_module_gbps: float


@dataclass
class GPU(Buyable):
    vram_gb: float
    memory_bw_gbps: float
    tflops: float


@dataclass
class CPU(Buyable):
    tflops: float


@dataclass
class LLM:
    name: str
    total_parameters: float  # in billions
    active_parameters_per_token: float  # in billions
    # kvcache_size_per_user_token is estimated as 2 * num_layers * hidden_dim
    # We will use a simplified value in GB for 1 bit precision
    kvcache_size_per_user_token_gb: float  # GB per token at 1-bit precision
    bits_per_weight: int = (
        5  # Default to 5-bit quantization (4-bit on largest layers and 8-, 16- and 32- bits on accumulation layers)
    )
    efficiency_factor: float = 0.6  # Practical efficiency of GPU FLOPs


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

        self.ram_sticks = [ram for _ in range(self.motherboard.ram_slots)]
        self.gpus = [gpu for _ in range(self.motherboard.pcie_slots)]
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
        kv_cache_bpw: int = 8,  # Use 8-bit (1 byte) for KV cache by default
    ) -> dict:
        """Calculate LLM performance metrics based on PC build hardware and LLM model."""

        if not self.gpus or not self.ram_sticks or not self.motherboard or not self.cpu:
            return {"error": "Incomplete PC build."}

        # 1. System Parameters
        num_gpus = len(self.gpus)
        gpu = self.gpus[0]
        ram = self.ram_sticks[0]

        total_vram_gb = num_gpus * gpu.vram_gb
        total_gpu_bw_gbps = num_gpus * gpu.memory_bw_gbps
        total_gpu_tflops = num_gpus * gpu.tflops

        total_ram_gb = len(self.ram_sticks) * ram.capacity_gb
        total_ram_bw_gbps = len(self.ram_sticks) * ram.memory_bw_per_module_gbps

        # PCIe bandwidth in GB/s per x16 slot
        pcie_gen_bw = {4: 32, 5: 64}
        # Effective PCIe BW is the total bandwidth available for CPU-GPU communication
        # It's limited by the number of GPUs communicating simultaneously.
        pcie_bw_gbps = pcie_gen_bw.get(self.motherboard.pcie_gen, 64) * num_gpus

        # 2. Model & Cache Sizes (in GB)
        P_gb = llm.total_parameters * llm.bits_per_weight / 8
        A_gb = llm.active_parameters_per_token * llm.bits_per_weight / 8
        kv_cache_gb = (
            context_length * llm.kvcache_size_per_user_token_gb * kv_cache_bpw / 8
        )

        # 3. Weight Distribution
        if kv_cache_gb >= total_vram_gb:
            return {
                "error": f"KV cache ({kv_cache_gb:.1f} GB) exceeds total VRAM ({total_vram_gb:.1f} GB)."
            }

        vram_available_for_weights = total_vram_gb - kv_cache_gb
        weights_on_gpu_gb = min(P_gb, vram_available_for_weights)
        weights_on_cpu_gb = P_gb - weights_on_gpu_gb

        if weights_on_cpu_gb > total_ram_gb:
            return {
                "error": f"Offloaded weights ({weights_on_cpu_gb:.1f} GB) exceeds total RAM ({total_ram_gb:.1f} GB)."
            }

        if P_gb == 0:  # Avoid division by zero
            return {"error": "Total parameters cannot be zero."}

        active_weights_on_gpu_gb = A_gb * (weights_on_gpu_gb / P_gb)
        active_weights_on_cpu_gb = A_gb * (weights_on_cpu_gb / P_gb)

        # Communication bottleneck for CPU-held weights is the minimum of RAM or PCIe bandwidth
        comm_bottleneck_bw_gbps = min(total_ram_bw_gbps, pcie_bw_gbps)

        # --- Time to First Token (TTFT) / Prompt Processing ---
        # Compute time for the entire prompt
        time_compute_prefill = (
            2 * llm.active_parameters_per_token * 1e12 * context_length
        ) / (total_gpu_tflops * 1e12 * llm.efficiency_factor)

        # Memory time to load all active weights once
        time_memory_prefill = (active_weights_on_gpu_gb / total_gpu_bw_gbps) + (
            active_weights_on_cpu_gb / comm_bottleneck_bw_gbps
        )

        ttft = time_compute_prefill + time_memory_prefill
        prompt_processing_speed = context_length / ttft if ttft > 0 else float("inf")

        # --- Time Per Output Token (TPOT) / Token Generation ---
        # This is memory-bound: read active weights + entire KV cache for each token
        time_decode_memory_gpu = (
            active_weights_on_gpu_gb + kv_cache_gb
        ) / total_gpu_bw_gbps
        time_decode_memory_cpu = active_weights_on_cpu_gb / comm_bottleneck_bw_gbps

        tpot = time_decode_memory_gpu + time_decode_memory_cpu
        token_generation_speed = 1 / tpot if tpot > 0 else float("inf")

        return {
            "Prompt Processing (tokens/s)": round(prompt_processing_speed, 2),
            "Token Generation (tokens/s)": round(token_generation_speed, 2),
            "TTFT (s)": round(ttft, 2),
            "TPOT (s)": round(tpot, 2),
            "Weights on GPU (%)": round(100 * weights_on_gpu_gb / P_gb, 1),
            "KV Cache (GB)": round(kv_cache_gb, 1),
        }


# --- Main Execution ---
if __name__ == "__main__":
    # Define Components
    rtx_3090 = GPU(
        name="RTX 3090", price=1000, vram_gb=24, memory_bw_gbps=930, tflops=35
    )
    epyc_9543 = CPU(
        name="Epyc 9543", price=900, tflops=0.3
    )  # Note: CPU TFlops is mostly ignored
    ddr5_ram = RAM(
        name="DDR5 Server RAM 64GB",
        price=300,
        capacity_gb=64,
        memory_bw_per_module_gbps=35,
    )

    # Define Motherboards
    mobo_7_pcie = Motherboard(
        name="7-PCIe Slot Server", price=1500, pcie_slots=7, ram_slots=8
    )
    mobo_4_pcie = Motherboard(
        name="4-PCIe Slot Server", price=1000, pcie_slots=4, ram_slots=12
    )

    # Define LLMs (with estimated parameters for 5-bit quantization)
    # Note: kvcache_size is estimated from model architecture (2 * layers * hidden_dim * bytes)
    # and normalized to GB per token at 1-bit precision.
    llms = [
        LLM(
            name="Mixtral 8x22B (5-bit)",
            total_parameters=141,
            active_parameters_per_token=45,
            kvcache_size_per_user_token_gb=0.6e-3,
        ),
        LLM(
            name="Qwen2 257B-A14B (5-bit)",
            total_parameters=257,
            active_parameters_per_token=14,
            kvcache_size_per_user_token_gb=1.2e-3,
        ),
        LLM(
            name="DeepSeek R1 680B (5-bit)",
            total_parameters=680,
            active_parameters_per_token=20,
            kvcache_size_per_user_token_gb=1.0e-3,
        ),
    ]

    builds = [
        PCBuild(mobo_7_pcie, epyc_9543).fill(ddr5_ram, rtx_3090),
        PCBuild(mobo_4_pcie, epyc_9543).fill(ddr5_ram, rtx_3090),
    ]

    # Run Analysis
    results = []
    context_lengths = [1024, 8192]

    for build in builds:
        for llm in llms:
            for context in context_lengths:
                perf = build.performance(llm, context)
                result_row = {
                    "Build": f"{build.motherboard.pcie_slots}x {build.gpus[0].name}",
                    "LLM": llm.name,
                    "Context": context,
                    **perf,
                }
                results.append(result_row)

    # Display Results
    df = pd.DataFrame(results)
    print("LLM Performance Estimation on Custom PC Builds")
    print("-" * 50)
    print(df.to_string())
