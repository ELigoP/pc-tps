import math
from dataclasses import dataclass
from typing import List, Optional

# --- Component and Model Definitions ---


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
    pcie_gen: int = 4


@dataclass
class CPU(Buyable):
    tflops: float


@dataclass
class LLM:
    name: str
    total_parameters: float  # in billions
    active_parameters_per_token: float  # in billions
    always_active_parameters_per_token: float  # in billions
    layers: int
    hidden_dim: int
    bits_per_weight: int = 5
    efficiency_factor: float = 0.5


# --- PC Build Class with Performance Modeling ---


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

    def __str__(self):
        return f"{len(self.gpus)}x{self.gpus[0].vram_gb}GB GPUs, {sum((stick.capacity_gb for stick in self.ram_sticks))}GB RAM ({self.motherboard.name})"

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
            return 0.0

        motherboard_cost = self.motherboard.price
        cpu_cost = self.cpu.price
        ram_cost = sum(ram.price for ram in self.ram_sticks)
        gpu_cost = sum(gpu.price for gpu in self.gpus)

        return motherboard_cost + cpu_cost + ram_cost + gpu_cost

    def performance(
        self,
        llm: LLM,
        context_length: int,
        kv_cache_bpw=1,  # Default to 16-bit (2 bytes) for KV cache
    ) -> dict:
        """Calculate LLM performance metrics based on PC build hardware and LLM model."""
        if not self.motherboard or not self.cpu or not self.gpus or not self.ram_sticks:
            raise ValueError("PC components are not fully specified.")

        # --- 1. System and Model Parameter Initialization ---

        # Hardware parameters
        num_gpus = len(self.gpus)
        gpu = self.gpus[0]
        ram = self.ram_sticks[0]

        # Effective (real-world) performance after applying efficiency factor
        eff = llm.efficiency_factor
        gpu_tflops_single = gpu.tflops * eff
        gpu_tflops_total = num_gpus * gpu_tflops_single
        gpu_mem_bw_total = num_gpus * gpu.memory_bw_gbps * eff
        cpu_tflops = self.cpu.tflops * eff
        ram_bw_total = len(self.ram_sticks) * ram.memory_bw_per_module_gbps * eff

        pcie_bw_gbps = {5: 64, 4: 32, 3: 16}.get(
            min(self.motherboard.pcie_gen, gpu.pcie_gen), 64
        ) * eff

        # Convert to base units (Bytes, FLOPs/s)
        GB_to_B = 10**9
        TFLOPs_to_FLOPs = 10**12

        vram_total_bytes = num_gpus * gpu.vram_gb * GB_to_B
        ram_total_bytes = len(self.ram_sticks) * ram.capacity_gb * GB_to_B
        gpu_flops_per_s = gpu_tflops_total * TFLOPs_to_FLOPs
        gpu_flops_single_per_s = gpu_tflops_single * TFLOPs_to_FLOPs
        gpu_bw_bytes_per_s = gpu.memory_bw_gbps * GB_to_B * eff
        pcie_bw_bytes_per_s = pcie_bw_gbps * GB_to_B
        ram_bw_bytes_per_s = ram_bw_total * GB_to_B

        # LLM parameters
        bytes_per_weight = llm.bits_per_weight / 8.0

        P_bytes = llm.total_parameters * 1e9 * bytes_per_weight
        A_bytes = llm.active_parameters_per_token * 1e9 * bytes_per_weight
        S_bytes = llm.always_active_parameters_per_token * 1e9 * bytes_per_weight

        P_exp_bytes = P_bytes - S_bytes

        kv_cache_bytes = 2 * llm.layers * llm.hidden_dim * context_length * kv_cache_bpw

        # --- 2. Layer Placement Logic ---

        # Prioritize KV Cache in VRAM
        kv_on_gpu = min(kv_cache_bytes, vram_total_bytes)
        vram_rem = vram_total_bytes - kv_on_gpu

        # Place Always-Active Parameters
        s_on_gpu = min(S_bytes, vram_rem)
        s_on_ram = S_bytes - s_on_gpu
        vram_rem -= s_on_gpu

        # Place Expert Parameters
        p_exp_on_gpu = min(P_exp_bytes, vram_rem)
        p_exp_on_ram = P_exp_bytes - p_exp_on_gpu

        # Determine Active Parameter Distribution
        alpha_gpu = p_exp_on_gpu / P_exp_bytes if P_exp_bytes > 0 else 0

        A_gpu_bytes = s_on_gpu + (A_bytes - S_bytes) * alpha_gpu
        A_ram_bytes = s_on_ram + (A_bytes - S_bytes) * (1 - alpha_gpu)

        # --- 3. Prefill Phase (TTFT) Calculation ---

        # Compute Time
        flops_parametric = 2 * llm.active_parameters_per_token * 1e9 * context_length
        flops_attention = 2 * llm.layers * llm.hidden_dim * (context_length**2)
        total_prefill_flops = flops_parametric + flops_attention
        t_compute_prefill = total_prefill_flops / gpu_flops_single_per_s

        # Memory Time
        # Time to load active weights from VRAM (sequential across GPUs) and RAM (sequential over PCIe)
        t_memory_prefill = (
            A_gpu_bytes
            / (
                # num_gpus *  # if tensor parallel
                gpu_bw_bytes_per_s
            )
        ) + context_length * (A_cpu_bytes / pcie_bw_bytes_per_s)

        # Communication Time (Pipeline Parallelism)
        activation_size_bytes = llm.hidden_dim * context_length * bytes_per_weight
        t_comm_prefill = (
            (num_gpus - 1) * activation_size_bytes / pcie_bw_bytes_per_s
            if num_gpus > 1
            else 0
        )

        ttft = t_compute_prefill + t_memory_prefill + t_comm_prefill

        # --- 4. Decode Phase (TPOT) Calculation ---

        # Compute Time
        total_decode_flops = 2 * llm.active_parameters_per_token * 1e9
        t_compute_decode = total_decode_flops / gpu_flops_per_s

        # Memory Time (Dominant Factor)
        # Read active weights AND the entire KV cache for each token
        t_memory_decode = (
            (
                A_gpu_bytes
                / (
                    # num_gpus * # if TP
                    gpu_bw_bytes_per_s
                )
            )
            + (A_cpu_bytes / pcie_bw_bytes_per_s)
            + (
                kv_on_gpu
                / (
                    # num_gpus * # if TP
                    gpu_bw_bytes_per_s
                )
            )
        )

        # Communication Time
        activation_size_bytes_decode = llm.hidden_dim * 1 * bytes_per_weight
        t_comm_decode = (
            (num_gpus - 1) * activation_size_bytes_decode / pcie_bw_bytes_per_s
            if num_gpus > 1
            else 0
        )

        tpot = t_compute_decode + t_memory_decode + t_comm_decode
        tokens_per_sec = 1.0 / tpot if tpot > 0 else float("inf")

        return {
            "llm_name": llm.name,
            "context_length": context_length,
            "total_vram_gb": vram_total_bytes / GB_to_B,
            "total_ram_gb": ram_total_bytes / GB_to_B,
            "kv_cache_size_gb": kv_cache_bytes / GB_to_B,
            "model_on_cpu_gb": (s_on_cpu + p_exp_on_cpu) / GB_to_B,
            "active_params_on_cpu_gb": A_cpu_bytes / GB_to_B,
            "ttft_s": ttft,
            "tppt_s": context_length / ttft,
            "tpot_s": tpot,
            "tokens_per_second": tokens_per_sec,
            "breakdown_ttft_ms": {
                "compute": t_compute_prefill * 1000,
                "memory": t_memory_prefill * 1000,
                "communication": t_comm_prefill * 1000,
            },
            "breakdown_tpot_ms": {
                "compute": t_compute_decode * 1000,
                "memory_vram": (
                    (A_gpu_bytes + kv_on_gpu)
                    / (
                        # num_gpus *  # if tensor parallel
                        gpu_bw_bytes_per_s
                    )
                )
                * 1000,
                "memory_pcie": (A_cpu_bytes / pcie_bw_bytes_per_s) * 1000,
                "communication": t_comm_decode * 1000,
            },
        }


def print_round_dict_values(d):
    for k, v in d.items():
        print(f"{k}: {round(v)}")


# --- Main Execution Block ---

if __name__ == "__main__":
    # Define Components
    rtx_3090 = GPU(
        name="RTX 3090", price=1100, vram_gb=24, memory_bw_gbps=930, tflops=35
    )
    tp_3970 = CPU(name="Threadripper 3970X", price=650, tflops=0.15)
    epyc_9543 = CPU(name="Epyc 9543", price=900, tflops=0.3)
    ddr5_ecc = RAM(
        name="DDR5 ECC 64GB", price=300, capacity_gb=64, memory_bw_per_module_gbps=35
    )
    ddr4_non_ecc = RAM(
        name="DDR4 ECC 32GB", price=150, capacity_gb=64, memory_bw_per_module_gbps=20
    )

    old_mb_4_gpu = Motherboard(
        name="4-Slot Old Workstation", price=350, ram_slots=3, pcie_slots=4, pcie_gen=3
    )
    mb_7_gpu = Motherboard(
        name="7-Slot Workstation", price=1400, ram_slots=8, pcie_slots=7
    )
    mb_4_gpu = Motherboard(
        name="4-Slot Workstation", price=1000, ram_slots=12, pcie_slots=4
    )

    # Define LLMs
    mixtral_8x22b = LLM(
        name="Mixtral 8x22B",
        total_parameters=141,
        active_parameters_per_token=39,
        always_active_parameters_per_token=5.0,  # Derived
        layers=56,
        hidden_dim=6144,
    )

    qwen3_235b = LLM(
        name="Qwen3 235B-A22B",
        total_parameters=235,
        active_parameters_per_token=22,
        always_active_parameters_per_token=7.8,  # Derived
        layers=94,
        hidden_dim=4096,
        bits_per_weight=3.5,
    )

    deepseek_r1_params = dict(
        total_parameters=671,
        active_parameters_per_token=37,
        always_active_parameters_per_token=16.55,  # Derived
        layers=61,
        hidden_dim=7168,
    )
    models = [qwen3_235b]
    for bpw in (
        # 3,
        # 5,
        # 7,
    ):
        models.append(
            LLM(
                **deepseek_r1_params,
                name=f"DeepSeek R1 {bpw}bit",
                bits_per_weight=bpw,
            )
        )

    # Create PC Builds
    builds: List[PCBuild] = []
    for motherboard, cpu, ram, gpu in (
        (old_mb_4_gpu, tp_3970, ddr4_non_ecc, rtx_3090),
        (mb_4_gpu, epyc_9543, ddr5_ecc, rtx_3090),
        (mb_7_gpu, epyc_9543, ddr5_ecc, rtx_3090),
    ):
        builds.append(PCBuild(motherboard, cpu).fill(ram, gpu))

    # Run Performance Analysis
    print(
        f"{'LLM':<20} "
        f"| {'TTFT (s)':<15} "
        f"/ {'Compute (ms)':<15} "
        f"/ {'Memory (ms)':<15} "
        f"/ {'Communication (ms)':<20} "
        f"| {'TPPT (tok/s)':<15} "
        f"| {'TPOT (tok/s)':<15} "
    )
    print("-" * 145)

    for build in builds:
        print(build)
        for model in models:
            for context_length in (16, 512, 16384):
                print(f"CTX={context_length}")
                results = build.performance(
                    llm=model, context_length=context_length, kv_cache_bpw=2
                )

                print(
                    f"{results['llm_name']:<20} "
                    f"| {results['ttft_s']:<15.2f} "
                    f"/ {results['breakdown_ttft_ms']['compute']:<15.2f} "
                    f"/ {results['breakdown_ttft_ms']['memory']:<15.2f} "
                    f"/ {results['breakdown_ttft_ms']['communication']:<20.2f} "
                    f"| {results['tppt_s']:<15.2f} "
                    f"| {results['tokens_per_second']:<15.2f}"
                )
                # for k in (
                #     "breakdown_ttft_ms",
                #     "breakdown_tpot_ms",
                # ):
                #     print(f"----{k.upper().replace('_',' ')}:")
                #     print_round_dict_values(results[k])
