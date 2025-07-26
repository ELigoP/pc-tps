from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Motherboard:
    name: str
    price: float
    ram_slots: int
    pcie_slots: int
    pcie_gen: int = 5


@dataclass
class RAM:
    name: str
    price: float
    capacity_gb: float
    bandwidth_per_stick: float  # GB/s


@dataclass
class GPU:
    name: str
    price: float
    vram_gb: float
    memory_bw: float
    tflops: float


@dataclass
class CPU:
    name: str
    price: float
    tflops: float


@dataclass
class LLM:
    total_parameters: float  # in billions
    active_parameters_per_token: float  # in billions
    shared_parameters_per_token: float  # in billions
    activations_size_per_token: float  # GB
    kvcache_size_per_user_token: float  # GB
    hidden_dim: int
    bits_per_weight: int = 5
    efficiency_factor: float = 0.5


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

    @property
    def pcie_bw(self) -> int:
        ASSUMED_N_PCIE_LANES = 8
        return (
            2 ** (self.motherboard.pcie_gen - 1) * ASSUMED_N_PCIE_LANES
            if self.motherboard
            else None
        )

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
        if not self.motherboard:
            return 0

        motherboard_cost = self.motherboard.price
        ram_cost = sum(ram.price for ram in self.ram_sticks)
        gpu_cost = sum(gpu.price for gpu in self.gpus)

        return motherboard_cost + ram_cost + gpu_cost

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
        parallel_requests: int = 1,
        cache_bpw=1,
    ) -> dict:
        """Calculate LLM performance metrics based on hardware."""
        if not self.motherboard or not self.gpus:
            return {}

        # --- System-wide hardware stats ---
        num_gpus = len(self.gpus)
        total_vram_gb = sum(gpu.vram_gb for gpu in self.gpus)
        total_gpu_tflops = sum(gpu.tflops for gpu in self.gpus)
        total_memory_bw = sum(gpu.memory_bw for gpu in self.gpus)

        # --- VRAM & Memory Allocation ---
        activations_required_gb = context_length * llm.activations_size_per_token
        kv_cache_required_gb = (
            context_length
            * parallel_requests
            * llm.kvcache_size_per_user_token
            * (cache_bpw / 8)
        )
        model_weights_gb = llm.total_parameters * (llm.bits_per_weight / 8)
        # at least that much should be placed in VRAM for better performance
        shared_weights_gb = llm.shared_parameters_per_token * (llm.bits_per_weight / 8)
        required_memory_gb = (
            model_weights_gb + activations_required_gb + kv_cache_required_gb
        )

        # Determine how weights are split between VRAM and system RAM
        available_vram_for_weights = total_vram_gb - kv_cache_required_gb
        weights_in_vram_gb = min(model_weights_gb, available_vram_for_weights)
        offloaded_to_ram_gb = model_weights_gb - weights_in_vram_gb

        vram_warning = None
        if total_vram_gb < kv_cache_required_gb:
            vram_warning = (
                f"Insufficient VRAM for KV Cache alone! "
                f"Need {kv_cache_required_gb:.1f}GB, "
                f"have {total_vram_gb:.1f}GB."
            )
        elif total_vram_gb < kv_cache_required_gb + shared_weights_gb:
            vram_warning = (
                f"VRAM is not enough for all shared experts."
                f"Offloading {offloaded_to_ram_gb:.1f}GB to system RAM."
            )
        elif total_vram_gb < required_memory_gb:
            vram_warning = (
                f"VRAM is not enough for the model."
                f"Offloading {offloaded_to_ram_gb:.1f}GB to system RAM."
            )

        # --- 1. PROMPT PROCESSING SPEED (TTFT-related) ---
        # Time to process one token in the initial prompt (prefill stage)
        # This involves compute, memory loads, and inter-GPU communication.

        # a) Compute time for the whole model
        # assuming same GPU
        # assuming sequential data transfer from previous to next GPUs
        # + proportion of experts offloading to CPU
        prefill_flops_per_token = 2 * llm.total_parameters * 1e9
        time_prefill_compute = prefill_flops_per_token / (
            self.gpus[0].tflops * 1e12 * llm.efficiency_factor
        )

        # b) Memory load time (loading all weights for one forward pass)
        time_prefill_hbm_load = weights_in_vram_gb / self.gpus[0].tflops
        time_prefill_pcie_load = offloaded_to_ram_gb / self.pcie_bw

        # c) Communication overhead (simplified All-Reduce for Tensor Parallelism)
        # Assuming batch size of 1 for this calculation.
        bytes_to_communicate = (
            2 * (num_gpus - 1) * llm.hidden_dim * 2
        )  # 2 bytes for FP16
        time_prefill_comm = (
            bytes_to_communicate / (self.pcie_bw * 1e9) if num_gpus > 1 else 0
        )

        time_per_prompt_token = (
            time_prefill_compute
            + time_prefill_hbm_load
            + time_prefill_pcie_load
            + time_prefill_comm
        )
        prompt_speed_tps = 1 / time_per_prompt_token if time_per_prompt_token > 0 else 0

        # --- 2. TOKEN GENERATION SPEED (TPOT-related) ---
        # Time to generate one new token (decode stage for MoE)
        # This is dominated by loading active params (experts) from RAM and compute.

        # For MoE, we assume only active parameters are in VRAM, the rest must be fetched.
        active_params_in_vram_b = (
            model_weights_gb
            / (model_weights_gb + offloaded_to_ram_gb)
            * llm.active_parameters_per_token
            if (model_weights_gb + offloaded_to_ram_gb) > 0
            else 0
        )
        active_params_in_ram_b = (
            llm.active_parameters_per_token - active_params_in_vram_b
        )

        # a) Time to load the required 'expert' weights from offloaded RAM
        bytes_from_ram_per_token = active_params_in_ram_b * 1e9 * bytes_per_param
        time_decode_pcie_load = (
            bytes_from_ram_per_token / (self.pcie_bw * 1e9)
            if offloaded_to_ram_gb > 0
            else 0
        )

        # b) Compute time for the active parameters
        decode_flops_per_token = 2 * llm.active_parameters_per_token * 1e9
        time_decode_compute = decode_flops_per_token / (
            total_gpu_tflops * 1e12 * llm.efficiency_factor
        )

        # c) Communication overhead (same as prefill for simplicity)
        time_decode_comm = time_prefill_comm

        time_per_output_token = (
            time_decode_pcie_load + time_decode_compute + time_decode_comm
        )
        gen_speed_tps = 1 / time_per_output_token if time_per_output_token > 0 else 0

        return {
            "prompt_speed_tps": prompt_speed_tps,
            "gen_speed_tps": gen_speed_tps,
            "vram_warning": vram_warning,
            "required_memory_gb": round(required_memory_gb, 1),
            "offloaded_to_ram_gb": round(offloaded_to_ram_gb, 1),
        }


# Component definitions
motherboards = [
    # Motherboard("4 RAM channels + 4 PCIe", 1100, 4, 4),
    Motherboard("8 RAM slots + 7 PCIe", 1100, 8, 7),
    Motherboard("12 RAM slots + 4 PCIe", 800, 12, 4),
]
# cpu = CPU('Threadripper 3970X', 700, 0.15)
cpu = CPU("Epyc 9534", 1100, 0.32)

ram_stick = RAM("64GB DDR5", 300, 64, 40)
gpu_card = GPU("RTX 3090", 850, 32, 850, 32)
# gpu_card = GPU("RTX 5090", 2400, 32, 1790, 142)
llm = LLM(
    680,
    35,
    15,
    18.83 / 131072,
    16.37 / 2 / 131072,
    18432,  # 7168,
    5,
    0.7,
)

# Active parameters in VRAM
active_in_vram = 15  # billion params

# Evaluate both configurations
print("LLM Performance Comparison\n" + "=" * 50)

CTX_SIZE = 32768
for motherboard in motherboards:
    build = PCBuild(motherboard, cpu)
    build.fill(ram_stick, gpu_card)

    perf = build.performance(llm, CTX_SIZE)

    print(f"\n{motherboard.name}:")
    print(f"  Total Price: ${build.price()}")
    print(f"  Token Gen Speed: {perf['gen_speed_tps']:.1f} tokens/sec")
    print(f"  Prompt Speed: {perf['prompt_speed_tps']} tokens/sec")
    print(f"  GPUs: {perf['components']['gpu_count']}x{gpu_card.vram_gb}GB")
    print(
        f"  VRAM: {perf['components']['total_vram_gb']}GB (Requires {perf['components']['vram_required']}GB)"
    )
    if perf["vram_warning"]:
        print("  " + perf["vram_warning"])
    else:
        print(f"Would take:")
        for prompt_length, generation_length in (
            (10000, 1000),
            (1000, 1000),
            (1000, 100),
            (1000, 10000),
            (100, 1000),
        ):
            t = (
                prompt_length / perf["prompt_speed_tps"]
                + generation_length / perf["gen_speed_tps"]
            )
            print(
                f"  {t:0.2f} seconds for {prompt_length} sequence and generate {generation_length} ."
            )
