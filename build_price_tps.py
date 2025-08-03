from dataclasses import dataclass
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Component and Model Definitions ---


@dataclass
class Buyable:
    name: str
    price: float

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.name == other.name


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
        return f"({self.motherboard.name}) {len(self.gpus)}x{self.gpus[0].vram_gb}GB GPUs, {len(self.ram_sticks):>2}x{self.ram_sticks[0].capacity_gb}GB RAM"

    def fill(self, ram: RAM, gpu: GPU, n_ram=None, n_gpu=None):
        """Fill available slots with specified components"""
        if not self.motherboard:
            raise ValueError("Motherboard must be selected first")
        if n_ram is None:
            n_ram = self.motherboard.ram_slots
        else:
            n_ram = min(n_ram, self.motherboard.ram_slots)
        self.ram_sticks = [ram for _ in range(n_ram)]
        if n_gpu is None:
            n_gpu = self.motherboard.pcie_slots
        else:
            n_gpu = min(self.motherboard.pcie_slots, n_gpu)
        self.gpus = [gpu for _ in range(n_gpu)]
        return self

    def price(self, *components) -> float:
        """Calculate total build price"""
        if not self.motherboard or not self.cpu:
            return 0.0
        sticks_to_buy = len(self.ram_sticks)
        gpus_to_buy = len(self.gpus)
        mobos_to_buy = 1
        cpus_to_buy = 1
        for components_list in components:
            for component in components_list:
                if (component == self.ram_sticks[0]) and sticks_to_buy:
                    sticks_to_buy -= 1
                if (component == self.gpus[0]) and gpus_to_buy:
                    gpus_to_buy -= 1
                if (component == self.cpu) and cpus_to_buy:
                    cpus_to_buy -= 1
                if (component == self.motherboard) and mobos_to_buy:
                    mobos_to_buy -= 1
        motherboard_cost = mobos_to_buy * self.motherboard.price
        cpu_cost = cpus_to_buy * self.cpu.price
        ram_cost = sticks_to_buy * self.ram_sticks[0].price
        gpu_cost = gpus_to_buy * self.gpus[0].price
        # print(
        #     motherboard_cost,
        #     cpu_cost,
        #     ram_cost,
        #     gpu_cost,
        # )

        return motherboard_cost + cpu_cost + ram_cost + gpu_cost

    def performance(
        self,
        llm: LLM,
        context_length: int,
        kv_cache_bpw=1,  # Default to 8-bit (1 byte) for KV cache
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
        # gpu_mem_bw_total = num_gpus * gpu.memory_bw_gbps * eff
        cpu_tflops = self.cpu.tflops * eff
        ram_bw_total = len(self.ram_sticks) * ram.memory_bw_per_module_gbps

        pcie_bw_gbps = {5: 64, 4: 32, 3: 16}.get(
            min(self.motherboard.pcie_gen, gpu.pcie_gen)
        ) * eff

        vram_total_bytes = num_gpus * gpu.vram_gb * 1e9
        ram_total_bytes = len(self.ram_sticks) * ram.capacity_gb * 1e9

        gpu_flops_single_per_s = gpu_tflops_single * 1e12
        gpu_flops_total_per_s = gpu_tflops_total * 1e12
        cpu_flops_per_s = cpu_tflops * 1e12

        gpu_bw_bytes_per_s = gpu.memory_bw_gbps * 1e9 * eff
        pcie_bw_bytes_per_s = pcie_bw_gbps * 1e9
        ram_bw_bytes_per_s = ram_bw_total * 1e9

        # LLM parameters
        bytes_per_weight = llm.bits_per_weight / 8.0
        P_bytes = llm.total_parameters * 1e9 * bytes_per_weight
        A_bytes = llm.active_parameters_per_token * 1e9 * bytes_per_weight
        AA_bytes = llm.always_active_parameters_per_token * 1e9 * bytes_per_weight
        P_exp_bytes = P_bytes - AA_bytes
        kv_cache_bytes = 2 * llm.layers * llm.hidden_dim * context_length * kv_cache_bpw

        # --- 2. Layer Placement Logic ---
        kv_on_gpu = min(kv_cache_bytes, vram_total_bytes)
        vram_rem = vram_total_bytes - kv_on_gpu

        aa_on_gpu = min(AA_bytes, vram_rem)
        aa_on_ram = AA_bytes - aa_on_gpu
        vram_rem -= aa_on_gpu
        model_on_vram_bytes = min(vram_total_bytes, P_bytes + kv_cache_bytes)
        if P_bytes + kv_cache_bytes <= vram_total_bytes:
            p_exp_on_gpu = P_exp_bytes
            p_exp_on_ram = 0
            model_on_ram_bytes = 0
        else:
            p_exp_on_gpu = min(P_exp_bytes, vram_rem)
            p_exp_on_ram = P_exp_bytes - p_exp_on_gpu
            model_on_ram_bytes = aa_on_ram + p_exp_on_ram

        alpha_gpu = p_exp_on_gpu / P_exp_bytes if P_exp_bytes > 0 else 0
        A_gpu_bytes = aa_on_gpu + (A_bytes - AA_bytes) * alpha_gpu
        A_ram_bytes = aa_on_ram + (A_bytes - AA_bytes) * (1 - alpha_gpu)
        if model_on_ram_bytes >= ram_total_bytes:
            return {
                "total_vram_gb": vram_total_bytes / 1e9,
                "total_ram_gb": ram_total_bytes / 1e9,
                "kv_cache_size_gb": kv_cache_bytes / 1e9,
                "model_on_ram_gb": model_on_ram_bytes / 1e9,
                "model_on_vram_gb": model_on_vram_bytes / 1e9,
                "active_params_on_ram_gb": A_ram_bytes / 1e9,
                "ttft_s": float("inf"),
                "pp_tps": 0,
                "tpot_s": 0,
                "tg_tps": 0,
                "breakdown_ttft_ms": {
                    "compute": float("inf"),
                    "memory": float("inf"),
                    "communication": float("inf"),
                },
                "breakdown_tpot_ms": {
                    "compute": float("inf"),
                    "memory_vram": float("inf"),
                    "memory_pcie_or_ram": float("inf"),
                    "communication": float("inf"),
                },
            }

        fraction_on_ram = A_ram_bytes / A_bytes if A_bytes > 0 else 0

        # --- 3. Prefill Phase (TTFT) Calculation ---

        # Total FLOPs for the prefill phase
        flops_parametric = 2 * llm.active_parameters_per_token * 1e9 * context_length
        flops_attention = 2 * llm.layers * llm.hidden_dim * (context_length**2)
        total_prefill_flops = flops_parametric + flops_attention

        # Distribute FLOPs based on parameter location
        prefill_flops_gpu = total_prefill_flops * (1 - fraction_on_ram)
        prefill_flops_ram = total_prefill_flops * fraction_on_ram

        # Time for work on GPU-resident parameters
        t_compute_gpu_prefill = prefill_flops_gpu / gpu_flops_single_per_s
        t_memory_gpu_prefill = A_gpu_bytes / gpu_bw_bytes_per_s  # *context_length
        t_gpu_work_prefill = t_compute_gpu_prefill + t_memory_gpu_prefill

        # Time for work on RAM-resident parameters (choose faster path: CPU or GPU)
        t_ram_work_prefill = 0
        t_compute_ram_prefill = 0
        t_memory_ram_prefill = 0
        if A_ram_bytes > 0:
            # Path A: Transfer to GPU, then compute
            t_ram_via_gpu = (prefill_flops_ram / gpu_flops_single_per_s) + (
                context_length * (A_ram_bytes / pcie_bw_bytes_per_s)
            )
            # Path B: Compute on CPU
            t_ram_via_cpu = (prefill_flops_ram / cpu_flops_per_s) + (
                context_length * (A_ram_bytes / ram_bw_bytes_per_s)
            )

            if t_ram_via_cpu < t_ram_via_gpu:
                t_ram_work_prefill = t_ram_via_cpu
                t_compute_ram_prefill = prefill_flops_ram / cpu_flops_per_s
                t_memory_ram_prefill = context_length * (
                    A_ram_bytes / ram_bw_bytes_per_s
                )
            else:
                t_ram_work_prefill = t_ram_via_gpu
                t_compute_ram_prefill = prefill_flops_ram / gpu_flops_single_per_s
                t_memory_ram_prefill = context_length * (
                    A_ram_bytes / pcie_bw_bytes_per_s
                )

        # Inter-GPU communication time (pipeline parallelism)
        activation_size_bytes = llm.hidden_dim * context_length * bytes_per_weight
        t_comm_prefill = (
            (num_gpus - 1) * activation_size_bytes / pcie_bw_bytes_per_s
            if num_gpus > 1
            else 0
        )

        # ttft = max(t_gpu_work_prefill, t_ram_work_prefill) + t_comm_prefill
        ttft = t_gpu_work_prefill + t_ram_work_prefill + t_comm_prefill

        # --- 4. Decode Phase (TPOT) Calculation ---

        total_decode_flops = 2 * llm.active_parameters_per_token * 1e9
        decode_flops_gpu = total_decode_flops * (1 - fraction_on_ram)
        decode_flops_ram = total_decode_flops * fraction_on_ram

        # Time for work on GPU-resident parameters (including KV cache read)
        t_compute_gpu_decode = decode_flops_gpu / gpu_flops_total_per_s
        t_memory_gpu_decode = (A_gpu_bytes + kv_on_gpu) / gpu_bw_bytes_per_s
        t_gpu_work_decode = t_compute_gpu_decode + t_memory_gpu_decode

        # Time for work on RAM-resident parameters (choose faster path)
        t_ram_work_decode = 0
        t_compute_ram_decode = 0
        t_memory_ram_pcie_decode = 0
        if A_ram_bytes > 0:
            # Path A: GPU
            t_ram_via_gpu = (decode_flops_ram / gpu_flops_total_per_s) + (
                A_ram_bytes / pcie_bw_bytes_per_s
            )
            # Path B: CPU
            t_ram_via_cpu = (decode_flops_ram / cpu_flops_per_s) + (
                A_ram_bytes / ram_bw_bytes_per_s
            )

            if t_ram_via_cpu < t_ram_via_gpu:
                t_ram_work_decode = t_ram_via_cpu
                t_compute_ram_decode = decode_flops_ram / cpu_flops_per_s
                t_memory_ram_pcie_decode = A_ram_bytes / ram_bw_bytes_per_s
            else:
                t_ram_work_decode = t_ram_via_gpu
                t_compute_ram_decode = decode_flops_ram / gpu_flops_total_per_s
                t_memory_ram_pcie_decode = A_ram_bytes / pcie_bw_bytes_per_s

        # Inter-GPU communication
        activation_size_bytes_decode = llm.hidden_dim * 1 * bytes_per_weight
        t_comm_decode = (
            (num_gpus - 1) * activation_size_bytes_decode / pcie_bw_bytes_per_s
            if num_gpus > 1
            else 0
        )

        tpot = t_gpu_work_decode + t_ram_work_decode + t_comm_decode
        tokens_per_sec = 1.0 / tpot

        return {
            "total_vram_gb": vram_total_bytes / 1e9,
            "total_ram_gb": ram_total_bytes / 1e9,
            "kv_cache_size_gb": kv_cache_bytes / 1e9,
            "model_on_ram_gb": model_on_ram_bytes / 1e9,
            "model_on_vram_gb": model_on_vram_bytes / 1e9,
            "active_params_on_ram_gb": A_ram_bytes / 1e9,
            "ttft_s": ttft,
            "pp_tps": context_length / ttft,
            "tpot_s": tpot,
            "tg_tps": tokens_per_sec,
            "breakdown_ttft_ms": {
                "compute": (t_compute_gpu_prefill + t_compute_ram_prefill) * 1000,
                "memory": (t_memory_gpu_prefill + t_memory_ram_prefill) * 1000,
                "communication": t_comm_prefill * 1000,
            },
            "breakdown_tpot_ms": {
                "compute": (t_compute_gpu_decode + t_compute_ram_decode) * 1000,
                "memory_vram": t_memory_gpu_decode * 1000,
                "memory_pcie_or_ram": t_memory_ram_pcie_decode * 1000,
                "communication": t_comm_decode * 1000,
            },
        }


# --- Main Execution Block ---

if __name__ == "__main__":
    # Define Components
    rtx_3090 = GPU(
        name="RTX 3090",
        price=1100,
        vram_gb=24,
        memory_bw_gbps=930,
        tflops=35,
        pcie_gen=4,
    )
    tp_3970 = CPU(name="Threadripper 3970X", price=650, tflops=0.15)
    epyc_9543 = CPU(name="Epyc 9543", price=900, tflops=0.3)
    ddr5_ecc = RAM(
        name="DDR5 ECC 64GB",
        price=300,
        capacity_gb=64,
        memory_bw_per_module_gbps=35,
    )
    ddr5_ecc_small = RAM(
        name="DDR5 ECC 16GB",
        price=80,
        capacity_gb=16,
        memory_bw_per_module_gbps=35,
    )
    ddr4_non_ecc = RAM(
        name="DDR4 Non-ECC 2x32GB",
        price=200,
        capacity_gb=64,
        memory_bw_per_module_gbps=20,
    )

    old_mb_4_gpu = Motherboard(
        name="Old 4 PCIe MB", price=350, ram_slots=4, pcie_slots=4, pcie_gen=3
    )
    mb_7_gpu = Motherboard(
        name="7 PCIe MB", price=1400, ram_slots=8, pcie_slots=7, pcie_gen=5
    )
    mb_4_gpu = Motherboard(
        name="4 PCIe MB", price=1000, ram_slots=12, pcie_slots=4, pcie_gen=5
    )

    # Define LLMs
    qwen3_235b_params = dict(
        name="Qwen3 235B-A22B",
        total_parameters=235,
        active_parameters_per_token=22,
        always_active_parameters_per_token=7.8,  # Derived
        layers=94,
        hidden_dim=4096,
    )

    deepseek_r1_params = dict(
        name="DS 671B R1",
        total_parameters=671,
        active_parameters_per_token=37,
        always_active_parameters_per_token=16.55,  # Derived
        layers=61,
        hidden_dim=7168,
    )
    models: List[LLM] = []
    for _model_params in [
        qwen3_235b_params,
        deepseek_r1_params,
    ]:
        for bpw in (
            # 2.6,
            # 2.9,
            3.5,
            5,
            # 8,
        ):
            # if _model_params["name"].startswith("Qwen") and bpw < 3:
            #     continue
            model_params = _model_params.copy()
            model_params["name"] = f"{_model_params['name']} {bpw:.1f}b"
            models.append(
                LLM(
                    **model_params,
                    bits_per_weight=bpw,
                )
            )

    # Create PC Builds
    builds: List[PCBuild] = []
    for motherboard, cpu, ram, gpu, n_ram, n_gpu in (
        (old_mb_4_gpu, tp_3970, ddr4_non_ecc, rtx_3090, 3, None),
        # (old_mb_4_gpu, tp_3970, ddr4_non_ecc, rtx_3090, 4, None),
        #####
        # (mb_4_gpu, epyc_9543, ddr5_ecc, rtx_3090, 3, None),
        # (mb_7_gpu, epyc_9543, ddr5_ecc, rtx_3090, 2, 5),
        #####
        # (mb_4_gpu, epyc_9543, ddr5_ecc_small, rtx_3090, None, None),
        # (mb_7_gpu, epyc_9543, ddr5_ecc_small, rtx_3090, None, 5),
        #####
        (mb_4_gpu, epyc_9543, ddr5_ecc, rtx_3090, 7, None),
        (mb_7_gpu, epyc_9543, ddr5_ecc, rtx_3090, 5, 5),
        #####
        # (mb_4_gpu, epyc_9543, ddr5_ecc, rtx_3090, None, None),
        # (mb_7_gpu, epyc_9543, ddr5_ecc, rtx_3090, None, 5),
        # (mb_7_gpu, epyc_9543, ddr5_ecc, rtx_3090, None, None),
    ):
        builds.append(
            PCBuild(
                motherboard,
                cpu,
            ).fill(
                ram,
                gpu,
                n_ram=n_ram,
                n_gpu=n_gpu,
            )
        )

    # --- Performance Analysis and Data Collection ---
    all_results = []

    for model in models:
        for build in builds:
            for context_length in [2**pow for pow in range(5, 16)]:
                results = build.performance(
                    llm=model, context_length=context_length, kv_cache_bpw=2
                )

                # Store results for plotting
                all_results.append(
                    {
                        "Build": str(build),
                        "LLM": model.name,
                        "CTX": context_length,
                        "PP (tps)": results["pp_tps"],
                        "TG (tps)": results["tg_tps"],
                    }
                )

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # --- Plotting ---

    # Create a directory for plots if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    unique_models = df["LLM"].unique()

    for model_name in unique_models:
        print(f"Generating plot for {model_name}...")

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()  # Create a second y-axis

        model_df = df[df["LLM"] == model_name].copy()

        # Replace 'OOM' with NaN so it's not plotted
        model_df.replace("OOM", np.nan, inplace=True)
        model_df = model_df.dropna()

        build_names = model_df["Build"].unique()
        colors = plt.cm.get_cmap("tab10", len(build_names))

        for i, build_name in enumerate(build_names):
            build_df = model_df[model_df["Build"] == build_name]

            # Plot Prompt Processing (PP) on the left axis
            ax1.plot(
                build_df["CTX"],
                build_df["PP (tps)"],
                label=f"{build_name} - PP",
                color=colors(i),
                linestyle="-",
            )

            # Plot Token Generation (TG) on the right axis
            ax2.plot(
                build_df["CTX"],
                build_df["TG (tps)"],
                label=f"{build_name} - TG",
                color=colors(i),
                linestyle="--",
            )

        # Formatting the plot
        ax1.set_xlabel("Context Length (CTX)")
        ax1.set_ylabel("Prompt Processing (tokens/s)", color="blue")
        ax2.set_ylabel("Token Generation (tokens/s)", color="red")

        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")

        ax1.set_xscale("log")  # Log scale for context length is often useful

        plt.title(f"LLM Performance vs. Context Length for {model_name}")

        # Create a single legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.tight_layout()

        # Sanitize filename
        safe_model_name = (
            model_name.replace(" ", "_").replace("/", "_").replace(".", "")
        )
        plot_filename = os.path.join("plots", f"{safe_model_name}_performance.png")
        plt.savefig(plot_filename)
