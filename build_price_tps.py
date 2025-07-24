from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Motherboard:
    name: str
    price: float
    ram_slots: int
    pcie_slots: int

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

@dataclass
class LLM:
    total_parameters: float  # in billions
    active_parameters_per_token: float  # in billions
    active_parameters_per_token_in_vram: float  # in billions
    context_length: int  # tokens
    bits_per_weight: int = 5
    efficiency_factor: float = 0.5
    kv_cache_per_ctxsq: float = 8e-9  # GB

class PCBuild:
    def __init__(self, motherboard: Optional[Motherboard] = None):
        self.motherboard = motherboard
        self.ram_sticks: List[RAM] = []
        self.gpus: List[GPU] = []
    
    def select_motherboard(self, motherboard: Motherboard):
        self.motherboard = motherboard
        return self
    
    def fill(self, ram: RAM, gpu: GPU):
        """Fill available slots with specified components"""
        if not self.motherboard:
            raise ValueError("Motherboard must be selected first")
        
        # Fill RAM slots
        self.ram_sticks = [ram for _ in range(self.motherboard.ram_slots)]
        
        # Fill GPU slots (practical limit of 8 GPUs)
        num_gpus = min(self.motherboard.pcie_slots, 8)
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
    
    def performance(self, llm: LLM) -> dict:
        """Calculate LLM performance metrics"""
        if not self.motherboard:
            return {}
        
        # RAM bandwidth
        total_ram_bandwidth = sum(ram.bandwidth_per_stick for ram in self.ram_sticks)
        
        # GPU configuration
        num_gpus = len(self.gpus)
        total_vram = sum(gpu.vram_gb for gpu in self.gpus)
        
        # Token generation speed
        active_in_ram = llm.active_parameters_per_token - llm.active_parameters_per_token_in_vram
        bytes_per_token = (active_in_ram * 1e9 * llm.bits_per_weight) / 8
        theoretical_token_speed = total_ram_bandwidth / (bytes_per_token / 1e9)
        real_token_speed = theoretical_token_speed * llm.efficiency_factor
        
        # Prompt processing speed (GPU-bound estimate)
        prompt_speed = 50 * num_gpus
        
        # VRAM sufficiency check
        kv_cache_total = llm.context_length**2 * llm.kv_cache_per_ctxsq
        weights_vram = (llm.active_parameters_per_token_in_vram * 1e9 * llm.bits_per_weight) / (8 * 1e9)
        vram_required = kv_cache_total + weights_vram
        
        vram_warning = None
        if vram_required > total_vram:
            deficit = vram_required - total_vram
            vram_warning = (f"⚠️ Insufficient VRAM! Need {vram_required:.1f}GB, "
                          f"only {total_vram}GB available. Deficit: {deficit:.1f}GB")
        
        return {
            "ram_bandwidth": total_ram_bandwidth,
            "token_gen_speed": real_token_speed,
            "prompt_speed": prompt_speed,
            "vram_warning": vram_warning,
            "components": {
                "gpu_count": num_gpus,
                "total_vram": total_vram,
                "vram_required": round(vram_required, 1)
            }
        }
    
    def processing_time(self, llm, prompt_length: int, generation_length: int):
        perf = self.performance(llm)
        return prompt_length / perf['prompt_speed'] + generation_length / perf['token_gen_speed']


# Component definitions
motherboards = [
    Motherboard("8 RAM slots + 7 PCIe", 1100, 8, 7),
    Motherboard("12 RAM slots + 4 PCIe", 800, 12, 4)
]

ram_stick = RAM("64GB DDR5", 300, 64, 40)
gpu_card = GPU("32GB GPU", 1000, 32)
llm = LLM(680, 35, 15, 11000)

# Active parameters in VRAM
active_in_vram = 15  # billion params

# Evaluate both configurations
print("LLM Performance Comparison\n" + "="*50)

for motherboard in motherboards:
    build = PCBuild(motherboard)
    build.fill(ram_stick, gpu_card)
    
    perf = build.performance(llm)
    
    print(f"\n{motherboard.name}:")
    print(f"  Total Price: ${build.price()}")
    print(f"  RAM Bandwidth: {perf['ram_bandwidth']} GB/s")
    print(f"  Token Gen Speed: {perf['token_gen_speed']:.1f} tokens/sec")
    print(f"  Prompt Speed: {perf['prompt_speed']} tokens/sec")
    print(f"  GPUs: {perf['components']['gpu_count']}x{gpu_card.vram_gb}GB")
    print(f"  VRAM: {perf['components']['total_vram']}GB (Requires {perf['components']['vram_required']}GB)")
    if perf['vram_warning']:
        print("  " + perf['vram_warning'])
    else:
        for prompt_length, generation_length in ((10000, 1000), (1000,1000), (1000,100), (1000,10000), (100,1000)):
            t = build.processing_time(llm, prompt_length, generation_length)
            print(f'Would take {t:0.2f} seconds to process {prompt_length} token long sequence and give {generation_length} token long answer.')