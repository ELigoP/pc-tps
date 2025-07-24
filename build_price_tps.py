def calculate_llm_performance(
    motherboard_price,
    motherboard_ram_slots,
    ram_stick_speed,    # GB/s per stick
    ram_stick_price,
    motherboard_pcie_slots,
    gpu_vram,           # GB per GPU
    gpu_price,
    # Model parameters (with sensible defaults)
    total_parameters=680,          # in billions
    active_parameters_per_token=35, # in billions
    active_in_vram=15,              # in billions
    context_length=11000,           # tokens
    bits_per_weight=5,
    efficiency_factor=0.5           # real-world efficiency
):
    """
    Calculates LLM system performance metrics and costs.
    
    Returns: {
        "total_price": float,
        "ram_bandwidth": float,    # GB/s
        "token_gen_speed": float,   # tokens/sec
        "prompt_speed": float,      # tokens/sec
        "vram_warning": str or None
    }
    """
    # Calculate RAM configuration
    total_ram_bandwidth = motherboard_ram_slots * ram_stick_speed
    ram_cost = motherboard_ram_slots * ram_stick_price
    
    # Calculate GPU configuration
    num_gpus = min(motherboard_pcie_slots, 8)  # Practical limit
    gpu_cost = num_gpus * gpu_price
    total_vram = num_gpus * gpu_vram
    
    # Cost calculations
    total_price = motherboard_price + ram_cost + gpu_cost
    
    # Performance calculations
    active_in_ram = active_parameters_per_token - active_in_vram
    bytes_per_token = (active_in_ram * 1e9 * bits_per_weight) / 8
    
    # Token generation speed (GB/s bandwidth → tokens/sec)
    theoretical_token_speed = total_ram_bandwidth / (bytes_per_token / 1e9)
    real_token_speed = theoretical_token_speed * efficiency_factor
    
    # Prompt processing speed (GPU-bound estimate)
    prompt_speed = 50 * num_gpus  # tokens/sec
    
    # VRAM sufficiency check
    kv_cache_per_token = 0.005  # GB (for 680B model)
    kv_cache_total = context_length * kv_cache_per_token
    weights_vram = (active_in_vram * 1e9 * bits_per_weight) / (8 * 1e9)  # GB
    vram_required = kv_cache_total + weights_vram
    
    vram_warning = None
    if vram_required > total_vram:
        deficit = vram_required - total_vram
        vram_warning = (f"⚠️ Insufficient VRAM! Need {vram_required:.1f}GB, "
                       f"only {total_vram}GB available. Deficit: {deficit:.1f}GB")

    return {
        "total_price": total_price,
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

# Example usage for both motherboard configurations
if __name__ == "__main__":
    # Common parameters
    params = {
        "ram_stick_speed": 40,     # GB/s
        "ram_stick_price": 300,     # $ (64GB stick)
        "gpu_vram": 24,             # GB
        "gpu_price": 1000,          # $
        "active_in_vram": 15,       # billion params
    }
    
    # Option 1: 8 RAM slots + 7 PCIe slots
    opt1 = calculate_llm_performance(
        motherboard_price=1100,
        motherboard_ram_slots=8,
        motherboard_pcie_slots=7,
        **params
    )
    
    # Option 2: 12 RAM slots + 4 PCIe slots
    opt2 = calculate_llm_performance(
        motherboard_price=800,
        motherboard_ram_slots=12,
        motherboard_pcie_slots=4,
        **params
    )

    # Print results
    print("Option 1 (8 RAM slots + 7 PCIe):")
    print(f"  Total Price: ${opt1['total_price']}")
    print(f"  RAM Bandwidth: {opt1['ram_bandwidth']} GB/s")
    print(f"  Token Gen Speed: {opt1['token_gen_speed']:.1f} tokens/sec")
    print(f"  Prompt Speed: {opt1['prompt_speed']} tokens/sec")
    print(f"  GPUs: {opt1['components']['gpu_count']}x{params['gpu_vram']}GB")
    print(f"  VRAM: {opt1['components']['total_vram']}GB (Requires {opt1['components']['vram_required']}GB)")
    if opt1['vram_warning']:
        print("  " + opt1['vram_warning'])
    
    print("\nOption 2 (12 RAM slots + 4 PCIe):")
    print(f"  Total Price: ${opt2['total_price']}")
    print(f"  RAM Bandwidth: {opt2['ram_bandwidth']} GB/s")
    print(f"  Token Gen Speed: {opt2['token_gen_speed']:.1f} tokens/sec")
    print(f"  Prompt Speed: {opt2['prompt_speed']} tokens/sec")
    print(f"  GPUs: {opt2['components']['gpu_count']}x{params['gpu_vram']}GB")
    print(f"  VRAM: {opt2['components']['total_vram']}GB (Requires {opt2['components']['vram_required']}GB)")
    if opt2['vram_warning']:
        print("  " + opt2['vram_warning'])
