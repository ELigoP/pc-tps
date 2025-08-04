# LLM performance estimator

This is a script to estimate performance of LLM model (prompt processing and token generation) on a specified PC build, and also to calculate the price (not counting PSUs, case, disks etc.).

## How to

Enter parameters of your components. All PC components have name and price.

Other parameters of your PC components:

- motherboard (PCIe slots number, RAM slots number, PCIe generation)
- CPU (flops)
- RAM modules (capacity, memory bandwidth)
- GPU (capacity, memory bandwidth, flops, PCIe generation)

Also enter model parameters for which you want to get performance estimate. There are presets for the DeepSeek R1 and Qwen3 235B A22B.

Then run `python build_price_tps.py`. You will get plots as pictures in `plots` dir, similar to

![Prompt Processing and Token Generation performance plot for DeepSeek R1model quantized to 2.9 bits per weight](/ELigoP/pc-tps/img/DS_671B_R1_2.9b_tps.png)

![Prompt Processing and Token Generation performance plot for Qwen3 235B A22B model quantized to 5 bits per weight](/ELigoP/pc-tps/img/Qwen3_235B-A22B_5.0b_tps.png)
