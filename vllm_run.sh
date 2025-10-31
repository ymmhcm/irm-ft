export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

vllm serve Llama3-8B \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 8000