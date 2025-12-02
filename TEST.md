# Running test_enhanced.py with FP32, FP64, and INT8

## Commands

### FP32 (default)
```bash
# CPU
python test_enhanced.py --device cpu --threads 8 --size 8192 --iters 50

# GPU
python test_enhanced.py --device cuda --size 8192 --iters 50
```

### FP64
```bash
# CPU
python test_enhanced.py --device cpu --dtype float64 --threads 8 --size 8192 --iters 50

# GPU
python test_enhanced.py --device cuda --dtype float64 --size 8192 --iters 50
```

### INT8
```bash
# CPU only (PyTorch INT8 quantization is CPU-only)
python test_enhanced.py --device cpu --dtype int8 --threads 8 --size 8192 --iters 50

# GPU INT8: use C++ comparison tool instead
chmod +x compare_int8.sh
./compare_int8.sh
```

## Summary

| Precision | Device | Command |
|-----------|--------|---------|
| FP32 | CPU | `python test_enhanced.py --device cpu --threads 8` |
| FP32 | GPU | `python test_enhanced.py --device cuda` |
| FP64 | CPU | `python test_enhanced.py --device cpu --dtype float64 --threads 8` |
| FP64 | GPU | `python test_enhanced.py --device cuda --dtype float64` |
| INT8 | CPU | `python test_enhanced.py --device cpu --dtype int8 --threads 8` |
| INT8 | GPU | `./compare_int8.sh` (C++ tool) |
