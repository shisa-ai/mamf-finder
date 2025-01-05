# mamf-finder

Results and helper scripts for testing a GPU node with @stas00 's `mamf-finder`.

You can get the latest version of that test:
```
wget https://raw.githubusercontent.com/stas00/ml-engineering/refs/heads/master/compute/accelerator/benchmarks/mamf-finder.py
```
And for more details/results, see: https://github.com/stas00/ml-engineering/tree/master/compute/accelerator


## ml.p5.48xlarge

Here's the results tested on a standard AWS H100 node:

## Results for bfloat16
| GPU# | Accelerator | MAMF | Theory | Efficiency | Best Shape MxNxK | torch ver | Notes |
|------|-------------|------|---------|------------|------------------|-----------|--------|
| 0 | H100 SXM | 785.5 | 989 | 79.4% | 4096x3072x13312 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 1 | H100 SXM | 777.7 | 989 | 78.6% | 2048x2048x12288 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 2 | H100 SXM | 765.5 | 989 | 77.4% | 10240x2048x11264 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 3 | H100 SXM | 758.6 | 989 | 76.7% | 2048x2048x15360 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 4 | H100 SXM | 766.0 | 989 | 77.5% | 2048x4096x11264 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 5 | H100 SXM | 766.1 | 989 | 77.5% | 6144x2048x14336 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 6 | H100 SXM | 766.4 | 989 | 77.5% | 2048x2048x15360 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 7 | H100 SXM | 767.0 | 989 | 77.6% | 5120x3072x13312 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |

## Results for float8_e4m3fn
| GPU# | Accelerator | MAMF | Theory | Efficiency | Best Shape MxNxK | torch ver | Notes |
|------|-------------|------|---------|------------|------------------|-----------|--------|
| 0 | H100 SXM | 1344.9 | 1979 | 68.0% | 15360x3072x4096 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 1 | H100 SXM | 1331.8 | 1979 | 67.3% | 15360x3072x4096 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 2 | H100 SXM | 1319.9 | 1979 | 66.7% | 15360x3072x5120 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 3 | H100 SXM | 1315.5 | 1979 | 66.5% | 13312x3072x5120 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 4 | H100 SXM | 1315.4 | 1979 | 66.5% | 11264x3072x5120 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 5 | H100 SXM | 1329.8 | 1979 | 67.2% | 11264x3072x5120 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 6 | H100 SXM | 1328.0 | 1979 | 67.1% | 13312x3072x6144 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
| 7 | H100 SXM | 1313.6 | 1979 | 66.4% | 13312x3072x5120 | 2.5.1+cu121 | gpu3, ml.p5.48xlarge, benchmark v2 |
