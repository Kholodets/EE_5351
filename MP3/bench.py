import subprocess

for size in [10, 100, 1000, 2000, 5000, 7500, 10000, 20000, 25000]:
    subprocess.run(["./2Dconvolution", str(size)])
