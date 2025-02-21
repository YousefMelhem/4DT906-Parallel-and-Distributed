import pandas as pd
import plotly.express as px

# Metrics and logs (as provided)
metrics = ["Name", "GFlop/s", "block_size", "Matrix Size", "Threads", "Other"]
logs = [
    ["Python", "0.014171", "1", "256", "1", ""],
    ["Numpy", "29.757932", "1", "1024", "1", ""],
    ["Numpy", "152.431957", "1", "1024", "1", ""],
    ["Numpy32byte", "452.396411", "1", "2048", "1", "np.float32"],
    ["cpp", "2.89", "1", "1024", "1", ""],
    ["cpp", "36.415234", "1", "1024", "1", "Transposed"],
    ["cpp", "36.415234", "1", "1024", "1", "Transposed"],
    ["cpp", "33.664894", "4", "1024", "1", "Transposed"],
    ["cpp", "46.528656", "8", "1024", "1", "Transposed"],
    ["cpp", "62.039108", "16", "1024", "1", "Transposed"],
    ["cpp", "31.578785", "32", "1024", "1", "Transposed"],
    ["cpp", "5.150805", "64", "1024", "1", "Transposed"],
    ["cpp", "203.244720", "16", "1024", "omp", "Transposed"],
    ["cpp", "220.164398", "16", "1024", "omp", "Transposed, Unrolled"],
    ["Accelerate", "1365.177979", "16", "8192", "Max?", "vDSP_mtrans"],
]
