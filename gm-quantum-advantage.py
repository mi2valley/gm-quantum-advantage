import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.tdm import borealis_gbs, get_mode_indices
import xcc
import numpy as np

connection = xcc.Connection.load()
borealis = xcc.Device(target="borealis", connection=connection)
eng = sf.RemoteEngine("borealis")
device = eng.device
print(borealis.status)

def gbs_tdm():
    gate_args_list = borealis_gbs(device, modes=216, squeezing="high")
    delays = [1, 6, 36]
    n, N = get_mode_indices(delays)

    from strawberryfields.ops import Sgate, Rgate, BSgate, MeasureFock

    prog = sf.TDMProgram(N)

    with prog.context(*gate_args_list) as (p, q):
        Sgate(p[0]) | q[n[0]]
        for i in range(len(delays)):
            Rgate(p[2 * i + 1]) | q[n[i]]
            BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
        MeasureFock() | q[0]

    return prog

if(borealis.status == "online"):
    prog = gbs_tdm()

    shots = 10_000
    results = eng.run(prog, shots=shots, crop=True)
    print(np.cov(samples[:, 0, :].T))
    print(results.state.cov())
    print(results.samples)

else:
    prog = gbs_tdm()

    run_options = {
        "shots": None,
        "crop": True,
        "space_unroll": True,
    }

    compile_options = {
        "device": device,
        "realistic_loss": True,
    }

    eng_sim = sf.Engine(backend="gaussian")
    results_sim = eng_sim.run(prog, **run_options, compile_options=compile_options)
    print(results_sim.state.cov())
