import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.tdm import borealis_gbs, get_mode_indices
import numpy as np
import datetime as dt

if(dt.date.today().weekday() == 4 or 5 or 6):
    prog = gbs_tdm()

    compile_options = {
        "device": device,
        "realistic_loss": True,
    }

    run_options = {
        "shots": None,
        "crop": True,
        "space_unroll": True,
    }

    eng_sim = sf.Engine(backend="gaussian")
    results_sim = eng_sim.run(prog, **run_options, compile_options=compile_options)
    print(results_sim.state.cov())



else:
    prog = gbs_tdm()

    shots = 10_000
    results = eng.run(prog, shots=shots, crop=True)
    print(results.samples)

def gbs_tdm():
    eng = sf.RemoteEngine("borealis")
    device = eng.device

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