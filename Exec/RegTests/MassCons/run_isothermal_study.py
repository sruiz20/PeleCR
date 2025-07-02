import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import erf

npoints = [16, 32, 64, 128, 256]

# for grid in grid:
alpha = 7.5362000690846e00

pltfile = "plt00400"

futils_path = "../../../Build/Submodules/PelePhysics/Submodules/amrex/Tools/Plotfile"
fextract = f"{futils_path}/amrex_fextract"

error = {}
for ngrid in npoints:
    error[ngrid] = {}
    os.system("rm -r plt*0")
    os.system(
        f"mpirun -n 8 ./PeleC3d.llvm.MPI.ex masscons-isothermal.inp amr.n_cell={ngrid} {ngrid} {ngrid}"
    )
    for idir in ["0", "1"]:
        os.system(f"{fextract} -d {idir} -s slice{idir} {pltfile}")
        fname = f"slice{idir}"

        with open(fname, "r") as f:
            junk = f.readline()
            timetext = f.readline()
            time = float(timetext.split()[-1])

        # remove first 2 lines, and # sign on third line
        with open(fname, "r") as f:
            lines = f.readlines()
        lines = [lines[2].lstrip("#").lstrip()] + lines[3:]
        with open(fname, "w") as f:
            f.writelines(lines)

        data = pd.read_csv(fname, sep="\\s+")

        coord = {"0": "x", "1": "y", "2": "z"}[idir]

        data["zeta1"] = (data[coord] + 0.5) / np.sqrt(4 * alpha * time)
        data["zeta2"] = (0.5 - data[coord]) / np.sqrt(4 * alpha * time)

        dT = {"0": 100, "1": 50, "2": 30}[idir]

        data["Tdelta2"] = -dT * erf(data["zeta2"])
        data["Tdelta1"] = dT * erf(data["zeta1"])
        data["Ttrue"] = 700.0 + data["Tdelta2"] + data["Tdelta1"]

        plt.figure(idir)
        plt.plot(data[coord], data["Temp"], "-", label=ngrid)
        if ngrid == npoints[-1]:
            plt.plot(data[coord], data["Ttrue"], "k:", label="analytical")
            plt.legend(frameon=False)
            plt.xlabel(coord + " (cm)")
            plt.ylabel("T (K)")
        error[ngrid][idir + "max"] = np.max(
            np.abs(data["Ttrue"] - data["Temp"])
        )  # np.linalg.norm( data['Ttrue'  ] - data['Temp'], ord=2) / np.sqrt(len(data[coord]))
        error[ngrid][idir + "mse"] = np.linalg.norm(
            data["Ttrue"] - data["Temp"], ord=2
        ) / np.sqrt(len(data[coord]))
        plt.savefig("figure_dir" + idir + ".png")

error = pd.DataFrame(error).T
print(error)
plt.figure()
plt.loglog(error.index, error["0mse"], "r-o", label="x-direction")
plt.loglog(error.index, error["1mse"], "b-o", label="y-direction")
plt.loglog(error.index, 100 * (1.0 / error.index) ** 2, "k:", label="2nd order")
plt.ylabel("Error")
plt.xlabel("N")
plt.xlim([8, 512])
plt.legend(frameon=False)
plt.savefig("figure_convergence.png")
