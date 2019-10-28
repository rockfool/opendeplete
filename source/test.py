import results
import zernike
import numpy as np

order = 10

rings = 20
wedges = 32

for i in range(31):
    r = results.read_results("./vera_1i_fet/step" + str(i) + ".pklz")

    con = r.num[0]

    #print(r.k)

    zer = zernike.ZernikePolynomial(order, con["10000", "Xe-135"]  * rings * wedges / (np.pi * 0.4096**2) / np.pi)

    print(zer.coeffs[0] / (rings * wedges) * np.pi)

    #zer.force_positive()

    # zer.plot_disk(rings, wedges, "testg" + str(i+1) + ".pdf")

    rea = r.rates[0]

    zer = rea.get_fet(["10000", "Xe-135", "(n,gamma)"])  * rings * wedges / (np.pi * 0.4096**2) / np.pi * 1.0e24

    #zer.plot_disk(rings, wedges, "testgr" + str(i+1) + ".pdf")
