#!/usr/bin/env python3
"""numpy-only reference generator for the `feffdat` Rust port.

Replicates `larch/xafs/feffdat.py` `FeffDatFile._read` and the *linear*
(`interp='lin'`) branch of `FeffPathGroup._calc_chi`, using only numpy. The
emitted reference files are consumed by `crates/feffdat/tests/parity.rs` to
check the Rust parser + EXAFS equation to ~1e-10.

The linear branch is reproduced exactly (it is `numpy.interp`). The cubic
branch (larch's default, `scipy.interpolate.UnivariateSpline(s=0)`) is emitted
ONLY when scipy is importable; in a scipy-less environment the `chi_cubic`
column is omitted and cubic parity stays unverified.

Usage:
    python3 ref_chi.py <feff.dat> <out.txt> KEY=VAL ...
KEYs: degen s02 e0 ei deltar sigma2 third fourth  (omitted -> larch defaults)
"""
import sys
import numpy as np

KTOE = 3.8099819442818976
ETOK = 1.0 / KTOE
SMALL = 1.0e-6

try:
    from scipy.interpolate import UnivariateSpline
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def read_feffdat(filename):
    """Port of FeffDatFile._read (fields needed for chi + verification)."""
    with open(filename) as fh:
        lines = fh.readlines()
    out = {"potentials": [], "geom": [], "shell": "", "absorber": None}
    mode = "header"
    data = []
    pcounter = 0
    for iline, line in enumerate(lines, start=1):
        line = line[:-1].strip()
        if line.startswith("#"):
            line = line[1:]
        line = line.strip()
        if iline == 1:
            out["title"] = line[:64].strip()
            out["version"] = line[64:].strip()
            continue
        if line.startswith("k") and line.endswith("real[p]@#"):
            mode = "arrays"
            continue
        elif "----" in line[2:10]:
            mode = "path"
            continue
        if mode == "path":
            pcounter += 1
            if pcounter == 1:
                w = [float(x) for x in line.split()[:5]]
                out["nleg"] = int(w[0])
                out["degen"], out["reff"], out["rnorman"], out["edge"] = w[1:]
            elif pcounter > 2:
                words = line.split()
                if len(words) >= 5:
                    if out["absorber"] is None:
                        out["absorber"] = words[5] if len(words) > 5 else f"Z{words[4]}"
        elif mode == "arrays":
            d = [float(x) for x in line.split()]
            if len(d) == 7:
                data.append(d)
    data = np.array(data).transpose()
    out["k"] = data[0]
    out["real_phc"] = data[1]
    out["mag_feff"] = data[2]
    out["pha_feff"] = data[3]
    out["red_fact"] = data[4]
    out["lam"] = data[5]
    out["rep"] = data[6]
    out["pha"] = data[1] + data[3]
    out["amp"] = data[2] * data[4]
    return out


def build_k(fdat, kstep=0.05, kmax=None):
    if kmax is None:
        kmax = 30.0
    kmax = min(max(fdat["k"]), kmax)
    return kstep * np.arange(int(1.01 + kmax / kstep), dtype="float64")


def calc_chi(fdat, k, pp_, interp="lin"):
    reff = fdat["reff"]
    s02, e0, ei = pp_["s02"], pp_["e0"], pp_["ei"]
    deltar, sigma2 = pp_["deltar"], pp_["sigma2"]
    third, fourth, degen = pp_["third"], pp_["fourth"], pp_["degen"]

    en = k * k - e0 * ETOK
    if np.min(np.abs(en)) < SMALL:
        en[np.where(np.abs(en) < 1.5 * SMALL)] = SMALL
    q = np.sign(en) * np.sqrt(np.abs(en))

    if interp == "lin":
        pha = np.interp(q, fdat["k"], fdat["pha"])
        amp = np.interp(q, fdat["k"], fdat["amp"])
        rep = np.interp(q, fdat["k"], fdat["rep"])
        lam = np.interp(q, fdat["k"], fdat["lam"])
    else:
        pha = UnivariateSpline(fdat["k"], fdat["pha"], s=0)(q)
        amp = UnivariateSpline(fdat["k"], fdat["amp"], s=0)(q)
        rep = UnivariateSpline(fdat["k"], fdat["rep"], s=0)(q)
        lam = UnivariateSpline(fdat["k"], fdat["lam"], s=0)(q)

    pp = (rep + 1j / lam) ** 2 + 1j * ei * ETOK
    p = np.sqrt(pp)
    cchi = np.exp(
        -2 * reff * p.imag
        - 2 * pp * (sigma2 - pp * fourth / 3)
        + 1j * (2 * q * reff + pha + 2 * p * (deltar - 2 * sigma2 / reff - 2 * pp * third / 3))
    )
    cchi = degen * s02 * amp * cchi / (q * (reff + deltar) ** 2)
    cchi[0] = 2 * cchi[1] - cchi[2]
    return cchi.imag


def main(argv):
    datfile, outfile = argv[1], argv[2]
    fdat = read_feffdat(datfile)
    params = dict(degen=fdat["degen"], s02=1.0, e0=0.0, ei=0.0,
                  deltar=0.0, sigma2=0.0, third=0.0, fourth=0.0)
    for kv in argv[3:]:
        key, val = kv.split("=")
        params[key] = float(val)

    k = build_k(fdat)
    chi_lin = calc_chi(fdat, k, params, interp="lin")
    chi_cub = calc_chi(fdat, k, params, interp="cubic") if HAVE_SCIPY else None

    with open(outfile, "w") as fh:
        fh.write(f"#source {datfile}\n")
        fh.write(f"#have_scipy {int(HAVE_SCIPY)}\n")
        for key in ("degen", "s02", "e0", "ei", "deltar", "sigma2", "third", "fourth"):
            fh.write(f"#param {key} {params[key]!r}\n")
        fh.write(f"#scalar reff {float(fdat['reff'])!r}\n")
        fh.write(f"#scalar nleg {int(fdat['nleg'])}\n")
        fh.write(f"#scalar file_degen {float(fdat['degen'])!r}\n")
        fh.write(f"#scalar n_kgrid {len(fdat['k'])}\n")
        fh.write(f"#scalar dat_k_first {float(fdat['k'][0])!r}\n")
        fh.write(f"#scalar dat_k_last {float(fdat['k'][-1])!r}\n")
        fh.write(f"#scalar dat_pha_first {float(fdat['pha'][0])!r}\n")
        fh.write(f"#scalar dat_amp_last {float(fdat['amp'][-1])!r}\n")
        cols = "k chi_lin" + (" chi_cubic" if chi_cub is not None else "")
        fh.write(f"#columns {cols}\n")
        fh.write("#data\n")
        for i in range(len(k)):
            if chi_cub is not None:
                fh.write(f"{float(k[i])!r} {float(chi_lin[i])!r} {float(chi_cub[i])!r}\n")
            else:
                fh.write(f"{float(k[i])!r} {float(chi_lin[i])!r}\n")
    print(f"wrote {outfile} ({len(k)} rows, scipy={'yes' if HAVE_SCIPY else 'no'})")


if __name__ == "__main__":
    main(sys.argv)
