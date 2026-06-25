#!/usr/bin/env python3
"""Reference generator for the `params` Rust port.

Two references:
  * ref_params_expr.txt    -- raw expression evaluation vs `asteval`
                              (the engine lmfit uses for constraints).
  * ref_params_resolve.txt -- a full Parameters constraint resolution vs
                              `lmfit.Parameters.update_constraints`.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_params.py
"""
from asteval import Interpreter
from lmfit import Parameters

OUT = "crates/feffit/tests/data"

# ---- expression cases (symtable + expressions) ------------------------------
SYM = {"x": 1.3, "y": -2.7, "reff": 2.5478, "amp": 2.0}
EXPRS = [
    "x+y*2",
    "(x+y)*2",
    "x**2**3",          # right-assoc power
    "-x**2",            # unary minus binds looser than **
    "2**-2",            # unary in exponent
    "sqrt(x)+exp(y)",
    "min(x,y,0.5)",
    "max(x,y,0.5)",
    "atan2(y,x)",
    "log(reff)",        # natural log
    "log10(1000)",
    "pi*x",
    "e**2",
    "-7 % 3",
    "7 % -3",
    "3.5e-2*x",
    "abs(y)",
    "sin(x)*cos(y)",
    "amp*reff/(x+1)",
    "(amp + 1)**0.5 - 2*reff",
]


def write_expr_ref():
    a = Interpreter()
    for k, v in SYM.items():
        a.symtable[k] = v
    with open(f"{OUT}/ref_params_expr.txt", "w") as fh:
        for k, v in SYM.items():
            fh.write(f"#sym {k} {float(v)!r}\n")
        for e in EXPRS:
            val = a(e)
            fh.write(f"#expr {float(val)!r} :: {e}\n")
    print(f"wrote ref_params_expr.txt ({len(EXPRS)} expressions)")


# ---- parameter constraint resolution ----------------------------------------
# definitions: (kind, name, *args).  kind in var/varb/fix/expr; const handled separately.
DEFS = [
    ("var", "amp", 2.0),
    ("var", "alpha", 0.002),
    ("varb", "s02", 0.9, 0.0, 1.2),
    ("varb", "tight", 1.5, 0.0, 1.2),     # value above max -> clamped by lmfit
    ("fix", "e0", 1.5),
    ("expr", "delr", "alpha*reff"),
    ("expr", "sig", "0.003 + amp*1e-4"),
    ("expr", "chained", "delr*2 + sqrt(s02)"),
    ("expr", "combo", "e0 + max(amp, s02) - min(alpha, 0.001)"),
]
CONSTS = {"reff": 2.5478}


def write_resolve_ref():
    p = Parameters()
    for name, val in CONSTS.items():
        p._asteval.symtable[name] = val
    for d in DEFS:
        kind, name = d[0], d[1]
        if kind == "var":
            p.add(name, value=d[2], vary=True)
        elif kind == "varb":
            p.add(name, value=d[2], vary=True, min=d[3], max=d[4])
        elif kind == "fix":
            p.add(name, value=d[2], vary=False)
        elif kind == "expr":
            p.add(name, expr=d[2])
    p.update_constraints()

    with open(f"{OUT}/ref_params_resolve.txt", "w") as fh:
        for name, val in CONSTS.items():
            fh.write(f"#const {name} {float(val)!r}\n")
        for d in DEFS:
            kind, name = d[0], d[1]
            if kind == "var":
                fh.write(f"#var {name} {float(d[2])!r}\n")
            elif kind == "varb":
                fh.write(f"#varb {name} {float(d[2])!r} {float(d[3])!r} {float(d[4])!r}\n")
            elif kind == "fix":
                fh.write(f"#fix {name} {float(d[2])!r}\n")
            elif kind == "expr":
                fh.write(f"#expr {name} {d[2]}\n")
        for d in DEFS:
            name = d[1]
            fh.write(f"#expect {name} {float(p[name].value)!r}\n")
    print(f"wrote ref_params_resolve.txt ({len(DEFS)} params)")
    print("  resolved:", {d[1]: round(p[d[1]].value, 8) for d in DEFS})


def main():
    write_expr_ref()
    write_resolve_ref()


if __name__ == "__main__":
    main()
