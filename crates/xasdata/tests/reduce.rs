//! End-to-end reduction orchestration test: load a real Cu `mu(E)`, then run
//! normalize → autobk → xftf through the `xasdata::reduce` adapters and check
//! that each stage populated the [`XasGroup`] sensibly. The underlying numeric
//! parity with larch is covered by the `xasproc`/`xafsft` test suites; this test
//! guards the field wiring.

use xasdata::{
    ColumnFile, FtParams, MuSpec, XasGroup, autobk_group, build_mu, normalize, xftf_group,
};
use xasproc::{AutobkParams, PreEdgeParams};

const CU_XMU: &str = include_str!("data/cu.xmu");

fn cu_group() -> XasGroup {
    let cf = ColumnFile::from_text(CU_XMU).expect("parse cu.xmu");
    // cu.xmu is a two-column energy/mu file.
    let (energy, mu) = build_mu(&cf, &MuSpec::Raw { energy: 0, mu: 1 }).unwrap();
    XasGroup::from_mu("cu", energy, mu)
}

#[test]
fn normalize_fills_edge_and_norm() {
    let mut g = cu_group();
    let params = PreEdgeParams::default();
    normalize(&mut g, &params);

    // Physical sanity: the Cu K edge sits near 8979 eV with a positive jump.
    let e0 = g.e0.expect("e0");
    assert!(
        (8960.0..9000.0).contains(&e0),
        "Cu K edge e0 should be ~8979 eV, got {e0}"
    );
    assert!(g.edge_step.unwrap() > 0.0, "edge step must be positive");

    // Mean normalized mu over a window safely above the edge averages to ~1
    // (EXAFS wiggles cancel); a single endpoint is not a reliable check.
    let norm = g.norm.as_ref().unwrap();
    assert_eq!(norm.len(), g.energy.len());
    let (mut sum, mut cnt) = (0.0_f64, 0usize);
    for (e, n) in g.energy.iter().zip(norm) {
        if *e >= e0 + 50.0 && *e <= e0 + 400.0 {
            sum += *n;
            cnt += 1;
        }
    }
    let mean = sum / cnt.max(1) as f64;
    assert!(
        cnt > 0 && (0.7..1.3).contains(&mean),
        "post-edge mean normalized mu should be ~1, got {mean:.3} over {cnt} pts"
    );

    // Orchestration contract: the group fields must equal a direct pre_edge call.
    let direct = xasproc::pre_edge(&g.energy, &g.mu, &params);
    assert_eq!(g.norm.as_ref().unwrap(), &direct.norm);
    assert_eq!(g.flat.as_ref().unwrap(), &direct.flat);
    assert_eq!(g.dmude.as_ref().unwrap(), &direct.dmude);
    assert_eq!(g.e0, Some(direct.e0));
    assert_eq!(g.edge_step, Some(direct.edge_step));
}

#[test]
fn autobk_then_xftf_gives_cu_first_shell_peak() {
    let mut g = cu_group();
    normalize(&mut g, &PreEdgeParams::default());
    autobk_group(&mut g, &AutobkParams::default(), 1.0);

    let k = g.k.as_ref().expect("k grid");
    let chi = g.chi.as_ref().expect("chi");
    assert_eq!(k.len(), chi.len());
    assert!(k.len() > 100, "expected a populated k grid");
    assert!(g.bkg.as_ref().unwrap().len() == g.energy.len());
    assert!(
        g.delta_chi.as_ref().is_some_and(|d| d.len() == k.len()),
        "uncertainty band delta_chi should be filled and k-length"
    );

    let did = xftf_group(
        &mut g,
        &FtParams {
            kmin: 2.0,
            kmax: 15.0,
            kweight: 2,
            ..FtParams::default()
        },
    );
    assert!(did, "xftf should run once k/chi exist");

    let r = g.r.as_ref().unwrap();
    let mag = g.chir_mag.as_ref().unwrap();
    assert_eq!(r.len(), mag.len());

    // The Cu fcc first-shell |chi(R)| peak sits near R ≈ 2.2–2.5 Å (before phase
    // correction). Find the global maximum and check it lands there.
    let (imax, _) = mag
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let r_peak = r[imax];
    assert!(
        (1.8..2.8).contains(&r_peak),
        "Cu first-shell |chi(R)| peak should be ~2.2–2.5 Å, got {r_peak:.3} Å"
    );
}
