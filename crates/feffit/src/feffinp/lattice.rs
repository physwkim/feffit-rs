//! A crystallographic unit cell and the fractional → Cartesian transform.

/// A unit cell: edge lengths `a, b, c` (Å) and angles `alpha, beta, gamma`
/// (degrees), with the conventional axis assignment (α between **b**,**c**;
/// β between **a**,**c**; γ between **a**,**b**).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lattice {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// α, between **b** and **c** (degrees).
    pub alpha: f64,
    /// β, between **a** and **c** (degrees).
    pub beta: f64,
    /// γ, between **a** and **b** (degrees).
    pub gamma: f64,
}

impl Lattice {
    /// A cubic cell of edge `a` (all angles 90°).
    pub fn cubic(a: f64) -> Self {
        Self {
            a,
            b: a,
            c: a,
            alpha: 90.0,
            beta: 90.0,
            gamma: 90.0,
        }
    }

    /// The three Cartesian lattice vectors **a**, **b**, **c** (rows), placing
    /// **a** along *x* and **b** in the *xy*-plane — the standard convention
    /// shared by FEFF/Atoms and pymatgen's `Lattice.from_parameters`.
    pub fn vectors(&self) -> [[f64; 3]; 3] {
        let (ca, cb, cg) = (
            self.alpha.to_radians().cos(),
            self.beta.to_radians().cos(),
            self.gamma.to_radians().cos(),
        );
        let sg = self.gamma.to_radians().sin();

        let va = [self.a, 0.0, 0.0];
        let vb = [self.b * cg, self.b * sg, 0.0];

        let cx = self.c * cb;
        let cy = self.c * (ca - cb * cg) / sg;
        // cz² = c²(1 − cos²β − ((cosα − cosβ·cosγ)/sinγ)²); clamp tiny negatives
        // from round-off to zero so a near-degenerate cell does not yield NaN.
        let cz_sq = self.c * self.c - cx * cx - cy * cy;
        let cz = cz_sq.max(0.0).sqrt();
        let vc = [cx, cy, cz];

        [va, vb, vc]
    }

    /// Convert fractional coordinates to Cartesian (Å):
    /// `cart = f₀·**a** + f₁·**b** + f₂·**c**`.
    pub fn frac_to_cart(&self, frac: [f64; 3]) -> [f64; 3] {
        let [va, vb, vc] = self.vectors();
        [
            frac[0] * va[0] + frac[1] * vb[0] + frac[2] * vc[0],
            frac[0] * va[1] + frac[1] * vb[1] + frac[2] * vc[1],
            frac[0] * va[2] + frac[1] * vb[2] + frac[2] * vc[2],
        ]
    }

    /// The cell volume (Å³), `|**a** · (**b** × **c**)|`.
    pub fn volume(&self) -> f64 {
        let [va, vb, vc] = self.vectors();
        let cross = [
            vb[1] * vc[2] - vb[2] * vc[1],
            vb[2] * vc[0] - vb[0] * vc[2],
            vb[0] * vc[1] - vb[1] * vc[0],
        ];
        (va[0] * cross[0] + va[1] * cross[1] + va[2] * cross[2]).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(p: [f64; 3], q: [f64; 3]) -> f64 {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    #[test]
    fn cubic_vectors_are_orthogonal_axes() {
        let l = Lattice::cubic(3.61);
        let [va, vb, vc] = l.vectors();
        assert!((va[0] - 3.61).abs() < 1e-12 && va[1].abs() < 1e-12 && va[2].abs() < 1e-12);
        assert!((vb[1] - 3.61).abs() < 1e-12 && vb[0].abs() < 1e-12);
        assert!((vc[2] - 3.61).abs() < 1e-12 && vc[0].abs() < 1e-12 && vc[1].abs() < 1e-12);
        assert!((l.volume() - 3.61f64.powi(3)).abs() < 1e-9);
    }

    #[test]
    fn fcc_nearest_neighbour_distance() {
        // fcc Cu, a = 3.61: the (½,½,0) face-centre sits a/√2 from the corner.
        let l = Lattice::cubic(3.61);
        let corner = l.frac_to_cart([0.0, 0.0, 0.0]);
        let face = l.frac_to_cart([0.5, 0.5, 0.0]);
        assert!((dist(corner, face) - 3.61 / 2f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn hexagonal_volume() {
        // a=b=3, c=5, γ=120°: V = a²·c·sin(120°).
        let l = Lattice {
            a: 3.0,
            b: 3.0,
            c: 5.0,
            alpha: 90.0,
            beta: 90.0,
            gamma: 120.0,
        };
        let expect = 3.0 * 3.0 * 5.0 * 120f64.to_radians().sin();
        assert!((l.volume() - expect).abs() < 1e-9);
    }
}
