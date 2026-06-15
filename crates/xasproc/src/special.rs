//! Faithful port of the Cephes special functions that `autobk` uncertainty
//! bands need: the error function `erf` and the inverse Student-t CDF
//! `t_ppf` (≡ Cephes `stdtri`).
//!
//! Every routine here is translated verbatim from the scipy 1.13.1 Cephes
//! C sources (`scipy/special/cephes/*.c`, configured with `UNK` + `DENORMAL`):
//!
//! - `erf`, `erfc`            — `ndtr.c`
//! - `ndtri`                  — `ndtri.c`
//! - `incbet` + `pseries`/`incbcf`/`incbd` — `incbet.c`
//! - `incbi`                  — `incbi.c`
//! - `stdtri` (`t_ppf`)       — `stdtr.c`
//! - `Gamma`, `lgam`/`lgam_sgn` — `gamma.c`
//! - `beta`, `lbeta`          — `beta.c`
//! - `polevl`, `p1evl`        — `polevl.h`
//!
//! Numeric constants and polynomial coefficients are the shortest decimal that
//! round-trips to the same `f64` as the verbatim Cephes literals (the cited `.c`
//! files carry a few extra, `f64`-indistinguishable digits); the parsed values
//! are therefore bit-identical to scipy's.
//!
//! The control flow (including the `goto`-driven state machine in `incbi`) is
//! reproduced exactly so the results match scipy/larch to round-off. larch's
//! `autobk_delta_chi` uses `scipy.special.erf` (this `erf`) and
//! `scipy.stats.t.ppf` (scipy's `stdtrit`); for the integer degrees of freedom
//! that arise in autobk (`len(chi)-nspl`, `nmue-nspl`) this `stdtri` agrees
//! with `stdtrit` to ~1e-14.

use std::f64::consts::PI;

// --- machine constants (const.c, UNK + DENORMAL) ---------------------------
const MACHEP: f64 = 1.110_223_024_625_156_5E-16; // 2**-53
const MAXLOG: f64 = 7.097_827_128_933_84E2; // log(DBL_MAX)
const MINLOG: f64 = -7.451_332_191_019_412E2; // log(2**-1075)
const MAXGAM: f64 = 171.624_376_956_302_7;

// --- polevl.h --------------------------------------------------------------

/// Evaluate a polynomial whose coefficients are stored highest-degree first:
/// `coef[0]*x^N + coef[1]*x^(N-1) + ... + coef[N]`. (`polevl`)
fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// As [`polevl`] but the leading (highest-degree) coefficient is assumed to be
/// `1.0` and omitted from `coef`. (`p1evl`)
fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

// --- gamma.c ---------------------------------------------------------------

const GAMMA_P: [f64; 7] = [
    1.601_195_224_767_518_5E-4,
    1.191_351_470_065_863_8E-3,
    1.042_137_975_617_615_8E-2,
    4.763_678_004_571_372E-2,
    2.074_482_276_484_359_8E-1,
    4.942_148_268_014_971E-1,
    1.0,
];
const GAMMA_Q: [f64; 8] = [
    -2.315_818_733_241_201_4E-5,
    5.396_055_804_933_034E-4,
    -4.456_419_138_517_973E-3,
    1.181_397_852_220_604_3E-2,
    3.582_363_986_054_986_5E-2,
    -2.345_917_957_182_433_5E-1,
    7.143_049_170_302_73E-2,
    1.0,
];
const STIR: [f64; 5] = [
    7.873_113_957_930_937E-4,
    -2.295_499_616_133_781_3E-4,
    -2.681_326_178_057_812_4E-3,
    3.472_222_216_054_586_6E-3,
    8.333_333_333_334_822E-2,
];
const MAXSTIR: f64 = 143.01608;
const SQTPI: f64 = 2.506_628_274_631_000_7;
const LOGPI: f64 = 1.144_729_885_849_400_2;

/// Gamma function via Stirling's formula (valid for `33 <= x <= 172`).
fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let mut w = 1.0 / x;
    w = 1.0 + w * polevl(w, &STIR);
    let mut y = x.exp();
    if x > MAXSTIR {
        // avoid overflow in pow()
        let v = x.powf(0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = x.powf(x - 0.5) / y;
    }
    SQTPI * y * w
}

/// Gamma function (`gamma.c` `Gamma`).
fn gamma(mut x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let q = x.abs();

    if q > 33.0 {
        let mut sgngam = 1.0;
        if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                return f64::INFINITY; // pole
            }
            let i = p as i64;
            if (i & 1) == 0 {
                sgngam = -1.0;
            }
            let mut z = q - p;
            if z > 0.5 {
                p += 1.0;
                z = q - p;
            }
            z = q * (PI * z).sin();
            if z == 0.0 {
                return sgngam * f64::INFINITY;
            }
            z = z.abs();
            z = PI / (z * stirf(q));
            return sgngam * z;
        } else {
            return stirf(x);
        }
    }

    let mut z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 0.0 {
        if x > -1.0e-9 {
            return gamma_small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1.0e-9 {
            return gamma_small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    x -= 2.0;
    let p = polevl(x, &GAMMA_P);
    let q = polevl(x, &GAMMA_Q);
    z * p / q
}

/// The `small:` tail of `Gamma` (small argument near a pole).
fn gamma_small(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        f64::INFINITY
    } else {
        z / ((1.0 + 0.5772156649015329 * x) * x)
    }
}

const LGAM_A: [f64; 5] = [
    8.116_141_674_705_085E-4,
    -5.950_619_042_843_014E-4,
    7.936_503_404_577_169E-4,
    -2.777_777_777_300_997E-3,
    8.333_333_333_333_319E-2,
];
const LGAM_B: [f64; 6] = [
    -1.378_251_525_691_208_6E3,
    -3.880_163_151_346_378_4E4,
    -3.316_129_927_388_712E5,
    -1.162_370_974_927_623E6,
    -1.721_737_008_208_396_6E6,
    -8.535_556_642_457_654E5,
];
const LGAM_C: [f64; 6] = [
    -3.518_157_014_365_234_5E2,
    -1.706_421_066_518_811_5E4,
    -2.205_285_905_538_544_5E5,
    -1.139_334_443_679_825_2E6,
    -2.532_523_071_775_829_4E6,
    -2.018_891_414_335_327_7E6,
];
const LS2PI: f64 = 0.918_938_533_204_672_8; // log(sqrt(2*pi))
const MAXLGM: f64 = 2.556348e305;

/// Logarithm of |Gamma(x)|, also returning the sign. (`gamma.c` `lgam_sgn`)
fn lgam_sgn(mut x: f64) -> (f64, i32) {
    let mut sign = 1;
    if !x.is_finite() {
        return (x, sign);
    }

    if x < -34.0 {
        let q = -x;
        let (w, _) = lgam_sgn(q);
        let p = q.floor();
        if p == q {
            return (f64::INFINITY, sign);
        }
        let i = p as i64;
        if (i & 1) == 0 {
            sign = -1;
        } else {
            sign = 1;
        }
        let mut z = q - p;
        if z > 0.5 {
            let p = p + 1.0;
            z = p - q;
        }
        z = q * (PI * z).sin();
        if z == 0.0 {
            return (f64::INFINITY, sign);
        }
        z = LOGPI - z.ln() - w;
        return (z, sign);
    }

    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return (f64::INFINITY, sign);
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            sign = -1;
            z = -z;
        } else {
            sign = 1;
        }
        if u == 2.0 {
            return (z.ln(), sign);
        }
        p -= 2.0;
        x += p;
        let p = x * polevl(x, &LGAM_B) / p1evl(x, &LGAM_C);
        return (z.ln() + p, sign);
    }

    if x > MAXLGM {
        return (sign as f64 * f64::INFINITY, sign);
    }

    let mut q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1.0e8 {
        return (q, sign);
    }
    let p = 1.0 / (x * x);
    if x >= 1000.0 {
        q += ((7.936_507_936_507_937e-4 * p - 2.777_777_777_777_778e-3) * p
            + 0.083_333_333_333_333_33)
            / x;
    } else {
        q += polevl(p, &LGAM_A) / x;
    }
    (q, sign)
}

/// Natural log of |Gamma(x)|. (`gamma.c` `lgam`)
fn lgam(x: f64) -> f64 {
    lgam_sgn(x).0
}

// --- beta.c ----------------------------------------------------------------

const ASYMP_FACTOR: f64 = 1e6;

/// Asymptotic expansion of `ln|B(a,b)|` for `a > ASYMP_FACTOR*max(|b|,1)`.
fn lbeta_asymp(a: f64, b: f64) -> (f64, i32) {
    let (mut r, sgn) = lgam_sgn(b);
    r -= b * a.ln();
    r += b * (1.0 - b) / (2.0 * a);
    r += b * (1.0 - b) * (1.0 - 2.0 * b) / (12.0 * a * a);
    r += -b * b * (1.0 - b) * (1.0 - b) / (12.0 * a * a * a);
    (r, sgn)
}

fn beta_negint(a: i32, b: f64) -> f64 {
    if b == (b as i32) as f64 && 1.0 - a as f64 - b > 0.0 {
        let sgn = if (b as i32) % 2 == 0 { 1.0 } else { -1.0 };
        sgn * beta(1.0 - a as f64 - b, b)
    } else {
        f64::INFINITY
    }
}

fn lbeta_negint(a: i32, b: f64) -> f64 {
    if b == (b as i32) as f64 && 1.0 - a as f64 - b > 0.0 {
        lbeta(1.0 - a as f64 - b, b)
    } else {
        f64::INFINITY
    }
}

/// Beta function `B(a,b) = Γ(a)Γ(b)/Γ(a+b)`. (`beta.c` `beta`)
fn beta(mut a: f64, mut b: f64) -> f64 {
    let mut sign = 1.0;

    if a <= 0.0 && a == a.floor() {
        if a == (a as i32) as f64 {
            return beta_negint(a as i32, b);
        } else {
            return sign * f64::INFINITY;
        }
    }
    if b <= 0.0 && b == b.floor() {
        if b == (b as i32) as f64 {
            return beta_negint(b as i32, a);
        } else {
            return sign * f64::INFINITY;
        }
    }

    if a.abs() < b.abs() {
        std::mem::swap(&mut a, &mut b);
    }

    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        let (y, s) = lbeta_asymp(a, b);
        return s as f64 * y.exp();
    }

    let mut y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let (yy, sgngam) = lgam_sgn(y);
        let mut y = yy;
        sign *= sgngam as f64;
        let (yb, sgngam) = lgam_sgn(b);
        y = yb - y;
        sign *= sgngam as f64;
        let (ya, sgngam) = lgam_sgn(a);
        y += ya;
        sign *= sgngam as f64;
        if y > MAXLOG {
            return sign * f64::INFINITY;
        }
        return sign * y.exp();
    }

    y = gamma(y);
    a = gamma(a);
    b = gamma(b);
    if y == 0.0 {
        return sign * f64::INFINITY;
    }

    if (a.abs() - y.abs()).abs() > (b.abs() - y.abs()).abs() {
        y = b / y;
        y *= a;
    } else {
        y = a / y;
        y *= b;
    }
    y
}

/// Natural log of |beta(a,b)|. (`beta.c` `lbeta`)
fn lbeta(mut a: f64, mut b: f64) -> f64 {
    let mut sign = 1.0;

    if a <= 0.0 && a == a.floor() {
        if a == (a as i32) as f64 {
            return lbeta_negint(a as i32, b);
        } else {
            return sign * f64::INFINITY;
        }
    }
    if b <= 0.0 && b == b.floor() {
        if b == (b as i32) as f64 {
            return lbeta_negint(b as i32, a);
        } else {
            return sign * f64::INFINITY;
        }
    }

    if a.abs() < b.abs() {
        std::mem::swap(&mut a, &mut b);
    }

    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        let (y, _) = lbeta_asymp(a, b);
        return y;
    }

    let mut y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let (yy, sgngam) = lgam_sgn(y);
        let mut y = yy;
        sign *= sgngam as f64;
        let (yb, sgngam) = lgam_sgn(b);
        y = yb - y;
        sign *= sgngam as f64;
        let (ya, sgngam) = lgam_sgn(a);
        y += ya;
        sign *= sgngam as f64;
        let _ = sign;
        return y;
    }

    y = gamma(y);
    a = gamma(a);
    b = gamma(b);
    if y == 0.0 {
        return sign * f64::INFINITY;
    }

    if (a.abs() - y.abs()).abs() > (b.abs() - y.abs()).abs() {
        y = b / y;
        y *= a;
    } else {
        y = a / y;
        y *= b;
    }
    if y < 0.0 {
        y = -y;
    }
    y.ln()
}

// --- ndtr.c: erf / erfc ----------------------------------------------------

const ERF_T: [f64; 5] = [
    9.604_973_739_870_516,
    9.002_601_972_038_427E1,
    2.232_005_345_946_843E3,
    7.003_325_141_128_051E3,
    5.559_230_130_103_949E4,
];
const ERF_U: [f64; 5] = [
    3.356_171_416_475_031E1,
    5.213_579_497_801_527E2,
    4.594_323_829_709_801E3,
    2.262_900_006_138_909_5E4,
    4.926_739_426_086_359E4,
];
const ERFC_P: [f64; 9] = [
    2.461_969_814_735_305E-10,
    5.641_895_648_310_689E-1,
    7.463_210_564_422_699,
    4.863_719_709_856_814E1,
    1.965_208_329_560_771E2,
    5.264_451_949_954_773E2,
    9.345_285_271_719_576E2,
    1.027_551_886_895_157_2E3,
    5.575_353_353_693_994E2,
];
const ERFC_Q: [f64; 8] = [
    1.322_819_511_547_449_9E1,
    8.670_721_408_859_897E1,
    3.549_377_788_878_199E2,
    9.757_085_017_432_055E2,
    1.823_909_166_879_097_3E3,
    2.246_337_608_187_109_7E3,
    1.656_663_091_941_613_4E3,
    5.575_353_408_177_277E2,
];
const ERFC_R: [f64; 6] = [
    5.641_895_835_477_551E-1,
    1.275_366_707_599_781,
    5.019_050_422_511_805,
    6.160_210_979_930_536,
    7.409_742_699_504_489_5,
    2.978_866_653_721_002_2,
];
const ERFC_S: [f64; 6] = [
    2.260_528_632_201_172_6,
    9.396_035_249_380_015,
    1.204_895_398_080_966_6E1,
    1.708_144_507_475_659E1,
    9.608_968_090_632_859,
    3.369_076_451_000_815,
];

/// Error function. (`ndtr.c` `erf`)
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &ERF_T) / p1evl(z, &ERF_U)
}

/// Complementary error function. (`ndtr.c` `erfc`)
pub fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z < -MAXLOG {
        // underflow
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    let z = z.exp();
    let (p, q) = if x < 8.0 {
        (polevl(x, &ERFC_P), p1evl(x, &ERFC_Q))
    } else {
        (polevl(x, &ERFC_R), p1evl(x, &ERFC_S))
    };
    let mut y = (z * p) / q;
    if a < 0.0 {
        y = 2.0 - y;
    }
    if y != 0.0 {
        y
    } else {
        // underflow
        if a < 0.0 { 2.0 } else { 0.0 }
    }
}

// --- ndtri.c ---------------------------------------------------------------

const S2PI: f64 = 2.506_628_274_631_000_7; // sqrt(2*pi)
const NDTRI_P0: [f64; 5] = [
    -5.996_335_010_141_079E1,
    9.800_107_541_859_997E1,
    -5.667_628_574_690_703E1,
    1.393_126_093_872_796_8E1,
    -1.239_165_838_673_812_5,
];
const NDTRI_Q0: [f64; 8] = [
    1.954_488_583_381_417_6,
    4.676_279_128_988_815,
    8.636_024_213_908_905E1,
    -2.254_626_878_541_193_7E2,
    2.002_602_123_800_606_6E2,
    -8.203_722_561_683_334E1,
    1.590_562_251_262_117E1,
    -1.183_316_211_213_3,
];
const NDTRI_P1: [f64; 9] = [
    4.055_448_923_059_624_5,
    3.152_510_945_998_938_8E1,
    5.716_281_922_464_213E1,
    4.408_050_738_932_008E1,
    1.468_495_619_288_580_3E1,
    2.186_633_068_507_902_5,
    -1.402_560_791_713_545E-1,
    -3.504_246_268_278_482E-2,
    -8.574_567_851_546_854E-4,
];
const NDTRI_Q1: [f64; 8] = [
    1.577_998_832_564_667_5E1,
    4.539_076_351_288_792E1,
    4.131_720_382_546_72E1,
    1.504_253_856_929_075E1,
    2.504_649_462_083_094,
    -1.421_829_228_547_877_9E-1,
    -3.808_064_076_915_783E-2,
    -9.332_594_808_954_574E-4,
];
const NDTRI_P2: [f64; 9] = [
    3.237_748_917_769_460_3,
    6.915_228_890_689_842,
    3.938_810_252_924_744_4,
    1.333_034_608_158_075_5,
    2.014_853_895_491_790_8E-1,
    1.237_166_348_178_200_3E-2,
    3.015_815_535_082_354_3E-4,
    2.658_069_746_867_375_5E-6,
    6.239_745_391_849_833E-9,
];
const NDTRI_Q2: [f64; 8] = [
    6.024_270_393_647_42,
    3.679_835_638_561_608_7,
    1.377_020_994_890_813_2,
    2.162_369_935_944_966_3E-1,
    1.342_040_060_885_431_8E-2,
    3.280_144_646_821_277_4E-4,
    2.892_478_647_453_806_8E-6,
    6.790_194_080_099_813E-9,
];

/// Inverse of the standard normal CDF. (`ndtri.c` `ndtri`)
fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&y0) {
        return f64::NAN;
    }
    let mut code = 1;
    let mut y = y0;
    if y > 1.0 - 0.135_335_283_236_612_7 {
        // exp(-2)
        y = 1.0 - y;
        code = 0;
    }

    if y > 0.135_335_283_236_612_7 {
        y -= 0.5;
        let y2 = y * y;
        let mut x = y + y * (y2 * polevl(y2, &NDTRI_P0) / p1evl(y2, &NDTRI_Q0));
        x *= S2PI;
        return x;
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &NDTRI_P1) / p1evl(z, &NDTRI_Q1)
    } else {
        z * polevl(z, &NDTRI_P2) / p1evl(z, &NDTRI_Q2)
    };
    let x = x0 - x1;
    if code != 0 { -x } else { x }
}

// --- incbet.c --------------------------------------------------------------

const BIG: f64 = 4.503599627370496e15;
const BIGINV: f64 = 2.220_446_049_250_313e-16;

/// Power series for the incomplete beta integral (b*x small). (`incbet.c` `pseries`)
fn pseries(a: f64, b: f64, x: f64) -> f64 {
    let ai = 1.0 / a;
    let mut u = (1.0 - b) * x;
    let mut v = u / (a + 1.0);
    let t1 = v;
    let mut t = u;
    let mut n = 2.0;
    let mut s = 0.0;
    let z = MACHEP * ai;
    while v.abs() > z {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;

    let u = a * x.ln();
    if (a + b) < MAXGAM && u.abs() < MAXLOG {
        let t = 1.0 / beta(a, b);
        s = s * t * x.powf(a);
    } else {
        let t = -lbeta(a, b) + u + s.ln();
        s = if t < MINLOG { 0.0 } else { t.exp() };
    }
    s
}

/// Continued fraction expansion #1 for the incomplete beta integral. (`incbcf`)
fn incbcf(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = a + b;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = b - 1.0;
    let mut k7 = k4;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let mut ans = 1.0;
    let mut r = 1.0;
    let mut n = 0;
    let thresh = 3.0 * MACHEP;
    loop {
        let mut xk = -(x * k1 * k2) / (k3 * k4);
        let mut pk = pkm1 + pkm2 * xk;
        let mut qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (x * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if (qk.abs() + pk.abs()) > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
        n += 1;
        if n >= 300 {
            break;
        }
    }
    ans
}

/// Continued fraction expansion #2 for the incomplete beta integral. (`incbd`)
fn incbd(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = b - 1.0;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = a + b;
    let mut k7 = a + 1.0;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let z = x / (1.0 - x);
    let mut ans = 1.0;
    let mut r = 1.0;
    let mut n = 0;
    let thresh = 3.0 * MACHEP;
    loop {
        let mut xk = -(z * k1 * k2) / (k3 * k4);
        let mut pk = pkm1 + pkm2 * xk;
        let mut qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (z * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if (qk.abs() + pk.abs()) > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
        n += 1;
        if n >= 300 {
            break;
        }
    }
    ans
}

/// Incomplete beta integral `I_x(a,b)`. (`incbet.c` `incbet`)
fn incbet(aa: f64, bb: f64, xx: f64) -> f64 {
    if aa <= 0.0 || bb <= 0.0 {
        return f64::NAN;
    }
    if xx <= 0.0 || xx >= 1.0 {
        if xx == 0.0 {
            return 0.0;
        }
        if xx == 1.0 {
            return 1.0;
        }
        return f64::NAN;
    }

    let mut flag = 0;
    if (bb * xx) <= 1.0 && xx <= 0.95 {
        return incbet_done(pseries(aa, bb, xx), flag);
    }

    let mut w = 1.0 - xx;
    let (a, b, xc, x);
    if xx > (aa / (aa + bb)) {
        flag = 1;
        a = bb;
        b = aa;
        xc = xx;
        x = w;
    } else {
        a = aa;
        b = bb;
        xc = w;
        x = xx;
    }

    if flag == 1 && (b * x) <= 1.0 && x <= 0.95 {
        return incbet_done(pseries(a, b, x), flag);
    }

    // choose expansion for better convergence
    let y = x * (a + b - 2.0) - (a - 1.0);
    if y < 0.0 {
        w = incbcf(a, b, x);
    } else {
        w = incbd(a, b, x) / xc;
    }

    // multiply w by x^a (1-x)^b Γ(a+b)/(a Γ(a) Γ(b))
    let mut y = a * x.ln();
    let mut t = b * xc.ln();
    if (a + b) < MAXGAM && y.abs() < MAXLOG && t.abs() < MAXLOG {
        t = xc.powf(b);
        t *= x.powf(a);
        t /= a;
        t *= w;
        t *= 1.0 / beta(a, b);
        return incbet_done(t, flag);
    }
    // resort to logarithms
    y += t - lbeta(a, b);
    y += (w / a).ln();
    t = if y < MINLOG { 0.0 } else { y.exp() };
    incbet_done(t, flag)
}

/// The `done:` tail of `incbet` (un-reflect if `a`/`b` were swapped).
fn incbet_done(mut t: f64, flag: i32) -> f64 {
    if flag == 1 {
        if t <= MACHEP {
            t = 1.0 - MACHEP;
        } else {
            t = 1.0 - t;
        }
    }
    t
}

// --- incbi.c ---------------------------------------------------------------

/// Phases of the `incbi` root finder, mirroring the C `goto` labels.
enum Phase {
    Ihalve,
    Newt,
    Under,
    Done,
}

/// Inverse incomplete beta integral: returns `x` with `incbet(a,b,x) = yy0`.
/// (`incbi.c` `incbi`, a verbatim translation of its `goto` state machine.)
fn incbi(aa: f64, bb: f64, yy0: f64) -> f64 {
    if yy0 <= 0.0 {
        return 0.0;
    }
    if yy0 >= 1.0 {
        return 1.0;
    }

    let mut x0 = 0.0;
    let mut yl = 0.0;
    let mut x1 = 1.0;
    let mut yh = 1.0;
    let mut nflg = 0;

    let mut a;
    let mut b;
    let mut y0;
    let mut x;
    let mut y = 0.0;
    let mut rflg;
    let mut dithresh;
    let mut dir;
    let mut di;
    let mut lgm;
    let mut yp;
    let mut d;

    let mut phase;

    if aa <= 1.0 || bb <= 1.0 {
        dithresh = 1.0e-6;
        rflg = 0;
        a = aa;
        b = bb;
        y0 = yy0;
        x = a / (a + b);
        y = incbet(a, b, x);
        phase = Phase::Ihalve;
    } else {
        dithresh = 1.0e-4;
        yp = -ndtri(yy0);
        if yy0 > 0.5 {
            rflg = 1;
            a = bb;
            b = aa;
            y0 = 1.0 - yy0;
            yp = -yp;
        } else {
            rflg = 0;
            a = aa;
            b = bb;
            y0 = yy0;
        }
        lgm = (yp * yp - 3.0) / 6.0;
        x = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        d = yp * (x + lgm).sqrt() / x
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lgm + 5.0 / 6.0 - 2.0 / (3.0 * x));
        d *= 2.0;
        if d < MINLOG {
            phase = Phase::Under; // x=1; goto under -> x=0; done
        } else {
            x = a / (a + b * d.exp());
            y = incbet(a, b, x);
            yp = (y - y0) / y0;
            if yp.abs() < 0.2 {
                phase = Phase::Newt;
            } else {
                phase = Phase::Ihalve;
            }
        }
    }

    loop {
        match phase {
            Phase::Ihalve => {
                dir = 0;
                di = 0.5;
                let mut next: Option<Phase> = None;
                for i in 0..100 {
                    if i != 0 {
                        x = x0 + di * (x1 - x0);
                        if x == 1.0 {
                            x = 1.0 - MACHEP;
                        }
                        if x == 0.0 {
                            di = 0.5;
                            x = x0 + di * (x1 - x0);
                            if x == 0.0 {
                                next = Some(Phase::Under);
                                break;
                            }
                        }
                        y = incbet(a, b, x);
                        yp = (x1 - x0) / (x1 + x0);
                        if yp.abs() < dithresh {
                            next = Some(Phase::Newt);
                            break;
                        }
                        yp = (y - y0) / y0;
                        if yp.abs() < dithresh {
                            next = Some(Phase::Newt);
                            break;
                        }
                    }
                    if y < y0 {
                        x0 = x;
                        yl = y;
                        if dir < 0 {
                            dir = 0;
                            di = 0.5;
                        } else if dir > 3 {
                            di = 1.0 - (1.0 - di) * (1.0 - di);
                        } else if dir > 1 {
                            di = 0.5 * di + 0.5;
                        } else {
                            di = (y0 - y) / (yh - yl);
                        }
                        dir += 1;
                        if x0 > 0.75 {
                            if rflg == 1 {
                                rflg = 0;
                                a = aa;
                                b = bb;
                                y0 = yy0;
                            } else {
                                rflg = 1;
                                a = bb;
                                b = aa;
                                y0 = 1.0 - yy0;
                            }
                            x = 1.0 - x;
                            y = incbet(a, b, x);
                            x0 = 0.0;
                            yl = 0.0;
                            x1 = 1.0;
                            yh = 1.0;
                            next = Some(Phase::Ihalve);
                            break;
                        }
                    } else {
                        x1 = x;
                        if rflg == 1 && x1 < MACHEP {
                            x = 0.0;
                            next = Some(Phase::Done);
                            break;
                        }
                        yh = y;
                        if dir > 0 {
                            dir = 0;
                            di = 0.5;
                        } else if dir < -3 {
                            di *= di;
                        } else if dir < -1 {
                            di *= 0.5;
                        } else {
                            di = (y - y0) / (yh - yl);
                        }
                        dir -= 1;
                    }
                }
                if let Some(p) = next {
                    phase = p;
                    continue;
                }
                // for loop ran all 100 iterations (LOSS)
                if x0 >= 1.0 {
                    x = 1.0 - MACHEP;
                    phase = Phase::Done;
                } else if x <= 0.0 {
                    phase = Phase::Under;
                } else {
                    phase = Phase::Newt; // fall through
                }
            }
            Phase::Under => {
                x = 0.0;
                phase = Phase::Done;
            }
            Phase::Newt => {
                if nflg != 0 {
                    phase = Phase::Done;
                    continue;
                }
                nflg = 1;
                lgm = lgam(a + b) - lgam(a) - lgam(b);
                let mut goto_done = false;
                for i in 0..8 {
                    if i != 0 {
                        y = incbet(a, b, x);
                    }
                    if y < yl {
                        x = x0;
                        y = yl;
                    } else if y > yh {
                        x = x1;
                        y = yh;
                    } else if y < y0 {
                        x0 = x;
                        yl = y;
                    } else {
                        x1 = x;
                        yh = y;
                    }
                    if x == 1.0 || x == 0.0 {
                        break;
                    }
                    d = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + lgm;
                    if d < MINLOG {
                        goto_done = true;
                        break;
                    }
                    if d > MAXLOG {
                        break;
                    }
                    d = d.exp();
                    d = (y - y0) / d;
                    let mut xt = x - d;
                    if xt <= x0 {
                        y = (x - x0) / (x1 - x0);
                        xt = x0 + 0.5 * y * (x - x0);
                        if xt <= 0.0 {
                            break;
                        }
                    }
                    if xt >= x1 {
                        y = (x1 - x) / (x1 - x0);
                        xt = x1 - 0.5 * y * (x1 - x);
                        if xt >= 1.0 {
                            break;
                        }
                    }
                    x = xt;
                    if (d / x).abs() < 128.0 * MACHEP {
                        goto_done = true;
                        break;
                    }
                }
                if goto_done {
                    phase = Phase::Done;
                } else {
                    // did not converge
                    dithresh = 256.0 * MACHEP;
                    phase = Phase::Ihalve;
                }
            }
            Phase::Done => {
                if rflg != 0 {
                    if x <= MACHEP {
                        x = 1.0 - MACHEP;
                    } else {
                        x = 1.0 - x;
                    }
                }
                return x;
            }
        }
    }
}

// --- stdtr.c: inverse Student-t CDF ----------------------------------------

/// Inverse of the Student-t CDF with `df` degrees of freedom: returns `t` such
/// that the Student-t CDF at `t` equals `p`. (`stdtr.c` `stdtri`, with the
/// real-valued `rk = df` exactly as the integer `k` is used there.)
///
/// This is the function larch reaches through `scipy.stats.t.ppf`.
pub fn t_ppf(p: f64, df: f64) -> f64 {
    let rk = df;
    if df <= 0.0 || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    if p > 0.25 && p < 0.75 {
        if p == 0.5 {
            return 0.0;
        }
        let z = 1.0 - 2.0 * p;
        let z = incbi(0.5, 0.5 * rk, z.abs());
        let t = (rk * z / (1.0 - z)).sqrt();
        return if p < 0.5 { -t } else { t };
    }

    let mut rflg = -1.0;
    let mut p = p;
    if p >= 0.5 {
        p = 1.0 - p;
        rflg = 1.0;
    }
    let z = incbi(0.5 * rk, 0.5, 2.0 * p);
    if f64::MAX * z < rk {
        return rflg * f64::INFINITY;
    }
    let t = (rk / z - rk).sqrt();
    rflg * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    #[test]
    fn erf_matches_cephes_spot_values() {
        // (x, scipy.special.erf(x)) from ref_autobk_delta.txt; erf here is the
        // same Cephes routine scipy uses, so agreement is to round-off.
        let cases = [
            (0.0_f64, 0.0_f64),
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (std::f64::consts::FRAC_1_SQRT_2, 0.6826894921370859),
            (1.0, 0.8427007929497148),
            (1.5, 0.9661051464753108),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
            (-0.5, -0.5204998778130465),
            (-1.3, -0.9340079449406524),
            (5.0, 0.9999999999984626),
            (0.25, 0.2763263901682369),
            (4.2, 0.9999999971445058),
        ];
        for (x, want) in cases {
            let got = erf(x);
            if want == 0.0 {
                assert!(got.abs() < 1e-15, "erf({x}) = {got}, want 0");
            } else {
                assert!(
                    rel(got, want) < 1e-13,
                    "erf({x}) = {got}, want {want}, rel={}",
                    rel(got, want)
                );
            }
        }
    }

    #[test]
    fn t_ppf_matches_scipy_spot_values() {
        // (p, df, scipy.stats.t.ppf(p, df)) from ref_autobk_delta.txt. The
        // reference uses scipy's `stdtrit` (a root-find on the t CDF); this is
        // Cephes `stdtri` (inverse via `incbi`). The two algorithms differ by up
        // to ~1.2e-11 across this range; for the large integer df that autobk
        // actually uses (485, 496) they agree to ~1e-14 (see tppf_chi/tppf_bkg
        // in autobk_delta_parity).
        let cases = [
            (0.8413447460685429_f64, 500.0_f64, 1.0010010004999081_f64),
            (0.8413447460685429, 50.0, 1.0101004992285798),
            (0.8413447460685429, 7.0, 1.076713375416128),
            (0.8413447460685429, 3.0, 1.196881354403155),
            (0.9772498680518208, 300.0, 2.0083674672415524),
            (0.9772498680518208, 12.0, 2.2313480940183688),
            (0.5, 25.0, 6.704684444310922e-17),
            (0.6, 25.0, 0.25605968482715136),
            (0.75, 9.0, 0.7027221467513188),
            (0.95, 100.0, 1.66023432606575),
            (0.99, 4.0, 3.7469473879811366),
            (0.025, 30.0, -2.042272456301238),
            (0.1, 1000.0, -1.2823987214609245),
        ];
        for (p, df, want) in cases {
            let got = t_ppf(p, df);
            if want.abs() < 1e-12 {
                assert!(
                    (got - want).abs() < 1e-12,
                    "t_ppf({p}, {df}) = {got}, want {want}"
                );
            } else {
                assert!(
                    rel(got, want) < 5e-11,
                    "t_ppf({p}, {df}) = {got}, want {want}, rel={}",
                    rel(got, want)
                );
            }
        }
    }
}
