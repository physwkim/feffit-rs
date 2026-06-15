//! `params` — `lmfit.Parameters`-style parameters with constraint expressions.
//!
//! Provides a parameter collection where each entry is a free fit variable, a
//! fixed value, or a constraint expression (an [`expr`] AST) over the other
//! parameters, resolved in dependency order. This is the layer feffit uses to
//! turn path parameters (`s02`, `e0`, `deltar`, …) into either fit variables or
//! algebraic constraints — the role `asteval` plays inside `lmfit`.
//!
//! Verified against `lmfit`/`asteval` for the supported grammar. Not yet
//! ported: the EXAFS `sigma2_debye`/`sigma2_eins` helper functions.

pub mod expr;
pub mod parameters;

pub use expr::{parse, BinOp, Expr, ExprError};
pub use parameters::{Param, ParamError, Parameters};
