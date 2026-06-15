//! `params` — `lmfit.Parameters`-style parameters with constraint expressions.
//!
//! Provides a parameter collection where each entry is a free fit variable, a
//! fixed value, or a constraint expression (an [`expr`] AST) over the other
//! parameters, resolved in dependency order. This is the layer feffit uses to
//! turn path parameters (`s02`, `e0`, `deltar`, …) into either fit variables or
//! algebraic constraints — the role `asteval` plays inside `lmfit`.
//!
//! Verified against `lmfit`/`asteval` for the supported grammar. The EXAFS
//! `sigma2_debye`/`sigma2_eins` helpers are not built in; they are supplied by
//! the caller through [`expr::FuncCtx`] (feffit binds them to a path's
//! geometry), since they need data this crate does not own.

pub mod expr;
pub mod parameters;

pub use expr::{BinOp, Expr, ExprError, FuncCtx, NoCtx, parse};
pub use parameters::{Param, ParamError, Parameters};
