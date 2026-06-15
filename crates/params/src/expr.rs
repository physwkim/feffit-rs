//! A small arithmetic expression parser/evaluator covering the subset of
//! `asteval` (the engine `lmfit` uses for constraint expressions) that EXAFS
//! fits need: numeric literals, named variables, the binary operators
//! `+ - * / % **`, unary `+`/`-`, parentheses, and the common math functions.
//!
//! Semantics are matched to asteval/numpy where they differ from naive parsing:
//! `**` is right-associative and binds tighter than unary minus (so `-2**2`
//! is `-4` and `2**-2` is `0.25`), and `log` is the natural logarithm.
//!
//! Not supported (deliberately): comparisons, boolean/conditional expressions,
//! indexing, attribute access, and the EXAFS `sigma2_debye`/`sigma2_eins`
//! helpers (those need their own port).

use std::collections::HashMap;
use std::f64::consts::{E, PI};
use std::fmt;

/// An error from tokenizing, parsing, or evaluating an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprError {
    /// A character the tokenizer does not recognise.
    BadChar(char),
    /// A malformed number literal.
    BadNumber(String),
    /// The parser reached an unexpected token (or end of input).
    Parse(String),
    /// A variable name not present in the symbol table at eval time.
    UnknownVar(String),
    /// A function name that is not supported.
    UnknownFunc(String),
    /// A supported function called with the wrong number of arguments.
    Arity(String),
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprError::BadChar(c) => write!(f, "unexpected character '{c}'"),
            ExprError::BadNumber(s) => write!(f, "invalid number '{s}'"),
            ExprError::Parse(s) => write!(f, "parse error: {s}"),
            ExprError::UnknownVar(s) => write!(f, "unknown variable '{s}'"),
            ExprError::UnknownFunc(s) => write!(f, "unknown function '{s}'"),
            ExprError::Arity(s) => write!(f, "wrong number of arguments to '{s}'"),
        }
    }
}

impl std::error::Error for ExprError {}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Pow,
    LParen,
    RParen,
    Comma,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, ExprError> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit()
            || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // exponent part: e / E [+/-] digits
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let lit: String = chars[start..i].iter().collect();
            let val: f64 = lit.parse().map_err(|_| ExprError::BadNumber(lit.clone()))?;
            out.push(Tok::Num(val));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
        } else {
            match c {
                '+' => out.push(Tok::Plus),
                '-' => out.push(Tok::Minus),
                '*' => {
                    if i + 1 < chars.len() && chars[i + 1] == '*' {
                        out.push(Tok::Pow);
                        i += 1;
                    } else {
                        out.push(Tok::Star);
                    }
                }
                '/' => out.push(Tok::Slash),
                '%' => out.push(Tok::Percent),
                '(' => out.push(Tok::LParen),
                ')' => out.push(Tok::RParen),
                ',' => out.push(Tok::Comma),
                _ => return Err(ExprError::BadChar(c)),
            }
            i += 1;
        }
    }
    Ok(out)
}

/// A parsed expression (abstract syntax tree).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

/// A binary operator in an [`Expr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn expect(&mut self, t: &Tok) -> Result<(), ExprError> {
        match self.next() {
            Some(ref got) if got == t => Ok(()),
            other => Err(ExprError::Parse(format!("expected {t:?}, found {other:?}"))),
        }
    }

    // expr := add
    fn parse_expr(&mut self) -> Result<Expr, ExprError> {
        self.parse_add()
    }

    // add := mul (('+'|'-') mul)*
    fn parse_add(&mut self) -> Result<Expr, ExprError> {
        let mut lhs = self.parse_mul()?;
        while let Some(op) = match self.peek() {
            Some(Tok::Plus) => Some(BinOp::Add),
            Some(Tok::Minus) => Some(BinOp::Sub),
            _ => None,
        } {
            self.next();
            let rhs = self.parse_mul()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    // mul := unary (('*'|'/'|'%') unary)*
    fn parse_mul(&mut self) -> Result<Expr, ExprError> {
        let mut lhs = self.parse_unary()?;
        while let Some(op) = match self.peek() {
            Some(Tok::Star) => Some(BinOp::Mul),
            Some(Tok::Slash) => Some(BinOp::Div),
            Some(Tok::Percent) => Some(BinOp::Rem),
            _ => None,
        } {
            self.next();
            let rhs = self.parse_unary()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    // unary := ('+'|'-') unary | power
    // (so that ** binds tighter than a leading unary minus: -2**2 == -4)
    fn parse_unary(&mut self) -> Result<Expr, ExprError> {
        match self.peek() {
            Some(Tok::Plus) => {
                self.next();
                self.parse_unary()
            }
            Some(Tok::Minus) => {
                self.next();
                Ok(Expr::Neg(Box::new(self.parse_unary()?)))
            }
            _ => self.parse_power(),
        }
    }

    // power := atom ('**' unary)?   (right-associative; RHS may be unary)
    fn parse_power(&mut self) -> Result<Expr, ExprError> {
        let base = self.parse_atom()?;
        if let Some(Tok::Pow) = self.peek() {
            self.next();
            let exp = self.parse_unary()?;
            Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    // atom := number | name | name '(' args ')' | '(' expr ')'
    fn parse_atom(&mut self) -> Result<Expr, ExprError> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Expr::Num(v)),
            Some(Tok::LParen) => {
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                if let Some(Tok::LParen) = self.peek() {
                    self.next();
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            match self.peek() {
                                Some(Tok::Comma) => {
                                    self.next();
                                }
                                _ => break,
                            }
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(ExprError::Parse(format!("unexpected {other:?}"))),
        }
    }
}

/// Parse an expression string into an [`Expr`] AST.
pub fn parse(s: &str) -> Result<Expr, ExprError> {
    let toks = tokenize(s)?;
    let mut p = Parser { toks, pos: 0 };
    let e = p.parse_expr()?;
    if p.pos != p.toks.len() {
        return Err(ExprError::Parse(format!(
            "trailing tokens after expression: {:?}",
            &p.toks[p.pos..]
        )));
    }
    Ok(e)
}

impl Expr {
    /// Collect the variable names referenced by this expression (for dependency
    /// resolution). Function names are not included.
    pub fn vars(&self, out: &mut Vec<String>) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(n) => out.push(n.clone()),
            Expr::Neg(e) => e.vars(out),
            Expr::Bin(_, a, b) => {
                a.vars(out);
                b.vars(out);
            }
            Expr::Call(_, args) => {
                for a in args {
                    a.vars(out);
                }
            }
        }
    }

    /// Evaluate against a symbol table. Built-in constants `pi`/`e` are used as
    /// fallbacks only when not shadowed by the table.
    pub fn eval(&self, sym: &HashMap<String, f64>) -> Result<f64, ExprError> {
        match self {
            Expr::Num(v) => Ok(*v),
            Expr::Var(n) => {
                if let Some(v) = sym.get(n) {
                    Ok(*v)
                } else {
                    // built-in constants, used only when not shadowed by the table
                    match n.as_str() {
                        "pi" => Ok(PI),
                        "e" => Ok(E),
                        _ => Err(ExprError::UnknownVar(n.clone())),
                    }
                }
            }
            Expr::Neg(e) => Ok(-e.eval(sym)?),
            Expr::Bin(op, a, b) => {
                let x = a.eval(sym)?;
                let y = b.eval(sym)?;
                Ok(match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                    // Python/numpy float `%`: result takes the sign of the divisor
                    BinOp::Rem => x - y * (x / y).floor(),
                    BinOp::Pow => x.powf(y),
                })
            }
            Expr::Call(name, args) => {
                let vals: Result<Vec<f64>, _> = args.iter().map(|a| a.eval(sym)).collect();
                call_func(name, &vals?)
            }
        }
    }
}

fn call_func(name: &str, a: &[f64]) -> Result<f64, ExprError> {
    let one = || -> Result<f64, ExprError> {
        if a.len() == 1 {
            Ok(a[0])
        } else {
            Err(ExprError::Arity(name.to_string()))
        }
    };
    let two = || -> Result<(f64, f64), ExprError> {
        if a.len() == 2 {
            Ok((a[0], a[1]))
        } else {
            Err(ExprError::Arity(name.to_string()))
        }
    };
    Ok(match name {
        "sin" => one()?.sin(),
        "cos" => one()?.cos(),
        "tan" => one()?.tan(),
        "asin" | "arcsin" => one()?.asin(),
        "acos" | "arccos" => one()?.acos(),
        "atan" | "arctan" => one()?.atan(),
        "sinh" => one()?.sinh(),
        "cosh" => one()?.cosh(),
        "tanh" => one()?.tanh(),
        "exp" => one()?.exp(),
        "log" | "ln" => one()?.ln(),
        "log10" => one()?.log10(),
        "sqrt" => one()?.sqrt(),
        "abs" | "fabs" => one()?.abs(),
        "floor" => one()?.floor(),
        "ceil" => one()?.ceil(),
        "atan2" | "arctan2" => {
            let (y, x) = two()?;
            y.atan2(x)
        }
        "pow" => {
            let (x, y) = two()?;
            x.powf(y)
        }
        "min" => {
            if a.is_empty() {
                return Err(ExprError::Arity(name.to_string()));
            }
            a.iter().copied().fold(f64::INFINITY, f64::min)
        }
        "max" => {
            if a.is_empty() {
                return Err(ExprError::Arity(name.to_string()));
            }
            a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
        _ => return Err(ExprError::UnknownFunc(name.to_string())),
    })
}
