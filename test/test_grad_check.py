"""Numerical (central-difference) gradient checks for the extended ``Value`` class.

For a scalar function ``f`` built out of ``Value`` operations we compute the
partial derivatives w.r.t. the two inputs in two independent ways:

* **analytic** -- what ``Value.backward()`` produces, i.e. the chain rule applied
  over the computational graph.
* **numeric**  -- the central difference ``(f(x+h) - f(x-h)) / (2h)``, which only
  needs forward evaluations and shares no code with the backward pass.

The two must agree to within ``tol`` (default ``1e-6``). Because the paths are
independent, agreement is strong evidence that every ``_backward`` closure in
``engine.py`` computes the correct local derivative.

Run it as a script to print a table, or let pytest run the test functions:

    .venv\\Scripts\\python.exe test/test_grad_check.py
    .venv\\Scripts\\python.exe -m pytest test/test_grad_check.py -v
"""

from micrograd.engine import Value


def grad_check(f, a, b, h=1e-6, tol=1e-6, label=''):
    """Compare backprop gradients of ``f(a, b)`` against central differences.

    f     : callable(Value, Value) -> Value, the scalar output to differentiate
    a, b  : float, the point at which the derivatives are evaluated
    h     : float, central-difference step; truncation error is O(h^2) while
            roundoff error grows like eps/h, so ~1e-6 balances the two
    tol   : float, maximum allowed absolute difference between the two methods
    label : str, only used to make assertion messages readable

    Returns a dict with the forward value, both gradient pairs and the errors.
    """
    # --- backward pass: build the graph once, then let backprop fill in .grad ---
    va, vb = Value(a), Value(b)
    out = f(va, vb)
    out.backward()
    analytic = (va.grad, vb.grad)

    # --- forward pass only: perturb one input at a time, no graph involved ---
    def forward(x, y):
        return f(Value(x), Value(y)).data

    numeric = (
        (forward(a + h, b) - forward(a - h, b)) / (2.0 * h),
        (forward(a, b + h) - forward(a, b - h)) / (2.0 * h),
    )

    errors = []
    for name, an, nu in zip(('a', 'b'), analytic, numeric):
        err = abs(an - nu)
        errors.append(err)
        assert err < tol, (
            f"{label or 'grad_check'}: d/d{name} disagrees -> "
            f"backprop={an!r}, numeric={nu!r}, error={err:.3e} (tol={tol:g})"
        )

    return {
        'out': out.data,
        'analytic': analytic,   # d f/da, d f/db from Value.backward()
        'numeric': numeric,     # d f/da, d f/db from central differences
        'errors': tuple(errors),
    }


# name -> (f, a, b). The comment above each entry is the hand-derived gradient,
# which is what both methods above should reproduce.
CASES = {
    # f = a + b                    -> da = 1,            db = 1
    'add': (lambda a, b: a + b, 2.0, 3.0),
    # f = a * b                    -> da = b = 3,        db = a = 2
    'mul': (lambda a, b: a * b, 2.0, 3.0),
    # f = a - b                    -> da = 1,            db = -1
    'sub': (lambda a, b: a - b, 5.0, 1.5),
    # f = a / b                    -> da = 1/b = 0.5,    db = -a/b^2 = -1.5
    'div': (lambda a, b: a / b, 6.0, 2.0),
    # f = a ** b (Value exponent)  -> da = b*a^(b-1) = 12, db = a^b*ln(a) = 8*ln2
    'pow': (lambda a, b: a ** b, 2.0, 3.0),
    # f = relu(a * b), a*b = 6 > 0 -> da = b = 2,        db = a = 3
    'relu': (lambda a, b: (a * b).relu(), 3.0, 2.0),
    # f = tanh(a * b), a*b = 0.65  -> da = (1-t^2)*b,     db = (1-t^2)*a
    'tanh': (lambda a, b: (a * b).tanh(), 0.5, 1.3),
    # f = exp(a * b), a*b = 0.65   -> da = f*b,           db = f*a
    'exp': (lambda a, b: (a * b).exp(), 0.5, 1.3),
    # f = log(a * b), a*b = 6 > 0  -> da = 1/a = 0.5,     db = 1/b = 1/3
    'log': (lambda a, b: (a * b).log(), 2.0, 3.0),

    # --- extra coverage for the reflected operators and the scalar exponent ---
    # f = relu(a * b), a*b = -6 < 0 -> da = 0,            db = 0
    'relu_neg': (lambda a, b: (a * b).relu(), -3.0, 2.0),
    # f = a**2 * b                 -> da = 2ab = 12,      db = a^2 = 4
    'pow_scalar': (lambda a, b: a ** 2 * b, 2.0, 3.0),
    # f = 1.5 - a + b              -> da = -1,            db = 1
    'rsub': (lambda a, b: 1.5 - a + b, 2.0, 3.0),
    # f = 1.5 / a + b              -> da = -1.5/a^2,      db = 1
    'rdiv': (lambda a, b: 1.5 / a + b, 2.0, 3.0),
}


def _run_case(name, **kwargs):
    f, a, b = CASES[name]
    return grad_check(f, a, b, label=name, **kwargs)


def test_add():
    _run_case('add')


def test_mul():
    _run_case('mul')


def test_sub():
    _run_case('sub')


def test_div():
    _run_case('div')


def test_pow():
    _run_case('pow')


def test_relu():
    _run_case('relu')


def test_tanh():
    _run_case('tanh')


def test_exp():
    _run_case('exp')


def test_log():
    _run_case('log')


def test_extra_paths():
    """The reflected operators (``__rsub__``/``__rtruediv__``) and the scalar
    exponent branch of ``__pow__``, plus the negative side of ReLU."""
    for name in ('relu_neg', 'pow_scalar', 'rsub', 'rdiv'):
        _run_case(name)


if __name__ == '__main__':
    header = (f"{'case':<11}{'a':>7}{'b':>7}{'f(a,b)':>12}"
              f"{'da(numeric)':>14}{'da(backprop)':>14}"
              f"{'db(numeric)':>14}{'db(backprop)':>14}{'err':>10}")
    print(header)
    print('-' * len(header))

    worst = 0.0
    for name, (f, a, b) in CASES.items():
        r = _run_case(name)
        na, nb = r['numeric']
        aa, ab = r['analytic']
        worst = max(worst, max(r['errors']))
        print(f"{name:<11}{a:>7.3f}{b:>7.3f}{r['out']:>12.6f}"
              f"{na:>14.6f}{aa:>14.6f}{nb:>14.6f}{ab:>14.6f}{max(r['errors']):>10.1e}")

    print(f"\nall {len(CASES)} cases agree within 1e-6 "
          f"(worst error {worst:.2e})")
