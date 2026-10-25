"""Compile shared MAE arithmetic and check independent complex/thermal algebra.

These tests target SOC packing, spin rotation and small energy differences.
They do not qualify a GPU executor or the physical force-theorem approximation.
"""
import ctypes
from decimal import Decimal, localcontext
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler required for native MAE arithmetic')
    work = tmp_path_factory.mktemp('mae_math')
    source = work/'check.cpp'
    source.write_text(r'''
#include "frozen_potential_math.hpp"
using namespace remag::mae;
extern "C" int rotate(int count,const double* q,const double* h,const double* soc,double* out) {
    SpinRotation r;
    if(!spin_rotation(q,r))return 1;
    for(int i=0;i<count;i++)if(!rotate_exchange(r,h+8*i,soc+8*i,out+8*i))return 2;
    return 0;
}
extern "C" void soc_pack(const double* off,double* out) {offsite_soc(off,out);}
extern "C" int invalid_merge(int self) {
    BandPairSum a(1.), b(2.);
    a.add(0.,0.,1.,1.,1.); b.add(0.,0.,1.,1.,1.);
    if(a.merge(self?a:b))return 1;
    return a.finish(.01,0.,0.,1.).finite_result?2:0;
}
extern "C" int bands(int count,int chunk,const double* rows,double origin,double t,
                      double mu_a,double mu_b,double electrons,double* out) {
    BandPairSum total(origin);
    for(int first=0;first<count;first+=chunk) {
        BandPairSum part(origin);
        for(int i=first;i<count&&i<first+chunk;i++) {
            const auto p=rows+5*i;
            part.add(p[0],p[1],p[2],p[3],p[4]);
        }
        total.merge(part);
    }
    const auto r=total.finish(t,mu_a,mu_b,electrons);
    const double values[]={r.band_energy,r.band_free_energy,r.canonical_free_energy,r.entropy,
        r.charge_a,r.charge_b,r.charge_difference,r.charge_error_a,r.charge_error_b,
        r.finite_result?1.:0.,r.electron_count_passed?1.:0.};
    for(int i=0;i<11;i++)out[i]=values[i];
    return r.finite_result?0:1;
}
''')
    lib = work/'check.so'
    subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                    '-shared', '-fPIC', '-I'+str(ROOT/'native'), str(source), '-o', str(lib)],
                   check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(lib))
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    loaded.rotate.argtypes = [ctypes.c_int, ptr, ptr, ptr, ptr]
    loaded.rotate.restype = ctypes.c_int
    loaded.soc_pack.argtypes = [ptr, ptr]
    loaded.soc_pack.restype = None
    loaded.invalid_merge.argtypes = [ctypes.c_int]
    loaded.invalid_merge.restype = ctypes.c_int
    loaded.bands.argtypes = [ctypes.c_int, ctypes.c_int, ptr, *([ctypes.c_double]*5), ptr]
    loaded.bands.restype = ctypes.c_int
    return loaded


def unpack(h):
    h = np.asarray(h)
    return np.stack([h[..., 0]+1j*h[..., 4], h[..., 2]-1j*h[..., 3],
                     h[..., 6]+1j*h[..., 7], h[..., 1]+1j*h[..., 5]], axis=-1).reshape((*h.shape[:-1], 2, 2))


def pack(matrix):
    a, b, c, d = (matrix[..., i, j] for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])
    return np.stack([a.real, d.real, b.real, -b.imag, a.imag, d.imag, c.real, c.imag], axis=-1).copy()


def unitary(q):
    q = np.array(q)/max(abs(np.array(q)))
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[w-1j*z, -y-1j*x], [y-1j*x, w+1j*z]])


def rotate(native, h, soc, q):
    h, soc, q = [np.ascontiguousarray(x, dtype=float) for x in (h, soc, q)]
    out = np.full_like(h, 12345.)
    code = native.rotate(h.size//8, q, h, soc, out)
    return code, out


def test_complex_spin_blocks_keep_soc_fixed(native):
    rng = np.random.default_rng(807)
    h, soc = rng.normal(size=(2, 13, 8))
    q = [0.3, -0.2, 0.7, 0.9]
    code, out = rotate(native, h, soc, q)
    assert code == 0
    u = unitary(q)
    expected = u @ (unpack(h)-unpack(soc)) @ u.conj().T + unpack(soc)
    np.testing.assert_allclose(unpack(out), expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(np.trace(unpack(out), axis1=-2, axis2=-1),
                               np.trace(unpack(h), axis1=-2, axis2=-1), atol=2e-15)
    # Rotating the lattice SOC too is the concrete wrong alternative.
    assert np.max(abs(unpack(out)-u@unpack(h)@u.conj().T)) > 0.1
    np.testing.assert_array_equal(rotate(native, h, soc, -np.array(q))[1], out)


def test_identity_scalar_and_rotation_direction(native):
    h = np.array([[2.**40, 2.**40-1, .125, .25, .5, -.5, -.25, .125]])
    soc = np.zeros_like(h)
    np.testing.assert_array_equal(rotate(native, h, soc, [1, 0, 0, 0])[1], h)
    # A positive pi/2 rotation around y takes the exchange z axis to +x.
    h = pack(np.array([[[3., 0.], [0., 1.]]], dtype=complex))
    _, out = rotate(native, h, soc, [1, 0, 1, 0])
    np.testing.assert_allclose(unpack(out), [[[2., 1.], [1., 2.]]], atol=5e-16)
    # Spin-independent complex orbital entries must not acquire magnetism.
    scalar = pack(np.array([[[3+2j, 0.], [0., 3+2j]]]))
    np.testing.assert_array_equal(rotate(native, scalar, soc, [.3, .4, .5, .6])[1], scalar)
    for scale in (1e-300, 1e300):
        np.testing.assert_allclose(rotate(native, h, soc, np.array([1, 0, 1, 0])*scale)[1], out, atol=0)


@pytest.mark.parametrize('q', [[0, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, 1, 0, 0]])
def test_invalid_rotation_does_not_publish(native, q):
    code, out = rotate(native, np.ones((1, 8)), np.zeros((1, 8)), q)
    assert code == 1
    np.testing.assert_array_equal(out, 12345.)


def test_nonfinite_hamiltonian_does_not_publish(native):
    h = np.ones((1, 8)); h[0, 5] = np.nan
    code, out = rotate(native, h, np.zeros_like(h), [1, 0, 1, 0])
    assert code == 2
    np.testing.assert_array_equal(out, 12345.)


def test_soc_layout_against_pinned_upstream_statements(native):
    source = ROOT/'upstream/siesta/Src/setup_hamiltonian.F'
    if not source.exists():
        pytest.skip('Pinned upstream required for SOC convention check')
    # Extract the actual eight post-grid additions, rather than duplicating
    # their sign/component table in the test.
    statements = re.findall(r'H\(:,(\d)\) = H\(:,\1\) ([+-]) (real|aimag)\(H_so_off\(:,(\d)\)(?:, dp)?\)', source.read_text())
    assert len(statements) == 8
    off = np.array([1+2j, 3+4j, 5+6j, 7+8j])
    expected = np.zeros(8)
    for output, sign, part, index in statements:
        z = off[int(index)-1]
        expected[int(output)-1] = (1 if sign == '+' else -1)*(z.real if part == 'real' else z.imag)
    out = np.empty(8)
    native.soc_pack(off.view(float), out)
    np.testing.assert_array_equal(out, expected)


def test_generalized_spectrum_invariance_only_without_fixed_soc(native):
    linalg = pytest.importorskip('scipy.linalg')
    rng = np.random.default_rng(18)
    a = rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6))
    h = (a+a.conj().T)*.5
    a = rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6))
    soc = (a+a.conj().T)*.1
    a = rng.normal(size=(3, 3))
    overlap = np.kron(a@a.T+np.eye(3), np.eye(2))
    blocks = lambda x: x.reshape(3, 2, 3, 2).transpose(0, 2, 1, 3)
    matrix = lambda x: unpack(x).transpose(0, 2, 1, 3).reshape(6, 6)
    ph, psoc = pack(blocks(h)), pack(blocks(soc))
    q = [.7, .2, -.3, .4]
    without = matrix(rotate(native, ph, np.zeros_like(ph), q)[1])
    with_soc = matrix(rotate(native, ph, psoc, q)[1])
    np.testing.assert_allclose(with_soc, with_soc.conj().T, atol=5e-16)
    original = linalg.eigvalsh(h, overlap)
    np.testing.assert_allclose(linalg.eigvalsh(without, overlap), original, atol=2e-15)
    assert np.max(abs(linalg.eigvalsh(with_soc, overlap)-original)) > 1e-3


def bands(native, rows, *, origin=0., temperature=.01, mu_a=0., mu_b=0., electrons=1., chunk=7):
    rows = np.ascontiguousarray(rows, dtype=float)
    out = np.empty(11)
    code = native.bands(len(rows), chunk, rows, origin, temperature, mu_a, mu_b, electrons, out)
    return code, out


def entropy(p):
    occupied, empty = np.maximum(p, 1e-15), np.maximum(1-p, 1e-15)
    return -occupied*np.log(occupied)-empty*np.log(empty)


def test_band_thermodynamic_scopes_and_charge_residual(native):
    rows = np.array([[-3., -2.9, .8, .7, 1.], [1., 1.2, .2, .3+1e-8, 1.]])
    code, out = bands(native, rows, origin=-2., mu_a=-.1, mu_b=.2)
    ea, eb, qa, qb, w = rows.T
    delta_e = np.sum(qb*eb-qa*ea)
    delta_s = np.sum(w*(entropy(qb/w)-entropy(qa/w)))
    delta_f = delta_e-.01*delta_s
    canonical = delta_f+.2*(1-qb.sum())-(-.1)*(1-qa.sum())
    assert code == 0 and out[9] == 1 and out[10] == 0
    np.testing.assert_allclose(out[:4], [delta_e, delta_f, canonical, delta_s], atol=5e-16)
    assert out[8] == pytest.approx(1e-8, abs=2e-16)
    # A charge error is retained and fails the unchanged 1e-10 electron gate.
    assert abs(out[1]-out[2]) > 1e-9


def test_small_directional_difference_survives_extensive_cancellation(native):
    # The input values are exactly binary-representable; Decimal gives an
    # independent high-precision sum of the supplied FP64 values.
    count = 8192
    ea = np.full(count, -2.**20)
    eb = ea.copy(); eb[::31] += 2.**-25
    qa = np.full(count, 1/count); qb = qa.copy()
    qa[0] += 2.**-40; qb[1] += 2.**-40
    rows = np.stack([ea, eb, qa, qb, np.full(count, 2/count)], axis=1)
    with localcontext() as context:
        context.prec = 65
        expected = sum(Decimal.from_float(float(b))*Decimal.from_float(float(qb))-
                       Decimal.from_float(float(a))*Decimal.from_float(float(qa)) for a,b,qa,qb,_ in rows)
    answers = []
    for chunk in [1, 17, 256, count]:
        code, out = bands(native, rows, origin=-2.**20, temperature=0., electrons=1+2.**-40, chunk=chunk)
        assert code == 0 and out[10] == 1
        assert out[0] == pytest.approx(float(expected), abs=1e-21)
        answers.append(out[0])
    assert max(answers)-min(answers) < 1e-21
    assert abs(np.sum(qb*eb)-np.sum(qa*ea)-float(expected)) > 1e-12


def test_canonical_difference_is_energy_gauge_invariant(native):
    # Different, sub-tolerance residual charges make the canonical correction
    # necessary: raw band differences shift by C*(N_b-N_a).
    rows = np.array([[-4., -3.75, .75, .75+2.**-36, 1.],
                     [2., 2.25, .25+2.**-37, .25, 1.]])
    _, before = bands(native, rows, origin=-2., mu_a=-1., mu_b=-.75)
    shift = 2.**20
    rows[:, :2] += shift
    _, after = bands(native, rows, origin=-2.+shift, mu_a=-1.+shift, mu_b=-.75+shift)
    assert before[10] == after[10] == 1
    np.testing.assert_array_equal(after[2:], before[2:])
    np.testing.assert_allclose(after[:2]-before[:2], shift*before[6], atol=1e-16)


def test_reduction_rejects_mixed_origins_and_duplicate_self_merge(native):
    assert native.invalid_merge(0) == 0
    assert native.invalid_merge(1) == 0


@pytest.mark.parametrize('row', [[np.nan, 0, .5, .5, 1], [0, 0, -.1, .5, 1],
                                  [0, 0, .5, 1.1, 1], [0, 0, 0, 0, 0]])
def test_invalid_band_data_fails_closed(native, row):
    assert bands(native, [row])[0] == 1
