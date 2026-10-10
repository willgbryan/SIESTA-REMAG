#!/usr/bin/env python3
"""Compare the device FD routine with compiled, unmodified upstream fermid.F.

Without --library, validate and retain only upstream reference fixtures locally.
GPU checks must run on the authorized single Modal instance.
"""
import argparse
import ctypes as ct
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Stubs supply configuration/constants only; the numerical reference is the
# pinned upstream source. Unsupported occupations/excitations are not selected.
STUBS = '''module precision
use iso_c_binding, only: dp=>c_double
end module
module parallel
logical, parameter :: IOnode=.false.
end module
module units
use precision
real(dp), parameter :: Pi=acos(-1._dp)
end module
module fdf
contains
logical function fdf_boolean(key,default)
character(*), intent(in) :: key
logical, intent(in) :: default
fdf_boolean=default
end function
integer function fdf_integer(key,default)
character(*), intent(in) :: key
integer, intent(in) :: default
fdf_integer=default
end function
function fdf_string(key,default) result(value)
character(*), intent(in) :: key,default
character(len(default)) :: value
value=default
end function
logical function leqi(a,b)
character(*), intent(in) :: a,b
leqi=a==b
end function
end module
module m_errorf
use precision
contains
real(dp) function derfc(x)
real(dp), intent(in) :: x
derfc=erfc(x)
end function
real(dp) function derf(x)
real(dp), intent(in) :: x
derf=erf(x)
end function
end module
module sys
contains
subroutine die(message)
character(*), intent(in) :: message
write(*,*) message
error stop 1
end subroutine
end module
'''
WRAPPER = '''subroutine remag_reference_fermi(nspinor,maxspn,nk,maxe,ne,temp,charge,e,wk,q,ef,entropy) bind(C)
use iso_c_binding
use m_fermid, only: fermid
integer(c_int), value :: nspinor,maxspn,nk,maxe,ne
real(c_double), value :: temp,charge
real(c_double), intent(in) :: e(maxe,maxspn,nk),wk(nk)
real(c_double), intent(out) :: q(maxe,maxspn,nk),ef,entropy
entropy=0._c_double
call fermid(nspinor,maxspn,nk,wk,maxe,ne,e,temp,charge,q,ef,entropy)
end subroutine
'''

class Problem(ct.Structure):
    _fields_ = [(n, ct.c_int) for n in ("nspinor", "maxspn", "nk", "maxe", "ne")] + [
        ("temperature", ct.c_double), ("charge", ct.c_double)]


class Result(ct.Structure):
    _fields_ = [(n, ct.c_double) for n in ("fermi", "entropy", "charge")] + [
        ("status", ct.c_int), ("iterations", ct.c_int)]


def pointer(array):
    return array.ctypes.data_as(ct.POINTER(ct.c_double))


def fixtures():
    rng = np.random.default_rng(98271)
    cases = []
    for label, nspinor, maxspn, nk, ne, padding, temperature in (
        ("spinless", 1, 1, 1, 18, 0, .002), ("collinear", 2, 2, 1, 576, 0, .002),
        ("noncollinear", 2, 1, 4, 288, 0, .002), ("soc", 2, 1, 8, 576, 0, .006),
        ("large_spinor", 2, 1, 4, 1152, 0, .006), ("padded_bands", 2, 2, 3, 19, 5, .003),
        ("temperature_floor", 2, 2, 2, 12, 0, 0.), ("degenerate", 2, 1, 4, 32, 0, .001),
        ("full_bands", 1, 1, 3, 12, 0, .002), ("gamma_noncollinear_half_weight",1,1,1,36,0,.002)):
        maxe = ne + padding
        weights = rng.uniform(.1, 1., nk)
        weights /= weights.sum()
        if label == "gamma_noncollinear_half_weight": weights *= .5
        energies = np.full((maxe, maxspn, nk), np.nan, order="F")
        values = rng.uniform(-.1, .15, (ne, maxspn, nk))
        if label == "temperature_floor": values *= .0001
        if label == "degenerate": values = np.round(values, 1)
        energies[:ne] = np.sort(values, axis=0)
        capacity = (2 if nspinor == 1 else 1) * maxspn * ne * weights.sum()
        charge = float(capacity if label == "full_bands" else capacity * .413)
        cases.append((label, Problem(nspinor,maxspn,nk,maxe,ne,temperature,charge), energies,weights))
    return cases


def run(upstream, output, library=None):
    source = upstream / "Src/fermid.F"
    report = dict(reference_source=str(source), reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  configuration="FD; no single excitation; variable total spin; positive k weights, including upstream NC Gamma half weight",
                  native_library=str(library) if library else None, checks=[])
    with tempfile.TemporaryDirectory(prefix="remag-fermi-reference-") as temporary:
        directory = Path(temporary)
        (directory / "stubs.f90").write_text(STUBS)
        (directory / "wrapper.f90").write_text(WRAPPER)
        subprocess.run(["gfortran", "-O2", "-shared", "-fPIC", "-cpp", "-ffixed-line-length-none",
                        "stubs.f90", str(source.resolve()), "wrapper.f90", "-o", "reference.so"],
                       cwd=directory,check=True)
        reference = ct.CDLL(str(directory / "reference.so")).remag_reference_fermi
        reference.argtypes = [ct.c_int]*5 + [ct.c_double]*2 + [ct.POINTER(ct.c_double)]*5
        reference.restype = None
        candidate = None
        if library:
            candidate = ct.CDLL(str(library)).remag_check_fermi
            candidate.argtypes = [ct.POINTER(Problem)] + [ct.POINTER(ct.c_double)]*3 + [ct.POINTER(Result),ct.c_int]
            candidate.restype = ct.c_int
        for name,p,e,w in fixtures():
            q = np.zeros_like(e, order="F")
            ef, entropy = ct.c_double(), ct.c_double()
            reference(p.nspinor,p.maxspn,p.nk,p.maxe,p.ne,p.temperature,p.charge,
                      pointer(e),pointer(w),pointer(q),ct.byref(ef),ct.byref(entropy))
            check = dict(name=name, reference_fermi=ef.value, reference_entropy=entropy.value,
                         reference_charge_error=abs(q.sum()-p.charge),
                         reference_finite=bool(np.isfinite(q).all() and np.isfinite([ef.value,entropy.value]).all()),
                         passed=None)
            if not check["reference_finite"] or check["reference_charge_error"] >= 1.e-10:
                raise RuntimeError(f"Invalid upstream fixture: {check}")
            for graph_mode in ((0,1) if candidate else ()):
                check = dict(check,graph_mode=bool(graph_mode))
                actual, result = np.full_like(q, np.nan), Result()
                status = candidate(ct.byref(p),pointer(e),pointer(w),pointer(actual),ct.byref(result),graph_mode)
                errors = dict(fermi=abs(result.fermi-ef.value), entropy=abs(result.entropy-entropy.value),
                              occupations=float(np.max(np.abs(actual-q))), charge=abs(result.charge-p.charge),
                              occupation_charge=abs(float(actual.sum())-p.charge))
                finite = np.isfinite(actual).all() and np.isfinite(list(errors.values())).all()
                check.update(cuda_status=status,device_status=result.status,iterations=result.iterations,errors=errors,
                             passed=bool(status==0 and result.status==0 and finite and errors["fermi"]<1.e-10
                                         and errors["entropy"]<1.e-9 and errors["occupations"]<1.e-9
                                         and errors["charge"]<1.e-10 and errors["occupation_charge"]<1.e-10))
                check["errors"]={key:value if np.isfinite(value) else None for key,value in errors.items()}
                report["checks"].append(check)
            if not candidate: report["checks"].append(check)
        if candidate:
            for defect,expected in (("shape",1),("nonfinite_energy",2),("nonfinite_temperature",2),
                                    ("negative_charge",2),("bad_weights",2),("insufficient_bands",3),
                                    ("iteration_limit",4)):
                p=Problem(2,1,1,1,1,.1,.5)
                e,w,q=np.zeros(1),np.ones(1),np.zeros(1)
                if defect=="shape": p.nspinor=3
                elif defect=="nonfinite_energy": e[0]=np.nan
                elif defect=="nonfinite_temperature": p.temperature=np.nan
                elif defect=="negative_charge": p.charge=-1.
                elif defect=="bad_weights": w[0]=-.8
                elif defect=="insufficient_bands": p.charge=2.
                elif defect=="iteration_limit": p.charge=0.
                result=Result()
                status=candidate(ct.byref(p),pointer(e),pointer(w),pointer(q),ct.byref(result),1)
                report["checks"].append(dict(name=defect,graph_mode=True,cuda_status=status,
                    device_status=result.status,expected_status=expected,
                    passed=status==0 and result.status==expected))
        report["passed"] = all(c["passed"] for c in report["checks"]) if candidate else None
        report["reference_fixtures_passed"] = True
        report["reference_fixture_count"] = len(fixtures())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Upstream FD fixtures: {report['reference_fixture_count']}; scenarios: {len(report['checks'])}; device checks passed={report['passed']}")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path,default=Path("upstream/siesta"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path)
    args=parser.parse_args()
    result=run(args.upstream,args.output,args.library)
    if args.library and not result["passed"]:
        raise SystemExit("Device Fermi regression failed")
