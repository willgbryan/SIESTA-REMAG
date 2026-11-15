import pytest
from siesta_remag.terminal_exports import requested_hamiltonian_exports


def test_hamiltonian_copy_requires_a_recorded_consumer():
    assert requested_hamiltonian_exports('SaveHS false',False)==1
    assert requested_hamiltonian_exports('SaveHS false',True)==0
    assert requested_hamiltonian_exports('',True)==4
    assert requested_hamiltonian_exports('SaveHS false\nSave_HS true',True)==0
    for setting,reason in (('Write.H.End.Of.Cycle true',2),('TS.HS.Save true',8),
                           ('Write.HS.History true',4),('Write.TSHS.History true',8),
                           ('SCF.Recompute-H-After-SCF true',64),('CDF.Save true',0)):
        assert requested_hamiltonian_exports('SaveHS false\n'+setting,True)==reason


def test_unqualified_hamiltonian_consumers_cannot_certify_zero_exports():
    for setting in ('Write.H true','OpticalCalculation true','COOP.Write true','BornCharge true',
                    'TDED.WF.Initialize true','%block ProjectedDensityOfStates\n0 1\n%endblock',
                    '%include other.fdf','MD.TypeOfRun FC'):
        with pytest.raises(ValueError):
            requested_hamiltonian_exports('SaveHS false\n'+setting,True)


@pytest.mark.parametrize('label', ['PAO.Basis', 'pao_basis', 'PAO-BASIS'])
def test_explicit_basis_does_not_add_a_hamiltonian_consumer(label):
    basis = f'%block {label}\nFe 1\n n=4 0 2\n 0.0 0.0\n%endblock {label}\n'
    assert requested_hamiltonian_exports(basis+'SaveHS true', True) == 4
    assert requested_hamiltonian_exports(basis+'SaveHS false', True) == 0
    for analysis in ('%block ProjectedDensityOfStates\n0 1\n%endblock',
                     '%block UnqualifiedAnalysis\n1\n%endblock',
                     'OpticalCalculation true'):
        with pytest.raises(ValueError):
            requested_hamiltonian_exports(basis+'SaveHS false\n'+analysis, True)
