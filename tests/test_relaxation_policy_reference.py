"""Compare Python accounting with compiled pinned native policy, without a solver.

The fragments are extracted verbatim from the pinned reader/history routine
and the actual REMAG export adapter. Interface doubles supply scalar options;
this is not a SIESTA calculation or a native GPU relaxation acceptance test.
"""
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from siesta_remag.density_history import policy
from siesta_remag.terminal_exports import (requested_density_exports,
    requested_energy_exports, requested_hamiltonian_exports)


STUBS = '''module fdf
 implicit none
 logical :: flags(22)=.false.
contains
 logical function fdf_get(key, default) result(value)
 character(*),intent(in)::key
 logical,intent(in)::default
 value=default
 select case(key)
 case('DM.AllowReuse'); value=flags(1)
 case('DM.AllowExtrapolation'); if(flags(2))value=flags(3)
 case('UseDIISforDMExtrapolation'); value=flags(20)
 case('UseSVD'); value=flags(21)
 case('UseSVDExperimental'); value=flags(22)
 end select
 end function
 logical function fdf_isblock(key)
 character(*),intent(in)::key
 fdf_isblock=.false.
 end function
end module
module siesta_options
 implicit none
 integer::DM_history_depth=3,idyn=-1
 logical::harrisfun=.false.
end module
module native_policy
 use fdf
 use siesta_options
 implicit none
 logical::ionode=.false.
contains
 logical function leqi(a,b)
 character(*),intent(in)::a,b
 leqi=trim(a)==trim(b)
 end function
 subroutine deprecated(key)
 character(*),intent(in)::key
 end subroutine
 subroutine die(message)
 character(*),intent(in)::message
 error stop 'unsupported native fixture mode'
 end subroutine
 integer function remag_cycle_h_export_gated()
 remag_cycle_h_export_gated=merge(1,0,flags(4))
 end function
 integer function remag_history_enabled()
 remag_history_enabled=merge(1,0,flags(5))
 end function
 integer function remag_integral_force_enabled()
 remag_integral_force_enabled=1
 end function
 integer function remag_final_grid_shadow_enabled()
 remag_final_grid_shadow_enabled=merge(1,0,flags(12))
 end function
 integer function remag_final_density_shadow_enabled()
 remag_final_density_shadow_enabled=merge(1,0,flags(13))
 end function
 subroutine remag_dm_check(code)
 integer,intent(in)::code
 if(code/=0)error stop 'native policy interface'
 end subroutine
 integer function remag_history_configure(depth,mode)
 integer,intent(in)::depth,mode
 write(*,'(2(i0,1x))',advance='no') depth,mode
 remag_history_configure=0
 end function
 subroutine choose(dyntyp)
 character(*),intent(in)::dyntyp
 logical::usesavecg,usesaveddata=.false.,broyden_optim=.false.,fire_optim=.false.,td_elec_dyn=.false.
 integer,parameter::dp=kind(1.d0)
 real(dp)::rstart_time,totime
'''

EXPORT_SCOPE = '''subroutine exports()
 type spin_type
 logical::SO
 end type
 type(spin_type)::spin
 integer::h_exports,energy_exports,density_exports,need_final_h,istp=2,nbk=0,nwk=0,nkpol=0
 logical::write_H_at_end_of_cycle,savehs,write_coop=.false.,write_hs_history,TS_HS_save,ts_write_history
 logical::fc_save_dhs=.false.,bornz=.false.,do_pdos=.false.,w90_processing=.false.
 logical::w90_wrapper_processing=.false.,writetdwf=.false.,TS_DE_save,write_DM_at_end_of_cycle
 logical::dumpcharge,orbmoms,write_cdf
 write_H_at_end_of_cycle=flags(14);savehs=flags(15);write_hs_history=flags(16)
 TS_HS_save=flags(17);ts_write_history=flags(18);need_final_h=merge(1,0,flags(19))
 TS_DE_save=flags(7);write_DM_at_end_of_cycle=flags(6);dumpcharge=flags(9)
 spin%SO=flags(10);orbmoms=flags(11);write_cdf=flags(8)
'''


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def test_broyden_and_cg_match_compiled_native_policies(tmp_path):
    compiler = shutil.which('gfortran')
    root = Path(__file__).resolve().parents[1]
    source = root/'upstream/siesta/Src'
    if not compiler or not source.exists():
        pytest.skip('Pinned source and Fortran compiler needed for native policy arithmetic')
    selector = between((source/'read_options.F90').read_text(),
        "  if (leqi(dyntyp,'cg')) then", '  ! Maximum number of steps in MD/coordinate optimization')
    history = between((source/'m_new_dm.F90').read_text(),
        '  subroutine get_allowed_history_depth(n)', '  end subroutine get_allowed_history_depth')
    extrapolation = between((root/'native/remag_history.F90.in').read_text(),
        '  subroutine remag_history_policy(depth)', '  subroutine remag_history_select(reuse)')
    exports = between((root/'native/remag_cycle.F90.in').read_text(),
        '    ! Host H is needed only', '    c=[kind,')
    program = STUBS + selector + '\nend subroutine\n' + history + '\nend subroutine\n'
    program += extrapolation + EXPORT_SCOPE + exports
    program += '''write(*,'(4(i0,1x))') idyn,h_exports,density_exports,energy_exports
end subroutine
end module
program check
 use native_policy
 implicit none
 integer::mode,raw(22),ios,n
 do
 read(*,*,iostat=ios)mode,raw
 if(ios/=0)exit
 flags=raw==1
 if(mode==1)then
 call choose('cg')
 else
 call choose('broyden')
 endif
 call get_allowed_history_depth(n)
 call remag_history_policy(n)
 call exports()
 enddo
end program
'''
    (tmp_path/'reference.F90').write_text(program)
    subprocess.run([compiler, '-cpp', '-DCDF', '-DNCDF_4', '-ffree-line-length-none',
        '-fcheck=all', 'reference.F90', '-o', 'reference'], cwd=tmp_path,
        check=True, capture_output=True, text=True)
    rng = random.Random(20261117)
    cases = [[False]*22, [True]*22] + [[bool(rng.getrandbits(1)) for _ in range(22)] for _ in range(48)]
    rows = []
    texts = []
    for mode in (1, 2):
        for flags in cases:
            flags = flags.copy()
            flags[11] = False  # Final-H shadow requires a separate consumer audit.
            labels = ['DM.AllowReuse', None, 'DM.AllowExtrapolation', None, None,
                'Write.DM.End.Of.Cycle', 'TS.DE.Save', 'CDF.Save', 'WriteDenchar', None,
                'WriteOrbMom', None, None, 'Write.H.End.Of.Cycle', 'SaveHS', 'Write.HS.History',
                'TS.HS.Save', 'Write.TSHS.History', 'SCF.Recompute-H-After-SCF',
                'UseDIISforDMExtrapolation', 'UseSVD', 'UseSVDExperimental']
            lines = ['Write.DM false', 'DM.History.Depth 3',
                'MD.TypeOfRun '+('CG' if mode == 1 else 'Broyden'),
                'Spin '+('spin-orbit' if flags[9] else 'polarized')]
            lines += [key+' '+str(value).lower() for i,(key,value) in enumerate(zip(labels,flags))
                      if key and (i != 2 or flags[1])]
            text = '\n'.join(lines)
            rows.append(' '.join(map(str,[mode]+[int(v) for v in flags])))
            texts.append((text, flags))
    output = subprocess.run([str(tmp_path/'reference')], input='\n'.join(rows)+'\n',
        capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(output) == len(texts) == 100
    for line,(text,flags) in zip(output,texts):
        depth, mode, idyn, h, d, e = map(int,line.split())
        assert idyn == 0
        assert policy(text) == dict(depth=depth, mode=mode)
        assert requested_hamiltonian_exports(text, flags[3]) == h
        assert requested_density_exports(text, flags[4], flags[12]) == d
        assert requested_energy_exports(text, True) == e
