#!/usr/bin/env python3
"""Apply a small, exact-match GPU integration to a disposable copy of pinned SIESTA."""
import argparse
import re
from pathlib import Path

DM_MODULE = '''module m_remag_dm
  use iso_c_binding, only: c_int, c_double, c_ptr
  use sys, only: die
  implicit none
  real(c_double), parameter :: remag_zero(3)=0._c_double
  real(c_double), parameter :: remag_one_weight(1)=1._c_double
  real(c_double), parameter :: remag_half_weight(1)=0.5_c_double
  interface
    integer(c_int) function remag_phase_enter(phase,geometry,iteration) bind(C)
      import c_int
      integer(c_int),value :: phase,geometry,iteration
    end function
    subroutine remag_geometry_changed() bind(C)
    end subroutine
    subroutine remag_trace_finalize() bind(C)
    end subroutine
    integer(c_int) function remag_defer_scf_log() bind(C)
      import c_int
    end function
    integer(c_int) function remag_dm_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_hs_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_hartree_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_flow_active() bind(C)
      import c_int
    end function
    integer(c_int) function remag_flow_needs_setup(mesh,spins,nsm,npcc,cell,atom,core,vna) bind(C)
      import c_int,c_double
      integer(c_int),intent(in) :: mesh(*)
      integer(c_int),value :: spins,nsm,npcc
      real(c_double),intent(in) :: cell(*),atom(*),core(*),vna(*)
    end function
    integer(c_int) function remag_flow_prepare(mesh,spins,nsm,npcc,cell,atom,core,vna,b,cutoff,g,volume,dvol,harris) bind(C)
      import c_int,c_double
      integer(c_int),intent(in) :: mesh(*)
      integer(c_int),value :: spins,nsm,npcc
      real(c_double),intent(in) :: cell(*),atom(*),core(*),vna(*),b(*),g(*)
      real(c_double),value :: cutoff,volume,dvol,harris
    end function
    integer(c_int) function remag_flow_arm(n,spins,rho,v) bind(C)
      import c_int,c_double
      integer(c_int),value :: n,spins
      real(c_double),intent(in) :: rho(*),v(*)
    end function
    integer(c_int) function remag_flow_solve(rho,v,energies,hs,xs) bind(C)
      import c_int,c_double
      real(c_double),intent(in) :: rho(*),v(*)
      real(c_double),intent(out) :: energies(*),hs(*),xs(*)
    end function
    integer(c_int) function remag_flow_end() bind(C)
      import c_int
    end function
    integer(c_int) function remag_hartree_needs_setup(mesh,cell) bind(C)
      import c_int,c_double
      integer(c_int), intent(in) :: mesh(*)
      real(c_double), intent(in) :: cell(*)
    end function
    integer(c_int) function remag_hartree_prepare(mesh,cell,b,cutoff,volume) bind(C)
      import c_int,c_double
      integer(c_int), intent(in) :: mesh(*)
      real(c_double), intent(in) :: cell(*),b(*)
      real(c_double), value :: cutoff,volume
    end function
    integer(c_int) function remag_hartree_solve(mesh,cell,rho,u,v,stress) bind(C)
      import c_int,c_double,c_ptr
      integer(c_int), intent(in) :: mesh(*)
      real(c_double), intent(in) :: cell(*)
      type(c_ptr), value :: rho,v
      real(c_double), intent(out) :: u,stress(*)
    end function
    integer(c_int) function remag_hs_armed() bind(C)
      import c_int
    end function
    integer(c_int) function remag_hs_prepare(kind,n,nnz,nh,no,nk,ldh,num,ptr,cols,mapping,xij,k,h,s) bind(C)
      import c_int,c_double
      integer(c_int),value :: kind,n,nnz,nh,no,nk,ldh
      integer(c_int),intent(in) :: num(*),ptr(*),cols(*),mapping(*)
      real(c_double),intent(in) :: xij(*),k(*),h(*),s(*)
    end function
    integer(c_int) function remag_hs_select(spin,k,h,s) bind(C)
      import c_int
      integer(c_int),value :: spin,k
      type(*),intent(in) :: h(*),s(*)
    end function
    integer(c_int) function remag_hs_end() bind(C)
      import c_int
    end function
    integer(c_int) function remag_fermi_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_spectra_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_spectra_armed() bind(C)
      import c_int
    end function
    integer(c_int) function remag_fermi_export(n,e,q) bind(C)
      import c_int,c_double
      integer(c_int),value :: n
      real(c_double),intent(inout) :: e(*),q(*)
    end function
    integer(c_int) function remag_fermi_armed() bind(C)
      import c_int
    end function
    integer(c_int) function remag_fermi_end() bind(C)
      import c_int
    end function
    integer(c_int) function remag_fermi_prepare(nspinor,maxspn,nk,maxe,ne,e,q,wk,temp,charge) bind(C)
      import c_int,c_double
      integer(c_int),value :: nspinor,maxspn,nk,maxe,ne
      real(c_double),intent(in) :: e(*),q(*),wk(*)
      real(c_double),value :: temp,charge
    end function
    integer(c_int) function remag_fermi_compute(nspinor,maxspn,nk,maxe,ne,e,wk,temp,charge,q,ef,entropy) bind(C)
      import c_int,c_double
      integer(c_int),value :: nspinor,maxspn,nk,maxe,ne
      real(c_double),intent(in) :: e(*),wk(*)
      real(c_double),value :: temp,charge
      real(c_double),intent(out) :: q(*),ef,entropy
    end function
    integer(c_int) function remag_dm_normalization_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_dm_normalization_arm(rows,nh,eh,spinor,s,charge,tolerance) bind(C)
      import c_int,c_double
      integer(c_int),value :: rows,nh,eh,spinor
      real(c_double),intent(in) :: s(*)
      real(c_double),value :: charge,tolerance
    end function
    integer(c_int) function remag_dm_normalization_complete() bind(C)
      import c_int
    end function
    integer(c_int) function remag_arm_vectors(slot,export_host) bind(C)
      import c_int
      integer(c_int),value :: slot,export_host
    end function
    integer(c_int) function remag_dm_begin(kind,n,nnz,nh,eh,no,num,ptr,cols,mapping,xyz) bind(C)
      import c_int,c_double
      integer(c_int),value :: kind,n,nnz,nh,eh,no
      integer(c_int),intent(in) :: num(*),ptr(*),cols(*),mapping(*)
      real(c_double),intent(in) :: xyz(*)
    end function
    integer(c_int) function remag_dm_accumulate(slot,nev,outspin,occ,energies,cutoff,k) bind(C)
      import c_int,c_double
      integer(c_int),value :: slot,nev,outspin
      real(c_double),value :: cutoff
      real(c_double),intent(in) :: occ(*),energies(*),k(*)
    end function
    integer(c_int) function remag_dm_finish(d,e) bind(C)
      import c_int,c_double
      real(c_double),intent(out) :: d(*),e(*)
    end function
    integer(c_int) function remag_dm_normalize(rows,nh,eh,factor,d,e) bind(C)
      import c_int,c_double
      integer(c_int),value :: rows,nh,eh
      real(c_double),value :: factor
      real(c_double),intent(in) :: d(*),e(*)
    end function
  end interface
contains
  subroutine remag_dm_check(status)
    integer,intent(in) :: status
    if(status /= 0) then
      write(*,*) 'REMAG density matrix error: ',status
      call die('REMAG density matrix operation failed')
    end if
  end subroutine
  subroutine remag_dm_arm(slot,export_host)
    integer,intent(in) :: slot
    logical,intent(in) :: export_host
    call remag_dm_check(remag_arm_vectors(slot,merge(1,0,export_host)))
  end subroutine
  subroutine remag_dm_serial(nuo,n,nodes)
    integer,intent(in) :: nuo,n,nodes
    if(nuo/=n.or.nodes/=1) call die('REMAG DM requires one process')
  end subroutine
end module
'''

SCF_MODULE = '''module m_remag_scf
  use iso_c_binding, only: c_int,c_double,c_ptr,c_loc,c_null_ptr
  use m_remag_dm, only: remag_dm_check
  implicit none
  real(c_double), save :: remag_scf_charge(8)=0._c_double
  interface
    integer(c_int) function remag_state_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_state_active() bind(C)
      import c_int
    end function
    integer(c_int) function remag_state_prepare(n,spins,rows,soc,history,weight,linear_weight,condition, &
        initial_energy,mix_first,num,ptr,s,kin,vkb,on,off,constants,h,hold,d,dold,e,eold) bind(C)
      import c_int,c_double,c_ptr
      integer(c_int),value :: n,spins,rows,soc,history,initial_energy,mix_first
      real(c_double),value :: weight,linear_weight,condition
      integer(c_int),intent(in) :: num(*),ptr(*)
      real(c_double),intent(in) :: s(*),kin(*),vkb(*),constants(*)
      type(c_ptr),value :: on,off,eold
      real(c_double),intent(inout) :: h(*),hold(*),d(*),dold(*),e(*)
    end function
    integer(c_int) function remag_state_begin(iteration) bind(C)
      import c_int
      integer(c_int),value :: iteration
    end function
    integer(c_int) function remag_state_seed() bind(C)
      import c_int
    end function
    integer(c_int) function remag_state_finish_hamiltonian() bind(C)
      import c_int
    end function
    integer(c_int) function remag_state_scalars(values) bind(C)
      import c_int,c_double
      real(c_double),intent(out) :: values(*)
    end function
    integer(c_int) function remag_state_mix() bind(C)
      import c_int
    end function
    integer(c_int) function remag_state_export() bind(C)
      import c_int
    end function
  end interface
contains
  subroutine remag_scf_guard()
    use sys, only: die
    use siesta_options
    use parallel, only: Nodes
    use m_spin, only: spin,Spiral
    use m_mixing, only: mix_method
    use m_mixing_scf, only: scf_mixs,scf_mix,mix_spin,MIX_SPIN_ALL
    use m_ts_global_vars, only: TSmode
    use dftu_specs, only: switch_dftu
    use option_charges_m, only: OPT_CHARGE_SCF,OPT_CHARGE_SCF_AFTER_MIX
    if (Nodes/=1.or.isolve/=SOLVE_DIAGON.or.TSmode.or.Spiral.or.switch_dftu.or. &
        .not.mixH.or.mix_charge.or.mix_after_convergence.or. &
        harrisfun.or.h_setup_only.or.nscf<1.or.monitor_forces_in_scf.or..not.normalize_dm_during_scf.or. &
        writeDM.or.writeH.or.writedm_cdf.or.writedm_cdf_history.or. &
        writedmhs_cdf.or.writedmhs_cdf_history.or.w90_wrapper_processing.or. &
        want_dftd3_dispersion.or.spin%DM/=spin%H.or.spin%EDM/=min(spin%H,4)) &
        call die('REMAG resident SCF: unsupported solver/consumer settings')
#ifdef SIESTA__FLOOK
    call die('REMAG resident SCF: dynamic Lua settings unsupported')
#endif
    if (option_charges%mulliken%run(OPT_CHARGE_SCF).or. &
        option_charges%mulliken%run(OPT_CHARGE_SCF_AFTER_MIX)) &
        call die('REMAG resident SCF: in-loop matrix analysis unsupported')
    if (.not.associated(scf_mixs).or..not.associated(scf_mix)) &
        call die('REMAG resident SCF: missing mixer')
    ! restart_save is inert when periodic/residual restart is disabled;
    ! upstream defaults it to one even for a mixer that never restarts.
    if (size(scf_mixs)/=1.or.mix_spin/=MIX_SPIN_ALL.or.scf_mix%m/=mix_method('pulay').or. &
        scf_mix%v/=0.or.scf_mix%n_hist<2.or.scf_mix%n_hist>16.or.scf_mix%n_itt/=0.or. &
        scf_mix%restart/=0.or.associated(scf_mix%next_conv)) &
        call die('REMAG resident SCF: requires one stable all-component Pulay mixer')
    if (.not.associated(scf_mix%rv)) call die('REMAG resident SCF: missing mixer parameters')
    if (lbound(scf_mix%rv,1)>-3.or.ubound(scf_mix%rv,1)<1) &
        call die('REMAG resident SCF: invalid mixer parameters')
    if (scf_mix%rv(-1)>0._c_double.or.scf_mix%rv(-2)>0._c_double) &
        call die('REMAG resident SCF: residual-triggered mixer restart/switch unsupported')
  end subroutine
  subroutine remag_scf_start()
    use precision, only: dp
    use sparse_matrices
    use class_dSpData1D, only: val
    use class_dSpData2D, only: val
    use class_zSpData2D, only: val
    use m_spin, only: spin
    use atomlist, only: no_l
    use m_energies, only: Ena,Eions,Emad
    use siesta_options, only: Temp,converge_EDM,mix_scf_first
    use m_mixing_scf, only: scf_mix
    real(dp),pointer :: kin(:),vkb(:),on(:,:)
    complex(dp),pointer :: off(:,:)
    type(c_ptr) :: on_pointer,off_pointer,old_energy_pointer
    integer :: soc
    call remag_scf_guard()
    kin=>val(H_kin_1D);vkb=>val(H_vkb_1D)
    on_pointer=c_null_ptr;off_pointer=c_null_ptr;soc=0
    old_energy_pointer=c_null_ptr
    if(associated(Eold)) old_energy_pointer=c_loc(Eold(1,1))
    if(spin%SO_onsite) then
      on=>val(H_so_on_2D);on_pointer=c_loc(on(1,1));soc=1
    else if(spin%SO_offsite) then
      off=>val(H_so_off_2D);off_pointer=c_loc(off(1,1));soc=2
    endif
    call remag_dm_check(remag_state_prepare(maxnh,spin%H,no_l,soc,scf_mix%n_hist,scf_mix%w, &
        scf_mix%rv(1),scf_mix%rv(-3),merge(1,0,converge_EDM),merge(1,0,mix_scf_first),numh,listhptr,S,kin,vkb, &
        on_pointer,off_pointer,[Ena,Eions,Emad,Temp],H,Hold,Dscf,Dold,Escf,old_energy_pointer))
    remag_scf_charge=0._c_double
  end subroutine
  subroutine remag_scf_collect(dDmax,dHmax,dEmax,q)
    use m_energies
    use siesta_options, only: converge_EDM
    use m_spin, only: spin
    use sys, only: die
    real(c_double),intent(out) :: dDmax,dHmax,q(:)
    real(c_double),intent(inout) :: dEmax
    real(c_double) :: values(16)
    if(size(q)/=spin%Grid) call die('REMAG resident SCF: invalid charge-vector shape')
    if (E_dftu_so/=0._c_double.or.E_correc_dc/=0._c_double.or.DUext/=0._c_double.or. &
        Ecorrec/=0._c_double.or.Emm/=0._c_double.or.Emeta/=0._c_double.or. &
        Edftu/=0._c_double.or.Evdw_d3/=0._c_double) &
        call die('REMAG resident SCF: unsupported energy correction')
    call remag_dm_check(remag_state_scalars(values))
    Ebs=values(1);DEharr=values(2);Etot=values(3);Eharrs=values(4)
    FreeE=values(5);DEna=values(6);dDmax=values(7);dHmax=values(8)
    if(converge_EDM) dEmax=values(9)
    q=values(10:9+size(q));remag_scf_charge=0._c_double
    remag_scf_charge(1:4)=values(10:13)
    Ekin=values(14);Enl=values(15);Eso=values(16)
    call update_E0()
  end subroutine
end module
'''

def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)

def dm_begin(kind, nh, eh, gamma):
    no, mapping, xyz = ("nuotot", "listd", "remag_zero") if gamma else ("no", "indxuo", "xij")
    return f'''      call remag_dm_serial(nuo,nuotot,Nodes)
      call remag_dm_check(remag_dm_begin({kind},nuotot,maxnd,
     &     {nh},{eh},{no},numd,listdptr,
     &     listd,{mapping},{xyz}))
'''

def dm_accumulate(slot, nev, outspin, occ, energies, k):
    return f'''      call remag_dm_check(remag_dm_accumulate({slot},{nev},
     &     {outspin},{occ},{energies},
     &     occtol,{k}))
'''

def patch_density_matrices(source):
    (source / "Src/remag_dm.F90").write_text(DM_MODULE)
    path = source / "Src/compute_dm.F"
    text = path.read_text()
    text = replace_once(text,"      use precision\n","      use precision\n      use m_remag_dm\n")
    anchor = "      elseif ((isolve .eq. SOLVE_DIAGON) .or. (CallDiagon)) then\n"
    text = replace_once(text,anchor,anchor+'''        if (remag_dm_normalization_enabled()==1.and.
     &      normalize_dm_during_scf.and..not.TSrun
#ifdef CDF
     &      .and..not.(writedmhs_cdf_history.or.writedmhs_cdf)
#endif
     &      ) then
          call remag_dm_check(remag_dm_normalization_arm(maxnh,
     &         spin%DM,spin%EDM,spin%spinor,S,qtot,
     &         dm_normalization_tol))
        endif
''')
    anchor = "         call normalize_dm( first= .false. )"
    text = replace_once(text,anchor,"         if (remag_dm_normalization_complete()==0) then\n"+
                        anchor+"\n         endif")
    path.write_text(text)
    path = source / "Src/normalize_dm.F90"
    text = path.read_text()
    text = replace_once(text, "  subroutine normalize_dm( first )\n", "  subroutine normalize_dm( first )\n" +
                        "    use m_remag_dm, only: remag_dm_enabled, remag_dm_normalize, remag_dm_check\n")
    text = replace_once(text, "!$OMP end parallel\n", "!$OMP end parallel\n" + '''
       if (remag_dm_enabled()==1) then
          call remag_dm_check(remag_dm_normalize(size(Dscf,1),size(Dscf,2), &
               size(Escf,2),scale,Dscf,Escf))
       end if
''')
    path.write_text(text)
    for name in ("diagg", "diagk", "diag2g", "diag3g", "diag2k", "diag3k"):
        path = source / f"Src/{name}.F"
        text = path.read_text()
        text = replace_once(text, "      use precision\n", "      use precision\n      use m_remag_dm\n")
        gamma = name.endswith("g")
        spinor = name[4] in "23"
        nh = ("8" if name.startswith("diag3") else "4") if spinor else "h_spin_dim"
        eh = "4" if spinor else "e_spin_dim"
        if name == "diagk":
            eh = "h_spin_dim"  # e_spin_dim is also imported upstream; scalar dimensions agree.
        kind = (2 if spinor else 0) + (0 if gamma else 1)
        # Only vector solves feeding this density builder are armed. Band-only and
        # wavefunction export preserve their original host-output contract.
        pattern = r"(?m)^([ ]+call [rc]diag\([^\n]*\n(?:[ ]*[.&$][^\n]*\n)+)"
        armed = 0
        def arm(match):
            nonlocal armed
            call = match.group(1)
            if not gamma and "neigneeded" not in call:
                return call
            slot = "ispin" if name == "diagg" else "0"
            nev = "2*neigwanted" if gamma and spinor else "neigwanted" if gamma else "neigneeded"
            armed += 1
            return f'''      if (remag_dm_enabled()==1.and.getD.and.{nev}>0)
     &     call remag_dm_arm({slot},getPSI)
''' + call
        text = re.sub(pattern, arm, text)
        assert armed == 2, (name, armed)
        finish = "      call remag_dm_check(remag_dm_finish(Dnew,Enew))\n"
        if gamma:
            slot = "0" if spinor else "ispin"
            nev = "2*neigwanted" if spinor else "neigwanted"
            occ, energies = ("qo", "eo") if spinor else ("qo(1,ispin)", "eo(1,ispin)")
            body = dm_begin(kind,nh,eh,True)
            if not spinor: body += "      do ispin=1,spinor_dim\n"
            body += dm_accumulate(slot,nev,"0" if spinor else "ispin-1",occ,energies,"remag_zero")
            if not spinor: body += "      enddo\n"
            body += finish
            block = "      if (remag_dm_enabled()==1) then\n      call timer('remag_dm',1)\n" + body + \
                    "      call timer('remag_dm',2)\n      return\n      endif\n"
            anchor = "      Enew(:,:) = 0.0_dp\n" if spinor else "      call timer( 'r-buildD', 1 )\n"
            text = replace_once(text, anchor, anchor + block if spinor else block + anchor)
        else:
            # Begin after upstream initializes the matrices, before the vector loop.
            label = re.search(r"(?m)^ *10 +continue[^\n]*\n",text)
            assert label
            body = "      if (remag_dm_enabled()==1.and.getD) then\n" + dm_begin(kind,nh,eh,False) + "      endif\n"
            text = text[:label.end()] + body + text[label.end():]
            occ, energies = ("qo(1,ik)", "eo(1,ik)") if spinor else ("qo(1,ispin,ik)", "eo(1,ispin,ik)")
            accum = dm_accumulate("0","neigneeded","0" if spinor else "ispin-1",occ,energies,"kpoint(1,ik)")
            if spinor:
                anchor = "if (.not. getD) CYCLE   ! process new k-point"
                assert text.count(anchor) == 1
                text = text.replace(anchor,anchor+"\n      if (remag_dm_enabled()==1) then\n"+
                                    "      call timer('remag_dm',1)\n"+accum+
                                    "      call timer('remag_dm',2)\n      cycle\n      endif\n")
                end = text.rfind("      enddo")
                end = text.index("\n",end)+1
                text = text[:end]+"      if (remag_dm_enabled()==1.and.getD) then\n"+finish+"      endif\n"+text[end:]
            else:
                anchor = "             call timer( 'c-buildD', 1 )\n"
                text = replace_once(text,anchor,anchor+"      if (remag_dm_enabled()==1) then\n"+accum+"      else\n")
                anchor = "            call timer( 'c-buildD', 2 )"
                text = replace_once(text,anchor,"      endif\n"+anchor)
                anchor = "      deallocate(done_juo)"
                text = replace_once(text,anchor,"      if (remag_dm_enabled()==1.and.getD) then\n"+finish+"      endif\n"+anchor)
        path.write_text(text)

INTERFACES = '''
  interface
    integer(c_int) function remag_gpu_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_solve_real(n, neig, ld, h, s, z, w) bind(C)
      import c_int, c_double
      integer(c_int), value :: n, neig, ld
      real(c_double), intent(in) :: h(*), s(*)
      real(c_double), intent(out) :: z(*), w(*)
    end function
    integer(c_int) function remag_solve_complex(n, neig, ld, h, s, z, w) bind(C)
      import c_int, c_double, c_double_complex
      integer(c_int), value :: n, neig, ld
      complex(c_double_complex), intent(in) :: h(*), s(*)
      complex(c_double_complex), intent(out) :: z(*)
      real(c_double), intent(out) :: w(*)
    end function
  end interface
'''

GRID_INTERFACE = '''
  interface
    integer(c_int) function remag_cycle_initial_grid() bind(C)
      import c_int
    end function
    integer(c_int) function remag_grid_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_grid(mode,no,np,nsp,nuo,nnz,ng,nh,nsc,ends,orbs,num,ptr,cols, &
                                     phi,input,output,dvol) bind(C)
      import c_int,c_double,c_float
      integer(c_int),value :: mode,no,np,nsp,nuo,nnz,ng,nh
      integer(c_int),intent(in) :: nsc(*),ends(*),orbs(*),num(*),ptr(*),cols(*)
      real(c_float),intent(in) :: phi(*)
      real(c_double),intent(in) :: input(*)
      real(c_double),intent(inout) :: output(*)
      real(c_double),value :: dvol
    end function
  end interface
'''

def patch_grids(source):
    path = source / "Src/listsc.f"
    text = path.read_text()
    anchor = "      integer, pointer, save :: IND1(:) => null()"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "      integer, save, public :: remag_nsc(3) = 0\n" + anchor)
    text = text.replace("      NO = NUO * NCELLS", "      NO = NUO * NCELLS\n      remag_nsc = NSC")
    path.write_text(text)
    for name in ("vmat", "rhoofd"):
        path = source / f"Src/{name}.F90"
        text = path.read_text()
        text = text.replace("  implicit none", "  use iso_c_binding, only: c_int,c_float,c_double\n  implicit none", 1)
        text = text.replace("\ncontains\n", GRID_INTERFACE + "\ncontains\n", 1)
        text = text.replace("only: LISTSC", "only: LISTSC, remag_nsc")
        if name == "vmat":
            text = text.replace("    use precision,", "    use sys, only: die\n    use precision,", 1)
            call = "remag_grid(1,no,np,nsp,nuo,nvmax,spin%Grid,spin%H,remag_nsc, &\n" + \
                   "          endpht,lstpht,numVs,listVsptr,listVs,phi,V,Vs,dvol)"
        else:
            call = "remag_grid(merge(2,0,remag_cycle_initial_grid()==1),no,np,nsp,nuo,maxnd,nspin,size(Dscf,2),remag_nsc, &\n" + \
                   "          endpht,lstpht,numd,listdptr,listd,phi,Dscf,rhoscf,0._dp)"
        anchor = f"  call timer('{name}',1)"
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + f'''
    if (remag_grid_enabled() == 1) then
       if (Nodes /= 1 .or. DirectPhi) call die('REMAG grid requires serial stored basis values')
       i = {call}
       if (i /= 0) then
          write(*,*) 'REMAG grid error: ', i
          call die('REMAG grid contraction failed')
       end if
       call timer('{name}',2)
       return
    end if
''')
        path.write_text(text)

def patch_execution(source):
    # Upstream calls dhscf_init unconditionally after every geometry change.
    # Invalidate before rebuilding its stored basis, including same-address reuse.
    path = source / "Src/dhscf.F"
    text = path.read_text()
    text = replace_once(text, "      module m_dhscf\n", "      module m_dhscf\n" +
                        "      use m_remag_dm, only: remag_geometry_changed\n")
    anchor = "      call timer( 'DHSCF_Init', 1 )"
    text = replace_once(text, anchor, "      call remag_geometry_changed()\n" + anchor)
    anchor = "            call de_alloc( rhopcc, 'rhopcc', 'dhscf_init' )"
    text = replace_once(text, anchor, "            call remag_geometry_changed()\n" + anchor)
    path.write_text(text)
    path = source / "Src/siesta_forces.F90"
    text = path.read_text()
    anchor = "    use precision, only: dp\n"
    text = replace_once(text, anchor, anchor + "    use m_remag_dm, only: remag_phase_enter, remag_dm_check, remag_defer_scf_log\n")
    for anchor, phase, iteration in (("    call timer('geom_init', 1)", 0, "0"),
            ("    call state_analysis( istep )", 3, "0")):
        text = replace_once(text, anchor,
            f"    call remag_dm_check(remag_phase_enter({phase},max(0,istep),{iteration}))\n" + anchor)
    # Include walltime/FDF checks at the top of every iteration in the I/O audit.
    anchor = "        iscf = iscf + 1"
    text = replace_once(text, anchor, anchor +
                        "\n        if (iscf==1.and.remag_defer_scf_log()==1) call pxfflush(6)"+
                        "\n        call remag_dm_check(remag_phase_enter(1,max(0,istep),iscf))")
    anchor = "    end do ! end of SCF cycle"
    text = replace_once(text, anchor, anchor +
                        "\n    call remag_dm_check(remag_phase_enter(2,max(0,istep),0))"+
                        "\n    if (remag_defer_scf_log()==1) call pxfflush(6)")
    path.write_text(text)
    path = source / "Src/scfconvergence_test.F"
    text = path.read_text()
    anchor = "      use precision, only: dp\n"
    text = replace_once(text, anchor, anchor + "      use m_remag_dm, only: remag_defer_scf_log\n")
    assert text.count("call pxfflush(6)") == 3
    text = text.replace("call pxfflush(6)","if (remag_defer_scf_log()==0) call pxfflush(6)")
    path.write_text(text)
    # Synchronize/flush optional CUDA tracing while the runtime is still alive.
    # C++ process destructors can run after CUDA has already begun unloading.
    path = source / "Src/siesta_end.F"
    text = path.read_text()
    anchor = "      use precision,       only : dp\n"
    text = replace_once(text,anchor,anchor+"      use m_remag_dm, only: remag_trace_finalize\n")
    anchor = "      call fdf_shutdown()"
    text = replace_once(text,anchor,anchor+"\n      call remag_trace_finalize()")
    path.write_text(text)

def patch_occupations(source):
    path = source / "Src/diagon.F"
    text = path.read_text()
    text = replace_once(text,"      use precision\n","      use precision\n      use m_remag_dm\n")
    anchor = "      getPSI = .false.\n"
    text = replace_once(text,anchor,anchor+'''
      if (remag_fermi_enabled()==1) then
        if (fixspin.or.Spiral.or.e1<e2.or.Nodes/=1.or.
     &      diag_wfs_cache/=0) then
          call die('REMAG occupations: unsupported diagonal route')
        endif
        if (spin%none.or.spin%Col) then
          if (gamma) then
            call remag_dm_check(remag_fermi_prepare(nspin,nspin,1,
     &        maxo,neigwanted,eo,qo,remag_one_weight,temp,qtot))
          else
            call remag_dm_check(remag_fermi_prepare(nspin,nspin,nk,
     &        maxo,neigwanted,eo,qo,wk,temp,qtot))
          endif
        elseif (gamma.and.spin%NCol) then
          call remag_dm_check(remag_fermi_prepare(1,1,1,2*maxo,
     &      2*neigwanted,eo,qo,remag_half_weight,temp,qtot))
        elseif (gamma) then
          call remag_dm_check(remag_fermi_prepare(2,1,1,2*maxo,
     &      2*neigwanted,eo,qo,remag_one_weight,temp,qtot))
        else
          call remag_dm_check(remag_fermi_prepare(2,1,nk,2*maxo,
     &      2*neigwanted,eo,qo,wk,temp,qtot))
        endif
      endif
''')
    anchor = "C Free local arrays\n"
    text = replace_once(text,anchor,"      if (remag_fermi_enabled()==1)\n"+
                        "     &  call remag_dm_check(remag_fermi_end())\n\n"+anchor)
    path.write_text(text)
    path = source / "Src/fermid.F"
    text = path.read_text()
    anchor = "      use precision,  only : dp\n"
    text = replace_once(text,anchor,anchor+"      use m_remag_dm, only: remag_fermi_armed,\n"+
                        "     &  remag_fermi_compute, remag_dm_check\n")
    anchor = "      logical, save :: excitd = .false.\n"
    text = replace_once(text,anchor,anchor+"      logical, save :: remag_checked = .false.\n")
    anchor = ("C Zero occupancies, including those not explicitly\nC calculated here if ne < maxe\n"
              "      wke(1:maxe,1:maxspn,1:nk) = 0.0d0\n")
    text = replace_once(text,anchor,'''
      if (remag_fermi_armed()==1) then
        if (.not.remag_checked) then
          if (excitd.or..not.leqi(
     &        fdf_string('OccupationFunction','FD'),'FD')) then
            call die('REMAG occupations require FD, no excitation')
          endif
          remag_checked=.true.
        endif
        call remag_dm_check(remag_fermi_compute(nspinor,maxspn,nk,
     &    maxe,ne,e,wk,temp,qtot,wke,ef,entropy))
        return
      endif

'''+anchor)
    path.write_text(text)

def patch_sparse_state(source):
    (source / "Src/remag_scf.F90").write_text(SCF_MODULE)
    # The existing GPU builder still followed upstream's host zero fills.
    # Host outputs become opaque in this owner and must remain untouched.
    for name in ("diagg","diagk","diag2g","diag3g","diag2k","diag3k"):
        path=source/f"Src/{name}.F";text=path.read_text()
        text=replace_once(text,"      use precision\n", "      use precision\n      use m_remag_scf, only: remag_state_active\n")
        pattern=r"(?m)^ *Dnew\([^\n]+ = 0\.0(?:d0|_dp)\n *Enew\([^\n]+ = 0\.0(?:d0|_dp)\n"
        text,count=re.subn(pattern,lambda m:"      if (remag_state_active()==0) then\n"+m.group()+"      endif\n",text)
        assert count==1,(name,count)
        path.write_text(text)
    path=source/"Src/compute_dm.F"
    text=path.read_text()
    text=replace_once(text,"      use precision\n","      use precision\n      use m_remag_scf, only: remag_state_active\n")
    text=replace_once(text,"      if (SIESTA_worker) then\n        ! Save present density matrix", 
                     "      if (SIESTA_worker.and.remag_state_active()==0) then\n        ! Save present density matrix")
    path.write_text(text)
    path=source/"Src/setup_hamiltonian.F"
    text=path.read_text()
    anchor="      use option_charges_m, only: OPT_CHARGE_SCF\n"
    text=replace_once(text,anchor,anchor+"      use m_remag_scf\n")
    start=text.index('!$OMP parallel default(shared), private(ispin,io)')
    end=text.index('      call update_E0()',start)+len('      call update_E0()')
    original=text[start:end]
    text=replace_once(text,original,'''
      if (remag_state_active()==1) then
        call remag_dm_check(remag_state_seed())
      else
'''+original+'''
      endif
''')
    start=text.index('      if ( spin%SO_offsite ) then',text.index('      call dhscf('))
    end=text.index('! Energy term coming from the chemical potential',start)
    original=text[start:end]
    text=replace_once(text,original,'''
      if (remag_state_active()==1) then
        call remag_dm_check(remag_state_finish_hamiltonian())
      else
'''+original+'''      endif

''')
    anchor="      call timer('setup_H',2)"
    text=replace_once(text,anchor,'''
      if (remag_state_enabled()==1.and.iscf==0)
     &  call remag_scf_start()
'''+anchor)
    path.write_text(text)
    path=source/"Src/siesta_forces.F90"
    text=path.read_text();anchor="    use precision, only: dp\n"
    text=replace_once(text,anchor,anchor+"    use m_remag_scf\n")
    anchor="        call remag_dm_check(remag_phase_enter(1,max(0,istep),iscf))"
    text=replace_once(text,anchor,anchor+'''
        if (remag_state_enabled()==1) call remag_dm_check(remag_state_begin(iscf))''')
    start=text.index('          if ( mixH ) then\n             \n             call compute_DM')
    end=text.index('          if ( mix_charge ) then\n             call compute_charge_diff',start)
    original=text[start:end]
    text=replace_once(text,original,'''          if (remag_state_active()==1) then
             call compute_DM(iscf)
             call setup_hamiltonian(iscf)
             call remag_scf_collect(dDmax,dHmax,dEmax,Qcur)
             if (spin%Grid>1) Qcur(1)=Qcur(1)+Qcur(2)
             dQ=Qcur(1)-Qtot
          else
'''+original+'''          endif
''')
    anchor="    call remag_dm_check(remag_phase_enter(2,max(0,istep),0))"
    text=replace_once(text,anchor,anchor+'''
    if (remag_state_enabled()==1) call remag_dm_check(remag_state_export())''')
    path.write_text(text)
    path=source/"Src/mixer.F"
    text=path.read_text();anchor="      use precision,        only: dp\n"
    text=replace_once(text,anchor,anchor+"      use m_remag_scf\n")
    anchor="      call timer( 'MIXER', 1 )"
    text=replace_once(text,anchor,anchor+'''
      if (remag_state_active()==1) then
        call remag_dm_check(remag_state_mix())
        call timer('MIXER',2)
        return
      endif''')
    path.write_text(text)
    path=source/"Src/scfconvergence_test.F"
    text=path.read_text();anchor="      use precision, only: dp\n"
    text=replace_once(text,anchor,anchor+"      use m_remag_scf, only: remag_state_active,\n     &  remag_scf_charge\n")
    anchor='''         call dm_charge(spin, no_l, numh, listhptr,
     &               S, Dscf, dummy_qspin)'''
    text=replace_once(text,anchor,'''         if(remag_state_active()==1) then
           dummy_qspin=remag_scf_charge
         else
'''+anchor+'''
         endif''')
    path.write_text(text)

def patch_spectral_residency(source):
    # cuSOLVER already computes the full spectrum. Avoid a host occupation scan
    # or a new band-count readback by retaining every requested vector. The
    # device density builder applies the original per-state occupation cutoff.
    for name in ("diagk", "diag2k", "diag3k"):
        path=source/f"Src/{name}.F"
        text=path.read_text()
        pattern=r"(?m)^([ ]+)if \(getPSI\) then\n((?:[ ]*!.*\n)+[ ]+neigneeded = [^\n]+\n)"
        text,count=re.subn(pattern,r"\1if (getPSI.or.remag_spectra_armed()==1) then\n\2",text)
        assert count==1,(name,count)
        path.write_text(text)
    path=source/"Src/siesta_forces.F90"
    text=path.read_text()
    anchor="    use precision, only: dp\n"
    text=replace_once(text,anchor,anchor+
        "    use m_eo, only: remag_eo => eo, remag_qo => qo\n"+
        "    use m_remag_dm, only: remag_spectra_enabled, remag_fermi_export\n")
    anchor="    call remag_dm_check(remag_phase_enter(2,max(0,istep),0))"
    text=replace_once(text,anchor,anchor+'''
    if (remag_spectra_enabled()==1) then
       call remag_dm_check(remag_fermi_export(size(remag_eo),remag_eo,remag_qo))
    endif''')
    path.write_text(text)

def patch_matrix_assembly(source):
    path=source/"Src/diagon.F"
    text=path.read_text()
    anchor="      getPSI = .false.\n"
    text=replace_once(text,anchor,anchor+'''
      if (remag_hs_enabled()==1) then
        if (Spiral.or.Nodes/=1.or.diag_wfs_cache/=0)
     &    call die('REMAG matrix assembly: unsupported route')
        call remag_dm_check(remag_hs_prepare(
     &    merge(0,2,spin%none.or.spin%Col)+merge(0,1,gamma),
     &    nuotot,maxnh,size(H,2),no,merge(1,nk,gamma),size(H,1),
     &    numh,listhptr,listh,indxuo,xij,kpoint,H,S))
      endif
''')
    anchor="C Free local arrays\n"
    text=replace_once(text,anchor,"      if (remag_hs_enabled()==1)\n"+
                      "     &  call remag_dm_check(remag_hs_end())\n\n"+anchor)
    path.write_text(text)
    for name in ("diagg","diagk","diag2g","diag2k","diag3g","diag3k"):
        path=source/f"Src/{name}.F"
        lines=path.read_text().splitlines(keepends=True)
        regions=[]
        # Keep the pinned CPU fallback verbatim, including OpenMP directives,
        # but bypass every numerical H/S assembly loop while a GPU bank is armed.
        for i,line in enumerate(lines):
            if not re.match(r"^\s*do (?:io|iuo)\s*=\s*1\s*,\s*nuo\s*$",line,re.I):
                continue
            depth=0
            for j in range(i,len(lines)):
                if re.match(r"^\s*do\b",lines[j],re.I): depth+=1
                if re.match(r"^\s*end\s*do\b",lines[j],re.I): depth-=1
                if depth==0: break
            body="".join(lines[i:j+1])
            if "Saux" not in body or "Haux" not in body: continue
            start,end=i,j+1
            while start>0:
                previous=lines[start-1]
                if (not previous.strip() or previous.lstrip().startswith("!$OMP") or
                    re.match(r"^\s*[SH]aux\s*=\s*cmplx\(0",previous,re.I)):
                    start-=1
                else: break
            if end<len(lines) and lines[end].lstrip().startswith("!$OMP end parallel do"):
                end+=1
            regions.append((start,end))
        expected=2 if name.endswith("g") else 3
        assert len(regions)==expected,(name,regions)
        assert all(a[1]<=b[0] for a,b in zip(regions,regions[1:])),(name,regions)
        for start,end in reversed(regions):
            lines[start:end]=["      if (remag_hs_armed()==0) then\n",*lines[start:end],"      endif\n"]
        text="".join(lines)
        spin="ispin-1" if name in ("diagg","diagk") else "0"
        k="0" if name.endswith("g") else "ik-1"
        pattern=r"(?m)^([ ]+call [rc]diag\()"
        text,count=re.subn(pattern,
            f"      if (remag_hs_armed()==1)\\n     &  call remag_dm_check(remag_hs_select({spin},{k},\\n"+
            r"     &    Haux,Saux))\n\1",text)
        assert count==expected,(name,count)
        path.write_text(text)

def patch_force_shadow(source):
    template=Path(__file__).resolve().parents[1]/'native/remag_force.F90.in'
    if not template.is_file():template=Path('/opt/remag/native/remag_force.F90.in')
    (source/'Src/remag_force.F90').write_text(template.read_text())
    path=source/'Src/dfscf.f';text=path.read_text()
    anchor='      use precision,     only: dp, grid_p\n'
    text=replace_once(text,anchor,anchor+'      use m_remag_force\n'+
        '      use m_remag_dm, only: remag_dm_check\n')
    anchor='      type(allocDefaults) oldDefaults\n'
    text=replace_once(text,anchor,anchor+'      logical :: remag_force_on, remag_force_gpu\n')
    anchor="      call timer('dfscf',1)\n"
    text=replace_once(text,anchor,anchor+'''
      remag_force_on = remag_force_active()==1
      remag_force_gpu = remag_grid_force_enabled()==1
      if (remag_force_on) then
        if (Nodes/=1.or.nuo/=nuotot.or.grid_p/=dp)
     &    call die('REMAG force shadow requires serial FP64 grids')
        call remag_dm_check(remag_force_shadow_begin(no,nuotot,
     &    np,nsp,maxnd,size(Dscf,2),maxval(indxua),endpht(np),
     &    indxua(iaorb),Dscf,Datm,Vscf,Vatm,dvol,VolCel,ifa,istr,
     &    Fal,Stressl))
        if (remag_orbital_basis_enabled()==1) then
          call timer('remag_grid_force',1)
          call remag_dm_check(remag_force_shadow_finish(Fal,Stressl))
          call timer('remag_grid_force',2)
          call timer('dfscf',2)
          return
        endif
      endif
''')
    anchor='          gC(1:3,1:nsp,ic) = grada(1:3,iphi,1:nsp)\n'
    text=replace_once(text,anchor,anchor+'''
          if (remag_force_on)
     &      call remag_dm_check(remag_force_shadow_orbital(imp,
     &        C(:,ic),gC(:,:,ic),dxsp))
''')
    anchor='C  Deallocate local memory\n'
    text=replace_once(text,anchor,'''      if (remag_force_on) then
        call timer('remag_grid_force',1)
        call remag_dm_check(remag_force_shadow_finish(Fal,Stressl))
        call timer('remag_grid_force',2)
      endif

'''+anchor)
    # Production mode keeps the exact FP64 basis/derivative generation, but
    # removes host electronic contraction, sparse DM caching and SOC copies.
    def cpu_region(start,end):
        nonlocal text
        a=text.index(start);b=text.index(end,a)
        text=text[:a]+'      if (.not.remag_force_gpu) then\n'+text[a:b]+'      endif\n'+text[b:]
    cpu_region('      if ( size(Dscf, 2) == 8 ) then','C  Allocate buffers to store partial copies')
    cpu_region("      call re_alloc( D, 0, maxb", "      call re_alloc( gC, 1, 3")
    cpu_region("      call re_alloc( ibc, 1, maxc", 'C  Set logical that determines')
    cpu_region('      D(:,:,:) = 0.0_dp','C  Find atomic cutoff radii')
    cpu_region('C  iob(ib)>0 means','C  Calculate all phi values')
    text=replace_once(text,'          if (istr.eq.1) then\n            do isp',
                     '          if (istr.eq.1.and..not.remag_force_gpu) then\n            do isp')
    cpu_region('C     Copy potential to a double precision array','C  End of mesh point loop')
    for line in ("      call de_alloc( xgC, 'xgC', 'dfscf' )", "      call de_alloc( iob, 'iob', 'dfscf' )",
                 "      call de_alloc( ibc, 'ibc', 'dfscf' )", "      call de_alloc( D, 'D', 'dfscf' )",
                 "      call de_alloc( V, 'V', 'dfscf' )"):
        text=replace_once(text,line,'      if (.not.remag_force_gpu)\n     &'+line[6:])
    text=replace_once(text,'      if (size(Dscf,2) == 8) then','      if (size(Dscf,2)==8.and..not.remag_force_gpu) then')
    path.write_text(text)


def patch_atomic_force_shadow(source):
    path=source/'Src/meshsubs.F';text=path.read_text()
    for kind,name,nsd,anchor,gradient in (
            (1,'PartialCoreOnMesh','nsd','      Gradients = ( Forces .or. Stress )','grrho'),
            (2,'NeutralAtomOnMesh','1','      Gradients = (Forces.or.Stress)','grva')):
        start=text.index('      subroutine '+name+'(')
        end=text.index('      end subroutine '+name,start)
        part=text[start:end]
        part=replace_once(part,'      use precision, only: dp, grid_p',
            '      use m_remag_force\n      use m_remag_dm, only: remag_dm_check\n'+
            '      use parallel, only: Nodes\n      use mesh, only: nmeshg, nsm\n'+
            '      use precision, only: dp, grid_p')
        declarations=[line for line in part.splitlines() if line.strip().endswith(':: Gradients')]
        assert len(declarations)==1
        part=replace_once(part,declarations[0],declarations[0]+', remag_atomic_on,\n     &        remag_atomic_gpu, remag_atomic_basis_on')
        part=replace_once(part,anchor,anchor+f'''
      remag_atomic_gpu = remag_atomic_grid_force_enabled()==1
      remag_atomic_on = Gradients.and.
     &                 (remag_atomic_force_active()==1)
      remag_atomic_basis_on = remag_atomic_on.and.
     &                      (remag_atomic_basis_shadow_enabled()==1)
      if (remag_atomic_on) then
        if (Nodes/=1.or.grid_p/=dp)
     &    call die('REMAG atomic forces require serial FP64 grids')
        call remag_dm_check(remag_atomic_force_begin({kind},ntpl,
     &    maxval(indxua(1:na)),{nsd},nmeshg,nsm,dvol,volume,
     &    merge(1,0,Forces),merge(1,0,Stress),fal,stressl))
        if (remag_atomic_basis_device_enabled()==1) goto 991
      endif
''')
        # Preserve the original radial evaluation and nongradient density /
        # potential construction; skip only the CPU electronic contraction.
        a=part.index('if (Gradients) then');a=part.index('\n',a)+1
        b=part.index('                  else' if kind==1 else '                else',a)
        part=part[:a]+'                  if (.not.remag_atomic_gpu) then\n'+part[a:b]+'                  endif\n'+part[b:]
        call=('                  call chcore_sub( is, dx, rhop, grrho )' if kind==1 else
              '                call phiatm( is, 0, dx, va, grva )')
        part=replace_once(part,call,call+f'''
                  if (remag_atomic_on)
     &              call remag_dm_check(remag_atomic_force_point(
     &                ip,iua,{gradient},dx))
                  if (remag_atomic_basis_on)
     &              call remag_dm_check(remag_atomic_basis_point(
     &                {kind},ia,iop,isp,ip,iua,{'rhop' if kind==1 else 'va'},
     &                {gradient},dx))
''')
        anchor=f"#ifdef DEBUG\n      call write_debug( '      POS {name}' )"
        part=replace_once(part,anchor,'''  991 continue
      if (remag_atomic_on) then
        call timer('remag_atomic_force',1)
        call remag_dm_check(remag_atomic_force_finish(fal,stressl))
        call timer('remag_atomic_force',2)
      endif
'''+anchor)
        text=text[:start]+part+text[end:]
    path.write_text(text)


def patch_final_grid(source):
    path=source/'Src/dhscf.F';text=path.read_text()
    anchor='      use gridXC, only : cellXC=>gridxc_cellXC       ! Finds xc energy and potential\n'
    text=replace_once(text,anchor,anchor+'      use m_remag_force\n')
    anchor='      logical :: use_rhog\n'
    text=replace_once(text,anchor,anchor+'''      logical :: remag_final_call
      real(dp) :: remag_unused_grid(1,4)
''')
    anchor='      volume = volcel(cell)\n'
    # Reuse the same restricted grid configuration guard as the resident
    # cycle, adapting only its force switches and the final H target.
    a=text.index('        if (Nodes/=1',text.index('      if (remag_flow_active()==1) then'))
    b=text.index('        remag_setup=remag_flow_needs_setup',a)
    guard=text[a:b].replace('ifa/=0.or.istr/=0.or.iHmat/=1','ifa/=1.or.istr/=1.or.iHmat/=1')
    branch='''
C     Final force-grid handoff. Exact radial/orbital derivatives below
C     still run on CPU; their electronic inputs are exclusively device-owned.
      remag_final_call=remag_final_grid_active()==1.and.
     &                 (ifa/=0.or.istr/=0)
      if (remag_final_call) then
'''+guard+'''        if(harrisfun.or.fdf_defined('Grid.CellSampling'))
     &    call die('REMAG final grid: Harris/sampling unsupported')
        if(remag_final_grid_enabled()==1) then
          call remag_dm_check(remag_final_grid(maxnh,size(Hmat,2),
     &      nspin,ntm,nsm,npcc,dvol,volume,Dscf,Hmat,remag_values,
     &      DStres,stressXC))
          Uscf=remag_values(1);DUscf=remag_values(2)
          Uatm=remag_values(3);Enaatm=remag_values(4)
          Enascf=remag_values(5);Ex=remag_values(6)
          Ec=remag_values(7);DEx=remag_values(8)
          DEc=remag_values(9);Exc=Ex+Ec;Dxc=DEx+DEc
          DUext=0._dp;dipol=0._dp
          stress=stress+stressXC
          stressl=stressl+DStres
          do i=1,3
            stress(i,i)=stress(i,i)+(Enascf-Enaatm)/volume
          enddo
          call timer('DHSCF3',2)
          call timer('DHSCF4',1)
C         These dummies are never read by the authoritative GPU adapters.
C         The original gradient-only loops use geometry/radial tables, and
C         all contractions borrow retained device fields. No grid reorder,
C         density/Hartree/XC/matrix recomputation or electronic upload here.
          remag_unused_grid=0._dp
          nsd=min(nspin,2)
          if(npcc==1) call PartialCoreOnMesh(na,isa,ntpl,rhopcc,
     &      indxua,nsd,dvol,volume,remag_unused_grid,
     &      remag_unused_grid,Fal,stressl,.true.,.true.)
          call NeutralAtomOnMesh(na,isa,ntpl,Vna,indxua,dvol,
     &      volume,remag_unused_grid,Fal,stressl,.true.,.true.)
          call dfscf(ifa,istr,na,norb,nuo,nuotot,nmpl,nspin,
     &      indxua,isa,iaorb,iphorb,maxnd,numd,listdptr,listd,
     &      Dscf,Datm,remag_unused_grid,remag_unused_grid,
     &      dvol,volume,Fal,stressl)
          call timer('DHSCF4',2)
          call timer('DHSCF',2)
          return
        endif
      endif
'''
    text=replace_once(text,anchor,anchor+branch)
    anchor="C     Stop time counter for SCF iteration part\n"
    text=replace_once(text,anchor,'''      if(remag_final_call.and.remag_final_grid_shadow_enabled()==1)
     &  then
        remag_values=(/Uscf,DUscf,Uatm,Enaatm,Enascf,
     &                 Ex,Ec,DEx,DEc/)
        call remag_dm_check(remag_final_grid(maxnh,size(Hmat,2),
     &    nspin,ntm,nsm,npcc,dvol,volume,Dscf,Hmat,remag_values,
     &    DStres,stressXC))
      endif

'''+anchor)
    path.write_text(text)


def patch_final_density(source):
    path=source/'Src/dhscf.F';text=path.read_text()
    text=replace_once(text,'      logical :: remag_final_call','      logical :: remag_final_call, remag_density_call')
    text=replace_once(text,'      use siesta_options, only: savebader','      use siesta_options, only: savebader, tdsaverho')
    a=text.index('        if (Nodes/=1',text.index('      if (remag_flow_active()==1) then'))
    b=text.index('        remag_setup=remag_flow_needs_setup',a)
    guard=text[a:b].replace('ifa/=0.or.istr/=0.or.iHmat/=1','ifa/=0.or.istr/=0.or.iHmat/=0')
    guard=guard.replace("filesOut%rho/=' '.or.filesOut%tdrho/=' '","filesOut%tdrho/=' '")
    guard=guard.replace("filesOut%drho/=' '.or.","")
    a=text.index("      if (filesOut%rho .ne. ' ') then")
    b=text.index('C-----------------------------------------------------------------------\nC Save TD-electron density',a)
    write_rho=text[a:b]
    a=text.index("      if (filesOut%drho .ne. ' ') then")
    b=text.index('      if (present(charge_density_only)) then',a)
    write_drho=text[a:b]
    branch='''
C     Requested final RHO/DRHO output. Numerical fields stay on GPU;
C     download each requested sequential field once for the original writer.
      remag_density_call=remag_final_density_active()==1.and.
     &                   ifa==0.and.istr==0.and.iHmat==0
      if(remag_density_call) then
'''+guard+'''        if((filesOut%rho/=' '.and.
     &      filesOut%rho/=trim(slabel)//'.RHO').or.
     &     (filesOut%drho/=' '.and.
     &      filesOut%drho/=trim(slabel)//'.DRHO').or.
     &     (filesOut%rho==' '.and.filesOut%drho==' ').or.
     &      tdsaverho.or.harrisfun.or.fdf_defined('Grid.CellSampling'))
     &    call die('REMAG density requires final RHO/DRHO output')
        if(remag_final_density_enabled()==1) then
          call re_alloc(DRho,1,ntpl,1,nspin,'DRho','dhscf')
          if(filesOut%rho/=' ') then
            call remag_dm_check(remag_final_density(1,maxnd,
     &        size(Dscf,2),nspin,ntm,nsm,Dscf,DRho))
'''+write_rho+'''          endif
          if(filesOut%drho/=' ') then
            call remag_dm_check(remag_final_density(2,maxnd,
     &        size(Dscf,2),nspin,ntm,nsm,Dscf,DRho))
'''+write_drho+'''          endif
          call de_alloc(DRho,'DRho','dhscf')
          call timer('DHSCF3',2)
          call timer('DHSCF',2)
          return
        endif
      endif
'''
    text=replace_once(text,'      volume = volcel(cell)\n','      volume = volcel(cell)\n'+branch)
    for kind,label,field in ((1,'electron density','rho'),(2,'electron density difference','drho')):
        anchor=f'C Save {label}\nC ----------------------------------------------------------------------\n'
        text=replace_once(text,anchor,anchor+f'''      if(remag_density_call.and.filesOut%{field}/=' '.and.
     &   remag_final_density_shadow_enabled()==1)
     &  call remag_dm_check(remag_final_density({kind},maxnd,
     &    size(Dscf,2),nspin,ntm,nsm,Dscf,DRho))
''')
    path.write_text(text)


def patch_hartree(source):
    path=source/"Src/poison.F"
    text=path.read_text()
    anchor="      use precision,   only : dp, grid_p\n"
    text=replace_once(text,anchor,anchor+"      use iso_c_binding, only: c_loc\n"+
                     "      use m_remag_dm, only: remag_dm_check,\n"+
                     "     &  remag_hartree_enabled, remag_hartree_needs_setup,\n"+
                     "     &  remag_hartree_prepare, remag_hartree_solve\n")
    text=replace_once(text,"      real(grid_p)          :: RHO(N1*N2*N3), V(N1*N2*N3)",
                     "      real(grid_p), target  :: RHO(N1*N2*N3), V(N1*N2*N3)")
    anchor="      call timer( 'POISON', 1 )"
    text=replace_once(text,anchor,anchor+'''
      if (remag_hartree_enabled()==1) then
        if (Nodes/=1.or.any(Mesh/=(/N1,N2,N3/)).or.grid_p/=dp)
     &    call die('REMAG Hartree requires serial full FP64 grid')
        I=remag_hartree_needs_setup(Mesh,CELL)
        if (I<0) call remag_dm_check(I)
        if (I==1) then
          VOLUME=VOLCEL(CELL)
          call reclat(CELL,B,1)
          G2MAX=1.e30_dp
          call CHKGMX(K0,B,Mesh,G2MAX)
          call remag_dm_check(remag_hartree_prepare(
     &      Mesh,CELL,B,G2MAX,VOLUME))
        endif
        call timer('remag_hartree',1)
        call remag_dm_check(remag_hartree_solve(
     &    Mesh,CELL,c_loc(RHO),U,c_loc(V),STRESS))
        call timer('remag_hartree',2)
        call timer('POISON',2)
        return
      endif
''')
    path.write_text(text)

def patch_resident_grid(source):
    # Read-only option queries preserve the original optional geometry paths.
    for kind in ('charge','hartree'):
        path=source/f'Src/m_{kind}_add.F90';text=path.read_text()
        text=replace_once(text,f'  public :: {kind}_add\n',f'  public :: {kind}_add, remag_{kind}_present\n')
        text=replace_once(text,'\ncontains\n',f'''\ncontains
  logical function remag_{kind}_present()
    remag_{kind}_present = N_geom > 0
  end function
''')
        path.write_text(text)
    path=source/'Src/dhscf.F';text=path.read_text()
    anchor='      use gridXC, only : cellXC=>gridxc_cellXC       ! Finds xc energy and potential\n'
    text=replace_once(text,anchor,anchor+'''      use gridXC, only: remag_getxc=>gridxc_getXC
      use cellsubs, only: remag_reclat=>reclat
      use m_chkgmx, only: chkgmx
      use m_remag_dm, only: remag_flow_active, remag_flow_needs_setup,
     &  remag_flow_prepare, remag_flow_arm, remag_flow_solve,
     &  remag_flow_end, remag_dm_check
''')
    text=replace_once(text,'      use m_charge_add,   only: charge_add','      use m_charge_add,   only: charge_add, remag_charge_present')
    text=replace_once(text,'      use m_hartree_add,  only: hartree_add','      use m_hartree_add,  only: hartree_add, remag_hartree_present')
    text=replace_once(text,'      use m_ts_hartree, only: ts_hartree_fix','      use m_ts_hartree, only: ts_hartree_fix, TS_HA_PLANES')
    anchor='      logical :: use_rhog\n'
    text=replace_once(text,anchor,anchor+'''      integer :: remag_setup, remag_nf, remag_axis
      real(dp) :: remag_b(3,3), remag_g(3,3), remag_step(3,3),
     &  remag_cutoff, remag_zero(3), remag_values(9),
     &  remag_wx(1), remag_wc(1)
      character(len=50) :: remag_family(1), remag_author(1)
''')
    anchor='      volume = volcel(cell)\n'
    text=replace_once(text,anchor,anchor+'''
C     Resident periodic PBE path, only during declared SCF iterations.
C     Final force/analysis consumers retain their original path for now.
      if (remag_flow_active()==1) then
        if (Nodes/=1.or.grid_p/=dp.or.any(ntml/=ntm).or.
     &      ntpl/=product(ntm).or.Spiral.or.nbcell/=3.or.
     &      ifa/=0.or.istr/=0.or.iHmat/=1.or.debug_dhscf.or.
     &      mix_charge.or.use_bsc_cellxc.or.doping_active.or.
     &      acting_efield.or.TSmode.or.any(TS_HA_PLANES).or.
     &      SLABDIPOLECORR/=SLABDIPOLECORR_NONE.or.
     &      mm_charges%n>0.or.
     &      remag_charge_present().or.remag_hartree_present().or.
     &      analyze_charge_density_only.or.
     &      save_initial_charge_density.or.
     &      read_charge_cdf.or.read_deformation_charge_cdf.or.
     &      want_partial_charges/=0)
     &    call die('REMAG resident grid: unsupported DHSCF settings')
        if (present(use_rhog_in)) then
          if(use_rhog_in)
     &      call die('REMAG grid: reciprocal density input')
        endif
        if (present(charge_density_only)) then
          if(charge_density_only)
     &      call die('REMAG grid: density-only call')
        endif
        if (filesOut%rho/=' '.or.filesOut%tdrho/=' '.or.
     &      filesOut%drho/=' '.or.filesOut%rhoxc/=' '.or.
     &      filesOut%psch/=' '.or.filesOut%toch/=' '.or.
     &      filesOut%vh/=' '.or.filesOut%vt/=' '.or.filesOut%vna/=' ')
     &    call die('REMAG resident grid requires deferred grid exports')
#ifdef SIESTA__ELSI
        call die('REMAG resident grid: ELSI export unsupported')
#endif
        remag_setup=remag_flow_needs_setup(ntm,nspin,nsm,npcc,
     &    cell,rhoatm,rhopcc,Vna)
        if(remag_setup<0) call remag_dm_check(remag_setup)
        if(remag_setup==1) then
          call remag_getxc(n=remag_nf)
          if(remag_nf/=1) call die('REMAG grid requires single PBE')
          call remag_getxc(func=remag_family,auth=remag_author,
     &      wx=remag_wx,wc=remag_wc)
          if(ucase(remag_family(1))/='GGA'.or.
     &       ucase(remag_author(1))/='PBE'.or.
     &       remag_wx(1)/=1._dp.or.remag_wc(1)/=1._dp)
     &      call die('REMAG grid requires built-in unit-weight PBE')
          call remag_reclat(cell,remag_b,1)
          remag_cutoff=1.e30_dp;remag_zero=0._dp
          call chkgmx(remag_zero,remag_b,ntm,remag_cutoff)
          do remag_axis=1,3
            remag_step(:,remag_axis)=cell(:,remag_axis)/ntm(remag_axis)
          enddo
          call remag_reclat(remag_step,remag_g,0)
          call remag_dm_check(remag_flow_prepare(ntm,nspin,nsm,npcc,
     &      cell,rhoatm,rhopcc,Vna,remag_b,remag_cutoff,remag_g,
     &      volume,dvol,Uharrs))
        endif
        call re_alloc(DRho_par,1,ntpl,1,nspin,'DRho_par','dhscf')
        call re_alloc(Vscf_par,1,ntpl,1,nspin,'Vscf_par','dhscf')
        call remag_dm_check(remag_flow_arm(ntpl,nspin,
     &    DRho_par,Vscf_par))
        call rhoofd(norb,nmpl,maxnd,numd,listdptr,listd,nspin,Dscf,
     &    DRho_par,nuo,nuotot,iaorb,iphorb,isa)
        call timer('remag_grid_flow',1)
        call remag_dm_check(remag_flow_solve(DRho_par,Vscf_par,
     &    remag_values,DStres,stressXC))
        call timer('remag_grid_flow',2)
        Uscf=remag_values(1);DUscf=remag_values(2)
        Uatm=remag_values(3);Enaatm=remag_values(4)
        Enascf=remag_values(5);Ex=remag_values(6);Ec=remag_values(7)
        DEx=remag_values(8);DEc=remag_values(9)
        Exc=Ex+Ec;Dxc=DEx+DEc;DUext=0._dp;dipol=0._dp
        stress=stress+stressXC
        call vmat(norb,nmpl,dvol,spin,Vscf_par,maxnd,numd,listdptr,
     &    listd,Hmat,nuo,nuotot,iaorb,iphorb,isa)
        call remag_dm_check(remag_flow_end())
        call de_alloc(DRho_par,'DRho_par','dhscf')
        call de_alloc(Vscf_par,'Vscf_par','dhscf')
        call timer('DHSCF3',2)
        call timer('DHSCF',2)
        return
      endif
''')
    path.write_text(text)

def patch_radial_tables(source):
    root=source/'Src'
    p=root/'matel_table.F90';s=p.read_text()
    s=s.replace('    use interpolation, only: spline','    use iso_c_binding, only: c_int, c_double\n    use interpolation, only: spline',1)
    anchor='    logical, external :: propor'
    declaration='''
    real(dp), allocatable :: radial_bank(:,:,:)
    integer(c_int), allocatable :: radial_l1(:), radial_l2(:)
    integer :: radial_maxl, radial_status, radial_length
    logical :: radial_enabled
    character(len=16) :: radial_option
    interface
      integer(c_int) function radial_batch(nr,n1,n2,lmax,mode,rmax,dq,l1,l2,a,b,g) &
          bind(C,name="remag_radial_batch")
        import c_int,c_double
        integer(c_int), value :: nr,n1,n2,lmax,mode
        real(c_double), value :: rmax,dq
        integer(c_int), intent(in) :: l1(*),l2(*)
        real(c_double), intent(in) :: a(*),b(*)
        real(c_double), intent(out) :: g(*)
      end function
    end interface
'''
    assert s.count(anchor)==1;s=s.replace(anchor,anchor+declaration)
    anchor='    ! Iterate all combinations of harmonics'
    setup='''    call get_environment_variable('REMAG_RADIAL_BATCH',radial_option,length=radial_length,status=radial_status)
    radial_enabled=radial_status==0.and.radial_length==1.and.radial_option=='1'.and.nlocal>0
    if (radial_enabled) then
      if (Nodes/=1.or.size(this%SpHa1%F,1)/=NQ+1.or.size(this%SpHa2%F,1)/=NQ+1) &
        call die('REMAG radial batch requires serial table setup and the pinned radial grid')
      allocate(radial_l1(dim1),radial_l2(dim2))
      do i=1,dim1
        radial_l1(i)=LOFILM(this%SpHa1%ILM(i+this%BASE_SPHA1))
      enddo
      do i=1,dim2
        radial_l2(i)=LOFILM(this%SpHa2%ILM(i+this%BASE_SPHA2))
      enddo
      radial_maxl=maxval(radial_l1)+maxval(radial_l2)
      allocate(radial_bank(0:NQ,0:radial_maxl,first:last))
      radial_status=radial_batch(NQ,dim1,dim2,radial_maxl,this%MODE,NQ*PI/RMAX,DQ,radial_l1,radial_l2, &
        this%SpHa1%F(:,this%BASE_SPHA1+1:this%BASE_SPHA1+dim1), &
        this%SpHa2%F(:,this%BASE_SPHA2+1:this%BASE_SPHA2+dim2),radial_bank)
      if (radial_status/=0) call die('REMAG radial batch failed')
    endif

'''
    assert s.count(anchor)==1;s=s.replace(anchor,setup+anchor)
    anchor='        call RADFFT(L3, NQ, NQ*PI/RMAX, FFQ, FFL)'
    assert s.count(anchor)==1;s=s.replace(anchor,'''        if (radial_enabled) then
          FFL(:)=radial_bank(:,L3,i)
        else
'''+anchor+'''
        endif''')
    p.write_text(s)

def patch_radial_compaction(source):
    root=source/'Src'
    p=root/'matel_table.F90';s=p.read_text()
    def change(a,b):
        nonlocal s
        assert s.count(a)==1,(a[:80],s.count(a));s=s.replace(a,b)
    change('    use iso_c_binding, only: c_int, c_double', '    use iso_c_binding, only: c_int, c_double, c_ptr')
    change('    real(dp), allocatable :: radial_bank(:,:,:)', '''    real(dp), allocatable :: radial_values(:,:), radial_coefficients(:,:)
    integer(c_int), allocatable :: radial_indices(:,:)
    integer(c_int) :: radial_count
    type(c_ptr) :: radial_owner''')
    a=s.index('      integer(c_int) function radial_batch(');b=s.index('    end interface',a)
    s=s[:a]+'''      integer(c_int) function compact_begin(nr,n1,n2,lmax,mode,rmax,dq,dr,tol,l1,l2,a,b,n,owner) &
          bind(C,name="remag_radial_compact_begin")
        import c_int,c_double,c_ptr
        integer(c_int), value :: nr,n1,n2,lmax,mode
        real(c_double), value :: rmax,dq,dr,tol
        integer(c_int), intent(in) :: l1(*),l2(*)
        real(c_double), intent(in) :: a(*),b(*)
        integer(c_int), intent(out) :: n
        type(c_ptr), intent(out) :: owner
      end function
      integer(c_int) function compact_export(owner,values,indices,coefficients) &
          bind(C,name="remag_radial_compact_export")
        import c_int,c_double,c_ptr
        type(c_ptr), value :: owner
        real(c_double), intent(out) :: values(*),coefficients(*)
        integer(c_int), intent(out) :: indices(*)
      end function
'''+s[b:]
    change('      if (Nodes/=1.or.size(this%SpHa1%F,1)/=NQ+1.or.size(this%SpHa2%F,1)/=NQ+1) &',
           '      if (Nodes/=1.or.NRTAB/=NQ.or.size(this%SpHa1%F,1)/=NQ+1.or.size(this%SpHa2%F,1)/=NQ+1) &')
    a=s.index('      allocate(radial_bank(');b=s.index('    endif\n',a)
    s=s[:a]+'''      radial_status=compact_begin(NQ,dim1,dim2,radial_maxl,this%MODE,NQ*PI/RMAX,DQ,DR,FFTOL, &
        radial_l1,radial_l2,this%SpHa1%F(:,this%BASE_SPHA1+1:this%BASE_SPHA1+dim1), &
        this%SpHa2%F(:,this%BASE_SPHA2+1:this%BASE_SPHA2+dim2),radial_count,radial_owner)
      if (radial_status/=0) call die('REMAG compact radial setup failed')
      allocate(radial_values(0:NQ,radial_count),radial_indices(0:radial_maxl,first:last), &
               radial_coefficients(0:radial_maxl,first:last))
      radial_status=compact_export(radial_owner,radial_values,radial_indices,radial_coefficients)
      if (radial_status/=0) call die('REMAG compact radial export failed')
      this%NFFR=radial_count
      this%MFFR=max(this%MFFR,this%NFFR)
      call RE_ALLOC(this%FFR,0,NRTAB,1,2,1,this%MFFR,'FFR',MYNAME)
      this%FFR(:,1,1:this%NFFR)=radial_values
      DFFR0=0._dp;DFFRMX=0._dp
      do JG=1,this%NFFR
        call SPLINE(RMAX/NRTAB,this%FFR(0:NRTAB,1,JG),NRTAB+1,DFFR0,DFFRMX,this%FFR(0:NRTAB,2,JG))
      enddo
      deallocate(radial_values)
'''+s[b:]
    change('''        if (radial_enabled) then
          FFL(:)=radial_bank(:,L3,i)
        else
        call RADFFT(L3, NQ, NQ*PI/RMAX, FFQ, FFL)
        endif''','        call RADFFT(L3, NQ, NQ*PI/RMAX, FFQ, FFL)')
    a=s.index('      do L3 = ABS(L1-L2), L1+L2, 2\n')+len('      do L3 = ABS(L1-L2), L1+L2, 2\n')
    b=s.index('      enddo\n\n      ! Reallocate some arrays',a)
    s=s[:a]+'''        if (radial_enabled) then
          IFFR(L3)=radial_indices(L3,i)
          CFFR(L3)=radial_coefficients(L3,i)
        else
'''+s[a:b]+'        endif\n'+s[b:]
    a=s.index('      ! Find orbitals convolution by multiplication in k-space\n')
    b=s.index('      ! Loop on possible values of l quantum number of product',a)
    s=s[:a]+'      if (.not.radial_enabled) then\n'+s[a:b]+'      endif\n\n'+s[b:]
    p.write_text(s)

def patch_two_center(source):
    template=Path(__file__).resolve().parent/'native/remag_two_center.F90.in'
    if not template.exists():template=Path(__file__).resolve().parent.parent/'native/remag_two_center.F90.in'
    (source/'Src/remag_two_center.F90').write_text(template.read_text())
    path=source/'Src/matel.F90';text=path.read_text()
    text=replace_once(text,'\nmodule matel_m\n','\nmodule matel_m\n  use m_remag_two_center, only: remag_two_center_prepare,remag_two_center_record\n')
    text=replace_once(text,'    main_tables_initialized = .true.','    call remag_two_center_prepare(tab_S,tab_T)\n    main_tables_initialized = .true.')
    for kind,table in ((1,'S'),(2,'T')):
        call=f'        call tab_{table}%get_matel(IG1, IG2, R12, S12, DSDR)'
        text=replace_once(text,call,call+f'\n        call remag_two_center_record({kind},IG1,IG2,R12,S12,DSDR)')
    path.write_text(text)
    imports='      use m_remag_two_center, only: remag_two_center_shadow_enabled,\n     &  remag_two_center_begin,remag_two_center_finish,remag_dm_check\n'
    def fixed(text,name,consumer):
        condition='.not.matrix_elements_only.and.\n     &    remag_two_center_shadow_enabled()==1'
        for action,mode in (('begin',1),('finish',2)):
            call=f"      call timer( '{name}', {mode} )"
            hook=f'      if ({condition})\n     &    call remag_dm_check(remag_two_center_{action}({consumer}))'
            text=replace_once(text,call,call+'\n'+hook if mode==1 else hook+'\n'+call)
        return text
    path=source/'Src/kinefsm.F';text=path.read_text();text=replace_once(text,'      module m_kinefsm\n','      module m_kinefsm\n'+imports)
    path.write_text(fixed(text,'kinefsm',1))
    path=source/'Src/nlefsm.f';text=path.read_text();text=replace_once(text,'      module m_nlefsm\n','      module m_nlefsm\n'+imports)
    split=text.index('      subroutine nlefsm_SO_off');text=fixed(text[:split],'nlefsm',2)+fixed(text[split:],'nlefsm',3);path.write_text(text)
    path=source/'Src/overfsm.f90';text=path.read_text();text=replace_once(text,'\nmodule m_overfsm\n','\nmodule m_overfsm\n  use m_remag_two_center, only: remag_two_center_shadow_enabled, &\n    remag_two_center_begin,remag_two_center_finish,remag_dm_check\n')
    for action,mode in (('begin',1),('finish',2)):
        call=f"    call timer( 'overfsm', {mode} )"
        hook=f'    if(remag_two_center_shadow_enabled()==1) call remag_dm_check(remag_two_center_{action}(4))'
        text=replace_once(text,call,call+'\n'+hook if mode==1 else hook+'\n'+call)
    path.write_text(text)

def patch_pair_force(source):
    template=Path(__file__).resolve().parent/'native/remag_pair_force.F90.in'
    if not template.exists():template=Path(__file__).resolve().parent.parent/'native/remag_pair_force.F90.in'
    (source/'Src/remag_pair_force.F90').write_text(template.read_text())
    path=source/'Src/kinefsm.F';text=path.read_text()
    text=replace_once(text,'      module m_kinefsm\n','''      module m_kinefsm
      use m_remag_pair_force, only: remag_pair_force_shadow_enabled,
     & remag_pair_force_begin,remag_pair_force_observe,
     & remag_pair_force_reference,remag_pair_geometry_prepare,
     & remag_pair_geometry_add,remag_pair_geometry_finish
''')
    anchor='      volume = nua * volcel(scell) / na'
    text=replace_once(text,anchor,anchor+'''
      if (.not.matrix_elements_only.and.
     &    remag_pair_force_shadow_enabled()==1) then
        call remag_dm_check(remag_pair_force_begin(1,nua,lasto(nua),
     &    no,maxnd,nspin,volume,numd,listdptr,listd,fa,stress))
      endif
      if (matrix_elements_only.and.
     &    remag_pair_force_shadow_enabled()==1) then
        call remag_dm_check(remag_pair_geometry_prepare(nua,lasto(nua),
     &    no,maxnh,volume,numh,listhptr,listh))
      endif
''')
    anchor='                  Ti(jo) = Ti(jo) + Tij'
    text=replace_once(text,anchor,'''                  if (matrix_elements_only.and.
     &                remag_pair_force_shadow_enabled()==1) then
                    call remag_dm_check(remag_pair_geometry_add(
     &                iio,jo,ia,jua,ig,jg,xij(1:3,jn)))
                  endif
                  if (.not.matrix_elements_only.and.
     &                remag_pair_force_shadow_enabled()==1) then
                    call remag_dm_check(remag_pair_force_observe(
     &                iio,jo,ia,jua))
                  endif
'''+anchor)
    anchor='      if (.not.matrix_elements_only.and.\n     &    remag_two_center_shadow_enabled()==1)\n     &    call remag_dm_check(remag_two_center_finish(1))'
    text=replace_once(text,anchor,'''      if (matrix_elements_only.and.
     &    remag_pair_force_shadow_enabled()==1)
     &    call remag_dm_check(remag_pair_geometry_finish())
      if (.not.matrix_elements_only.and.
     &    remag_pair_force_shadow_enabled()==1) then
        call remag_dm_check(remag_pair_force_reference(1,Ekin,
     &    fa,stress))
      endif
'''+anchor)
    path.write_text(text)
    path=source/'Src/setup_H0.F';text=path.read_text()
    text=replace_once(text,'      module m_setup_H0\n','''      module m_setup_H0
      use m_remag_pair_force, only: remag_pair_geometry_bind
      use m_remag_two_center, only: remag_dm_check
''')
    anchor="      call timer('Setup_H0',2)"
    text=replace_once(text,anchor,'      call remag_dm_check(remag_pair_geometry_bind())\n'+anchor);path.write_text(text)
    path=source/'Src/overfsm.f90';text=path.read_text()
    text=replace_once(text,'\nmodule m_overfsm\n','''
module m_overfsm
  use m_remag_pair_force, only: remag_pair_force_shadow_enabled, &
    remag_pair_force_begin,remag_pair_force_observe,remag_pair_force_reference
''')
    anchor='    volume = na_u * volcel(scell) / na_s'
    text=replace_once(text,anchor,anchor+'''
    if(remag_pair_force_shadow_enabled()==1) call remag_dm_check( &
      remag_pair_force_begin(4,na_u,no_l,no_s,maxnd,spin%spinor,volume,numd,listdptr,listd,fa,stress))
''')
    anchor="                call matel( 'S', ig, jg, xij(1:3,jn), Sij, grSij )"
    text=replace_once(text,anchor,anchor+'''
                if(remag_pair_force_shadow_enabled()==1) call remag_dm_check( &
                  remag_pair_force_observe(iio,jo,ia,jua))
''')
    anchor='    if(remag_two_center_shadow_enabled()==1) call remag_dm_check(remag_two_center_finish(4))'
    text=replace_once(text,anchor,'''    if(remag_pair_force_shadow_enabled()==1) call remag_dm_check( &
      remag_pair_force_reference(4,0._dp,fa,stress))
'''+anchor)
    path.write_text(text)

def patch_nonlocal_force(source):
    template=Path(__file__).resolve().parent/'native/remag_nonlocal_force.F90.in'
    if not template.exists():template=Path(__file__).resolve().parent.parent/'native/remag_nonlocal_force.F90.in'
    (source/'Src/remag_nonlocal_force.F90').write_text(template.read_text())
    path=source/'Src/nlefsm.f';text=path.read_text()
    text=replace_once(text,'      module m_nlefsm\n','''      module m_nlefsm
      use m_remag_nonlocal_force, only:
     & remag_nonlocal_force_shadow_enabled,remag_nonlocal_begin,
     & remag_nonlocal_record_center,remag_nonlocal_finish,
     & remag_nonlocal_h0_capture,remag_nonlocal_h0_active
     & ,remag_nonlocal_h0_enabled,remag_nonlocal_h0_neighbor,
     & remag_nonlocal_h0_warning
''')
    split=text.index('      subroutine nlefsm_SO_off');pieces=[text[:split],text[split:]]
    for i,part in enumerate(pieces):
        consumer=i+2;split_value='merge(1,0,split_sr_so)' if i else '0'
        h0='H0,H0_off' if i else 'H'
        anchor='      call GetNodeOrbs(nuotot,Node,Nodes,nuo)'
        part=replace_once(part,anchor,anchor+f'''
      if (remag_nonlocal_force_shadow_enabled()==1) then
        call remag_dm_check(remag_nonlocal_begin(
     &    merge(0,1,matrix_elements_only),{consumer},nua,nuo,no,maxnh,
     &    {split_value},volume,numh,listhptr,listh,fa,stress))
        if (matrix_elements_only.and.
     &      remag_nonlocal_h0_active()==1) then
          call remag_nonlocal_h0_capture(0,maxnh,{h0})
        endif
      endif
''')
        anchor='        max_nno_used = max(max_nno_used, nno)' if not i else 'C----- Loop on neighbour orbitals'
        part=replace_once(part,anchor,f'''
      if (remag_nonlocal_force_shadow_enabled()==1) then
        call remag_nonlocal_record_center({consumer},nua,nno,
     &    lastkb(ka)-lastkb(ka-1),kua,iano,iono,
     &    iphKB(lastkb(ka-1)+1),ks)
      endif
'''+anchor)
        anchor=f'      if (.not.matrix_elements_only.and.\n     &    remag_two_center_shadow_enabled()==1)\n     &    call remag_dm_check(remag_two_center_finish({consumer}))'
        ending='Enl,fa,stress,E_offsiteSO)' if i else 'Enl,fa,stress)'
        part=replace_once(part,anchor,'''      if (remag_nonlocal_force_shadow_enabled()==1)
     &  call remag_nonlocal_finish(matrix_elements_only,
     &    '''+ending+'\n'+anchor)
        finish='     &    '+ending+'\n'
        part=replace_once(part,finish,finish+f'''      if (matrix_elements_only.and.
     &  remag_nonlocal_h0_active()==1)
     &  call remag_nonlocal_h0_capture(1,maxnh,{h0})
''')
        if not i:
            anchor='          call remag_nonlocal_h0_capture(0,maxnh,H)'
            part=replace_once(part,anchor,'''          if(nspin/=1) call die(
     &      'REMAG H0 capture requires the scalar setup matrix')
'''+anchor)
        # Keep upstream neighbor selection; capture one registry/displacement
        # per accepted neighbor. Projector registries are collected per center.
        first_query=('              ikb = 0\n              ig = orb_gindex(is,ioa)'
                     if not i else '           ikb = 0\n           do ko = lastkb(ka-1)+1,lastkb(ka) !Generic positions of kb\'s')
        a=part.index(first_query)
        end_query='              enddo' if not i else '           enddo'
        b=part.index(end_query,a)+len(end_query)
        part=part[:a]+"""            if(matrix_elements_only.and.
     &        remag_nonlocal_h0_enabled()==1)then
              ig = orb_gindex(is,ioa)
              call remag_dm_check(remag_nonlocal_h0_neighbor(
     &          ig,xki(1:3,ina)))
            else
"""+part[a:b]+"\n            endif"+part[b:]
        first='        do ino = 1,nno    ! loop over overlaps' if not i else '       do ino = 1,nno'
        last='       enddo  ! loop over 1st orbitals' if not i else '       enddo     ! ino AOs'
        a=part.index(first);b=part.index(last,a)+len(last)
        part=part[:a]+"""      if(.not.(matrix_elements_only.and.
     &  remag_nonlocal_h0_enabled()==1))then
"""+part[a:b]+'\n      endif'+part[b:]
        if i:
            anchor='     &  call remag_nonlocal_h0_capture(1,maxnh,H0,H0_off)'
            part=replace_once(part,anchor,anchor+"""
      if(matrix_elements_only.and.
     &  remag_nonlocal_h0_enabled()==1)
     &  call remag_nonlocal_h0_warning(ill_defined_sr_so_split)
""")
        pieces[i]=part
    path.write_text(''.join(pieces))
    path=source/'Src/setup_H0.F';text=path.read_text()
    text=replace_once(text,'      module m_setup_H0\n','''      module m_setup_H0
      use m_remag_nonlocal_force, only: remag_nonlocal_bind_geometry,
     &  remag_dm_check
''')
    anchor="      call timer('Setup_H0',2)"
    text=replace_once(text,anchor,'      call remag_dm_check(remag_nonlocal_bind_geometry())\n'+anchor);path.write_text(text)

def patch_integral_force(source):
    template=Path(__file__).resolve().parent/'native/remag_integral_force.F90.in'
    if not template.exists():template=Path(__file__).resolve().parent.parent/'native/remag_integral_force.F90.in'
    (source/'Src/remag_integral_force.F90').write_text(template.read_text())
    path=source/'Src/kinefsm.F';text=path.read_text()
    text=replace_once(text,'      use m_remag_pair_force, only: remag_pair_force_shadow_enabled,',
        '      use m_remag_pair_force, only: remag_pair_force_shadow_enabled,\n     & remag_pair_force_active,')
    # Only the H0 geometry capture uses the active predicate. CPU force
    # reference hooks remain shadow-only and are bypassed in production.
    for indent in ('    ','                '):
        old='if (matrix_elements_only.and.\n     &'+indent+'remag_pair_force_shadow_enabled()==1)'
        text=text.replace(old,old.replace('shadow_enabled','active'))
    path.write_text(text)
    path=source/'Src/nlefsm.f';text=path.read_text()
    text=replace_once(text,'     & remag_nonlocal_force_shadow_enabled,remag_nonlocal_begin,',
        '     & remag_nonlocal_force_active,remag_nonlocal_begin,')
    text=text.replace('remag_nonlocal_force_shadow_enabled()==1','remag_nonlocal_force_active()==1');path.write_text(text)
    path=source/'Src/dhscf.F';text=path.read_text()
    text=replace_once(text,'      use m_remag_force\n','      use m_remag_force\n      use m_remag_integral_force\n')
    text=replace_once(text,'          stressl=stressl+DStres','''          if(remag_integral_force_enabled()==1) then
            call remag_dm_check(remag_integral_hartree_stress())
          else
            stressl=stressl+DStres
          endif''');path.write_text(text)
    path=source/'Src/final_H_f_stress.F';text=path.read_text()
    anchor='      use sys, only: die\n'
    text=replace_once(text,anchor,anchor+'''      use m_remag_integral_force
      use m_remag_force, only: remag_final_grid_enabled
''')
    text=replace_once(text,'      real(dp)              :: g2max','      real(dp)              :: remag_integral_energy(10)\n      real(dp)              :: g2max')
    start=text.index('!     Kinetic: energy, forces, stress and matrix elements')
    end=text.index('! Hubbard term for LDA+U:',start)
    block=text[start:end]
    text=text[:start]+'''      if(remag_integral_force_enabled()==1) then
        if(spin%SO_onsite.or.switch_dftu.or.
     &     remag_final_grid_enabled()/=1)
     &    call die('REMAG integral forces: unsupported mode')
        call remag_dm_check(remag_integral_force_begin(na_u,
     &    fal,stressl))
      else
'''+block+'      endif\n\n'+text[end:]
    start=text.index('      call overfsm(');end=text.index('     &              spin, Escf, fal, stressl )',start)+len('     &              spin, Escf, fal, stressl )')
    text=text[:start]+'''      if(remag_integral_force_enabled()==1) then
        call remag_dm_check(remag_integral_force_finish(na_u,
     &    fal,stressl,remag_integral_energy))
        Ekin=remag_integral_energy(1)
        Enl=remag_integral_energy(2)
        Eso=sum(remag_integral_energy(3:6))
        call update_E0()
      else
'''+text[start:end]+'\n      endif'+text[end:];path.write_text(text)

def patch_device_cycle(source):
    path=source/'Src/remag_dm.F90';text=path.read_text()
    anchor='  interface\n'
    text=replace_once(text,anchor,anchor+'''    integer(c_int) function remag_cycle_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_initialization_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_cycle_request() bind(C)
      import c_int
    end function
    integer(c_int) function remag_initial_mesh_enabled() bind(C)
      import c_int
    end function
    integer(c_int) function remag_cycle_wants_grid() bind(C)
      import c_int
    end function
    integer(c_int) function remag_cycle_grid(mesh,spins,nsm,npcc,atom,core,neutral,b,cutoff,g,volume,dvol,harris) bind(C)
      import c_int,c_double
      integer(c_int),intent(in) :: mesh(*)
      integer(c_int),value :: spins,nsm,npcc
      real(c_double),intent(in) :: atom(*),core(*),neutral(*),b(*),g(*)
      real(c_double),value :: cutoff,volume,dvol,harris
    end function
''')
    path.write_text(text)
    template=Path(__file__).resolve().parent/'native/remag_cycle.F90.in'
    if not template.exists():
        template=Path(__file__).resolve().parent.parent/'native/remag_cycle.F90.in'
    (source/'Src/remag_cycle.F90').write_text(template.read_text())
    (source/'Src/remag_basis.F90').write_text((template.parent/'remag_basis.F90.in').read_text())
    (source/'Src/remag_atomic_basis.F90').write_text((template.parent/'remag_atomic_basis.F90.in').read_text())
    path=source/'Src/dhscf.F';text=path.read_text()
    anchor='      use meshsubs,       only : PhiOnMesh\n'
    text=replace_once(text,anchor,anchor+'''      use m_remag_basis, only: remag_initial_mesh_shadow_enabled,
     &  remag_initial_mesh_check,remag_initial_mesh_enabled,
     &  remag_initial_mesh_produce
''')
    anchor="      call de_alloc( Vscf, 'Vscf', 'dhscf_init' )"
    text=replace_once(text,anchor,'''      if(remag_initial_mesh_shadow_enabled()==1) then
        if(Nodes/=1.or.nbcell/=3.or.any(ntml/=ntm))
     &    call die('REMAG initial mesh: requires serial periodic mesh')
        call remag_initial_mesh_check(ntm,nsm,norb,nuotot,nmpl,
     &    cell,datm,rhoatm,Vscf,Uharrs,DStres)
      endif
'''+anchor)
    start=text.index("      call re_alloc( rhoatm_par, 1, ntpl, 'rhoatm_par', 'dhscf_init' )")
    end=text.index("      call de_alloc( rhoatm_par, 'rhoatm_par', 'dhscf_init' )",start)
    end=text.index('\n',end)+1
    legacy=text[start:end]
    text=text[:start]+'''      if(remag_initial_mesh_enabled()==1)then
        if(frstme)call initialize_doping_uniform()
        if(Nodes/=1.or.nbcell/=3.or.any(ntml/=ntm).or.
     &     ntpl/=product(ntm).or.debug_dhscf.or.mix_charge.or.
     &     doping_active.or.mm_charges%n>0.or.TSmode.or.harrisfun)
     &    call die('REMAG initial mesh: unsupported initialization')
        call re_alloc(rhoatm,1,ntpl,'rhoatm','dhscf_init')
        call remag_initial_mesh_produce(ntm,nsm,norb,nuotot,nmpl,
     &    cell,datm,rhoatm,Uharrs)
      else
'''+legacy+'''      endif
'''+text[end:]
    path.write_text(text)
    path=source/'Src/meshsubs.F';text=path.read_text()
    start=text.index('      subroutine PhiOnMesh(')
    end=text.index('      end subroutine PhiOnMesh',start)
    part=text[start:end]
    part=replace_once(part,'      use precision, only : dp, grid_p',
        '      use m_remag_dm, only: remag_initial_mesh_enabled\n'+
        '      use precision, only : dp, grid_p')
    part=replace_once(part,'      logical                :: within',
        '      logical                :: within, remag_mesh_device')
    part=replace_once(part,"      call timer( 'PHION', 1 )",
        "      call timer( 'PHION', 1 )\n"+
        '      remag_mesh_device=remag_initial_mesh_enabled()==1')
    part=replace_once(part,'C     Allocate memory related to nmpl',
        "      if(remag_mesh_device.and.DirectPhi)\n"+
        "     &  call die('REMAG initial mesh: DirectPhi unsupported')\n"+
        'C     Allocate memory related to nmpl')
    part=replace_once(part,'              if (.not.DirectPhi) then',
        '              if (.not.DirectPhi.and..not.remag_mesh_device) then')
    text=text[:start]+part+text[end:];path.write_text(text)
    path=source/'Src/remag_scf.F90';text=path.read_text()
    text=replace_once(text,'  use m_remag_dm, only: remag_dm_check',
        '  use m_remag_dm, only: remag_dm_check,remag_cycle_enabled')
    text=replace_once(text,'    call remag_scf_guard()\n','    if(remag_cycle_enabled()==1) return\n    call remag_scf_guard()\n')
    path.write_text(text)
    # Stage constants from the ordinary initial Hamiltonian calculation. It
    # still runs the original initial physics; no accepted SCF step is hidden.
    path=source/'Src/dhscf.F';text=path.read_text()
    anchor='      use m_remag_dm, only: remag_flow_active, remag_flow_needs_setup,\n'
    text=replace_once(text,anchor,'      use m_remag_dm, only: remag_cycle_wants_grid,\n     &  remag_cycle_grid,remag_initialization_enabled\n'+anchor)
    start=text.index('      if (remag_flow_active()==1) then')
    guard_start=text.index('        if (Nodes/=1',start)
    guard_end=text.index('        remag_setup=remag_flow_needs_setup',guard_start)
    config_start=text.index('          call remag_getxc(n=remag_nf)',guard_end)
    config_end=text.index('          call remag_dm_check(remag_flow_prepare',config_start)
    stage='      if (remag_cycle_wants_grid()==1) then\n'+text[guard_start:guard_end]+text[config_start:config_end]+'''        call remag_dm_check(remag_cycle_grid(ntm,nspin,nsm,npcc,
     &    rhoatm,rhopcc,Vna,remag_b,remag_cutoff,remag_g,
     &    volume,dvol,Uharrs))
        if(remag_initialization_enabled()==1)then
          call re_alloc(DRho_par,1,ntpl,1,nspin,
     &                  'DRho_par','dhscf')
C         rhoofd stages only topology while the initial grid is pending.
C         Dscf and DRho_par are not read or written by that staging call.
          call rhoofd(norb,nmpl,maxnd,numd,listdptr,listd,nspin,
     &      Dscf,DRho_par,nuo,nuotot,iaorb,iphorb,isa)
          call de_alloc(DRho_par,'DRho_par','dhscf')
          DUext=0._dp
          call timer('DHSCF3',2)
          call timer('DHSCF',2)
          return
        endif
      endif
'''
    text=text[:start]+stage+text[start:];path.write_text(text)
    path=source/'Src/siesta_forces.F90';text=path.read_text()
    anchor='    use m_remag_scf\n'
    text=replace_once(text,anchor,anchor+'    use m_remag_cycle\n')
    text=replace_once(text,'    integer :: iscf\n',
        '    integer :: iscf,remag_iterations,remag_iteration\n    logical :: remag_expected,remag_final\n')
    text=replace_once(text,'         H = Hold\n', '''         ! The integral route publishes the DM-generating H directly.
         ! Hold/Dold/Eold remain private device-cycle history in this mode.
         if (.not.(remag_cycle_enabled()==1.and. &
                   remag_integral_force_enabled()==1)) H = Hold
''')
    anchor='            call setup_hamiltonian( 0 )'
    text=replace_once(text,anchor,'''            if(remag_cycle_enabled()==1) then
              call remag_cycle_guard()
              call remag_dm_check(remag_cycle_request())
            endif
'''+anchor)
    start=text.index('      iscf = 0\n      do while ( iscf < nscf )')
    end=text.index('    call remag_dm_check(remag_phase_enter(2,max(0,istep),0))',start)
    old=text[start:end]
    branch='''      if(remag_cycle_enabled()==1) then
        call pxfflush(6)
        call remag_cycle_run(max(0,istep),remag_iterations,remag_final)
        do remag_iteration=1,remag_iterations
          iscf=remag_iteration
          if(cml_p) call cmlStartStep(xf=mainXML,type='SCF',index=iscf)
          call remag_cycle_report(iscf,dDmax,dHmax,dEmax,dQ,remag_expected)
          call scfconvergence_test(iscf==1,iscf,dDmax,dHmax,dEmax,dQ,conv_harris,conv_freeE,SCFconverged)
          if(SCFconverged.and.iscf<min_nscf) SCFconverged=.false.
          if(SCFconverged.neqv.remag_expected) call die('REMAG cycle: convergence replay mismatch')
          if(cml_p) call cmlEndStep(mainXML)
        enddo
        if(SCFconverged.neqv.remag_final) call die('REMAG cycle: final convergence mismatch')
        call pxfflush(6)
      else
'''+old+'      endif\n'
    text=text[:start]+branch+text[end:]
    text=text.replace('if (remag_spectra_enabled()==1) then','if (remag_spectra_enabled()==1.and.remag_cycle_enabled()==0) then')
    text=text.replace('if (remag_state_enabled()==1) call remag_dm_check(remag_state_export())',
        'if (remag_state_enabled()==1.and.remag_cycle_enabled()==0) call remag_dm_check(remag_state_export())')
    path.write_text(text)
    path=source/'Src/scfconvergence_test.F';text=path.read_text()
    text=replace_once(text,'     &  remag_scf_charge\n','     &  remag_scf_charge\n      use m_remag_dm, only: remag_cycle_enabled\n')
    text=replace_once(text,'         if(remag_state_active()==1) then',
        '         if(remag_state_active()==1.or.remag_cycle_enabled()==1) then')
    path.write_text(text)


    path=source/'Src/siesta_analysis.F90';text=path.read_text()
    text=replace_once(text,'      use m_eo\n','      use m_eo\n'+
        '      use m_remag_cycle, only: remag_cycle_spectrum_deferred, remag_cycle_analysis_spectrum\n'+
        '      use m_remag_dm, only: remag_cycle_enabled, remag_dm_check\n')
    anchor='!     Print eigenvalues\n'
    text=replace_once(text,anchor,'         if (remag_cycle_enabled()==1.and.remag_cycle_spectrum_deferred()==1.and. &\n             scf_eigenvalues_available.and.IONode) &\n             call remag_dm_check(remag_cycle_analysis_spectrum(size(eo),eo))\n\n'+anchor)
    path.write_text(text)

def patch_initialization(source):
    """Defer the initial electronic consumers to the device-cycle owner."""
    path=source/'Src/state_init.F';text=path.read_text()
    anchor='      use m_normalize_dm, only: normalize_dm\n'
    text=replace_once(text,anchor,anchor+'''      use m_remag_dm, only: remag_initialization_enabled,
     &                     remag_cycle_enabled
      use m_remag_population
''')
    anchor='      call new_DM(auxchanged, DM_history, DM_2D, EDM_2D)\n\n      Dscf => val(DM_2D)\n      Escf => val(EDM_2D)\n'
    text=replace_once(text,anchor,anchor+'''      call remag_population_initial(
     &  option_charges%spin%run(OPT_CHARGE_INIT),
     &  option_charges%mulliken%run(OPT_CHARGE_INIT))
''')
    anchor='''          call dm_charge(spin, no_l, numh, listhptr,
     &             S, Dscf, dummy_qspin)
          call print_spin(dummy_qspin)'''
    text=replace_once(text,anchor,'''          if(remag_population_active()==1)then
            call remag_population_print_spin()
          else
'''+anchor+'''
          endif''')
    anchor='!     The number of old supercells is used in new_DM.'
    text=replace_once(text,anchor,'      call remag_population_initial_end()\n\n'+anchor)
    text=replace_once(text,'         call normalize_DM( first= .true. )\n         Escf(:,:) = 0.0_dp', '''         if(remag_initialization_enabled()==1)then
           if(remag_cycle_enabled()/=1)
     &       call die('REMAG GPU initialization requires device SCF')
C          Initial requested populations above use the unnormalized DM,
C          as upstream does. Normalization and zero EDM now run on GPU.
         else
           call normalize_DM( first= .true. )
           Escf(:,:) = 0.0_dp
         endif''')
    path.write_text(text)
    path=source/'Src/setup_hamiltonian.F';text=path.read_text()
    anchor='      use m_dhscf,      only: dhscf\n'
    text=replace_once(text,anchor,anchor+'''      use m_remag_dm, only: remag_initialization_enabled
''')
    anchor='      nullify(fal)\n'
    text=replace_once(text,anchor,anchor+'''      if(iscf==0.and.remag_initialization_enabled()==1)goto 8710
''')
    text=replace_once(text,'      g2max = g2cut\n',' 8710 g2max = g2cut\n')
    anchor='     .            use_rhog_in)\n'
    text=replace_once(text,anchor,anchor+'''
      if(iscf==0.and.remag_initialization_enabled()==1)then
        call de_alloc(fal,'fal','setup_hamiltonian')
        call timer('setup_H',2)
        return
      endif
''')
    path.write_text(text)
    path=source/'Src/setup_H0.F';text=path.read_text()
    anchor='      module m_setup_H0\n'
    text=replace_once(text,anchor,anchor+'      use m_remag_dm, only: remag_initialization_enabled\n')
    start=text.index('        do i = 1, maxnh\n')
    end=text.index('        enddo\n',start)+len('        enddo\n')
    # The initial SOC energy is produced together with the device H. The
    # matrix-element setup above is geometry-only and still runs.
    text=text[:start]+'        if(remag_initialization_enabled()/=1)then\n'+text[start:end]+'        endif\n'+text[end:]
    path.write_text(text)

def patch(source: Path):
    path = source / "Src/diag.F90"
    text = path.read_text()
    if "remag_gpu_enabled" in text:
        raise ValueError("Source already patched")
    assert text.count("  use precision\n") == 1
    text = text.replace("  use precision\n", "  use precision\n  use iso_c_binding, only: c_int, c_double, c_double_complex\n")
    text = text.replace("\ncontains\n", INTERFACES + "\ncontains\n", 1)
    for kind, timer in (("complex", "cdiag"), ("real", "rdiag")):
        anchor = f"    call timer('{timer}',1)"
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + f'''

    if (remag_gpu_enabled() == 1) then
       if (.not. Serial .or. Nodes /= 1 .or. nml /= n .or. nm < n) then
          call die('REMAG: only single-process full local matrices are supported')
       end if
       call timer('remag_gpu',1)
       ierror = remag_solve_{kind}(n, neig, nml, H, S, Z, w)
       call timer('remag_gpu',2)
       if (ierror /= 0) then
          write(*,*) 'REMAG cuSOLVER error: ', ierror
          call die('REMAG: GPU eigensolver failed')
       end if
       call timer('{timer}',2)
       return
    end if
''')
    path.write_text(text)
    patch_grids(source)
    patch_density_matrices(source)
    patch_execution(source)
    patch_occupations(source)
    patch_spectral_residency(source)
    patch_matrix_assembly(source)
    patch_hartree(source)
    patch_resident_grid(source)
    patch_sparse_state(source)
    patch_device_cycle(source)
    patch_force_shadow(source)
    patch_atomic_force_shadow(source)
    patch_final_grid(source)
    patch_final_density(source)
    patch_two_center(source)
    patch_radial_tables(source)
    patch_radial_compaction(source)
    patch_pair_force(source)
    patch_nonlocal_force(source)
    patch_integral_force(source)
    patch_population(source)
    patch_initialization(source)
    patch_density_history(source)
    patch_orientations(source)
    with (source / "CMakeLists.txt").open("a") as f:
        f.write('''
# REMAG integration; reference source is built separately and remains unmodified.
if(NOT REMAG_NATIVE_DIR)
  message(FATAL_ERROR "Set REMAG_NATIVE_DIR to the REMAG native source directory")
endif()
enable_language(CXX)
add_subdirectory(${REMAG_NATIVE_DIR} remag-native)
target_link_libraries(SIESTA.libsiesta PUBLIC remag_gpu)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_dm.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_scf.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_cycle.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_basis.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_atomic_basis.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_two_center.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_pair_force.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_nonlocal_force.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_integral_force.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_population.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_history.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_orientations.F90)
target_sources(SIESTA.libsiesta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/remag_force.F90)
''')

def patch_density_history(source):
    """Keep converged densities and geometry extrapolation on the device."""
    template=Path(__file__).resolve().parents[1]/'native/remag_history.F90.in'
    if not template.is_file():template=Path('/opt/remag/native/remag_history.F90.in')
    (source/'Src/remag_history.F90').write_text(template.read_text())
    path=source/'Src/state_init.F';text=path.read_text()
    text=replace_once(text,'      use m_remag_population\n',
        '      use m_remag_population\n      use m_remag_history, only: remag_history_enabled\n')
    start=text.index('      do i = 1, n_items(DM_history)')
    end=text.index('      end do',start)+len('      end do')
    text=text[:start]+'      if(remag_history_enabled()/=1)then\n'+text[start:end]+'\n      endif'+text[end:]
    path.write_text(text)
    path=source/'Src/m_new_dm.F90';text=path.read_text()
    text=replace_once(text,'    use siesta_options\n','    use siesta_options\n    use m_remag_history\n')
    text=replace_once(text,'    DM_in_history = n_items(DM_history)', '''    if(remag_history_enabled()==1)then
       call get_allowed_history_depth(n_depth)
       call remag_history_policy(n_depth)
       DM_in_history=remag_history_count()
       if(DM_in_history<0)call die('REMAG history: invalid history count')
    else
       DM_in_history = n_items(DM_history)
    endif''')
    text=replace_once(text,'    if ( DM_init ) then\n', '''    if(remag_history_enabled()==1)call remag_history_select(.not.DM_init)
    if ( DM_init ) then
''')
    text=replace_once(text,'''       call extrapolate_DM_with_coords(DM_history, na_u, xa(:,1:na_u), &
            sparse_pattern, DM_2D)''', '''       if(remag_history_enabled()==1)then
          ! Shape/identity for legacy interfaces; electronic values are
          ! produced directly in the initial device density allocation.
          call newdSpData2D(sparse_pattern,spin%DM,block_dist,DM_2D, &
               name='Device history density')
       else
          call extrapolate_DM_with_coords(DM_history, na_u, xa(:,1:na_u), &
               sparse_pattern, DM_2D)
       endif''')
    path.write_text(text)
    path=source/'Src/post_scf_work.F';text=path.read_text()
    text=replace_once(text,'      USE siesta_options\n', '''      USE siesta_options
      use m_remag_history, only: remag_history_enabled,
     &                          remag_history_finish
      use m_remag_dm, only: remag_dm_check
''')
    start=text.index('      ! Store the density matrix to the coordinate stack')
    end=text.index("      call timer( 'PostSCF', 2 )",start)
    text=text[:start]+'''      if(remag_history_enabled()==1)then
        call remag_dm_check(remag_history_finish(merge(1,0,
     &    max_size(DM_history)>0.and..not.
     &    ((idyn==6.or.idyn==7).and.istep>inicoor))))
      else
'''+text[start:end]+'''      endif

'''+text[end:]
    path.write_text(text)

def patch_population(source):
    template=Path(__file__).resolve().parents[1]/'native/remag_population.F90.in'
    if not template.is_file():template=Path('/opt/remag/native/remag_population.F90.in')
    (source/'Src/remag_population.F90').write_text(template.read_text())
    for name in ('siesta_analysis.F90','state_analysis.F'):
        path=source/'Src'/name;text=path.read_text()
        text=replace_once(text,'      use dm_charge_m, only: dm_charge\n',
            '      use dm_charge_m, only: dm_charge\n      use m_remag_population\n')
        start=text.index('         call dm_charge(spin, no_l, numh, listhptr,')
        end=text.index('S, Dscf, qspin)',start)+len('S, Dscf, qspin)')
        text=text[:start]+'''         if(remag_population_enabled()==1) then
           call remag_population_charge(qspin)
         else
'''+text[start:end]+'''\n         endif'''+text[end:]
        text=replace_once(text,'call print_spin(qspin)', '''if(remag_population_enabled()==1) then
           call remag_population_print_spin()
         else
           call print_spin(qspin)
         endif''')
        path.write_text(text)
    path=source/'Src/mulliken.F';text=path.read_text()
    text=replace_once(text,'      use precision,    only : dp\n',
        '      use precision,    only : dp\n      use m_remag_population\n      use sys, only: die\n')
    text=replace_once(text,'      real(dp) :: dm_aux(4)', '''      real(dp) :: dm_aux(4),remag_qtotal(spin%Grid)
      real(dp), allocatable :: remag_qo(:,:),remag_qa(:,:)
      real(dp), allocatable :: remag_ov(:,:),remag_av(:,:)
      real(dp), allocatable :: remag_sv(:,:)''')
    anchor='        partial_qa(:,:) = 0._dp'
    text=replace_once(text,anchor,anchor+'''
        if(remag_population_active()==1) then
          if(iopt/=1) call die('REMAG population: requires option 1')
          if(spin%DM<4)then
            call remag_population_values(natoms,nbasistot,
     &        remag_qo,remag_qa)
            call remag_population_charge(remag_qtotal)
            partial_qa(:,:)=remag_qa
          else
            call remag_population_views(natoms,nbasistot,ns,
     &        remag_ov,remag_av,remag_sv)
          endif
        endif''')
    text=replace_once(text,'            qtot = 0._dp', '''            qtot = 0._dp
            if(remag_population_active()==1)qtot=remag_qtotal(is)''')
    start=text.index('                 qa = 0.0_dp',text.index('Compute charge in each orbital of atom ia'))
    end=text.index('                 qtot = qtot + qa',start)+len('                 qtot = qtot + qa')
    text=text[:start]+'''                 if(remag_population_active()==1) then
                   qo(:)=remag_qo(is,lasto(ia-1)+1:lasto(ia))
                   qa=remag_qa(is,ia)
                 else
'''+text[start:end]+'''\n                 endif'''+text[end:]
    # All host contraction/MPI storage belongs to the CPU branch. GPU
    # consumers below only select values for existing upstream formatting.
    start=text.index('          call re_alloc( qos,',text.index('elseif (spin%DM .ge. 4) then'))
    end=text.index('          ! The above reduction',start)
    text=text[:start]+'''          if(remag_population_active()/=1)then
'''+text[start:end]+'''          endif

'''+text[end:]
    start=text.index('                 ! Reduce total charge per-atom')
    end=text.index('                 if ( IONode ) then',start)
    text=text[:start]+'''                 if(remag_population_active()==1)then
                   qtot=remag_ov(1,io);stot=remag_ov(2,io)
                   svec=remag_ov(3:5,io)
                 else
'''+text[start:end]+'''                 endif
'''+text[end:]
    text=replace_once(text,'              call spnvec( spin%Grid, qas, qtot, stot, svec )', '''              if(remag_population_active()==1)then
                qtot=remag_av(2,ia);stot=remag_av(3,ia)
                svec=remag_av(4:6,ia)
              else
                call spnvec(spin%Grid,qas,qtot,stot,svec)
              endif''')
    text=replace_once(text,'''              do is = 1 , spin%Grid
                 qts(is) = qts(is) + qas(is)
              end do''', '''              if(remag_population_active()/=1)then
                do is = 1 , spin%Grid
                   qts(is) = qts(is) + qas(is)
                end do
              endif''')
    text=replace_once(text,'            call spnvec( spin%Grid, qts, qtot, stot, svec )', '''            if(remag_population_active()==1)then
              qtot=remag_sv(1,ispec);stot=remag_sv(2,ispec)
              svec=remag_sv(3:5,ispec)
            else
              call spnvec(spin%Grid,qts,qtot,stot,svec)
            endif''')
    text=replace_once(text,"          call de_alloc( qos, 'qos', 'mulliken' )", '''          if(remag_population_active()/=1)
     &      call de_alloc(qos,'qos','mulliken')''')
    path.write_text(text)
    path=source/'Src/partial_charges.F90';text=path.read_text()
    text=replace_once(text,'  subroutine write_partial_charges(nspin, na_u, q, method)\n',
        '  subroutine write_partial_charges(nspin, na_u, q, method)\n    use m_remag_population\n')
    text=replace_once(text,'    real(dp), allocatable :: qout(:,:)',
        '    real(dp), allocatable :: qout(:,:)\n    logical :: remag_derived')
    text=replace_once(text,'    spin_label = "S"', '''    remag_derived=method==3.and.remag_population_active()==1
    if(remag_derived) &
      call remag_population_partial(nspin,na_u,qout,Stot,Svec,dqtot)
    spin_label = "S"''')
    # Keep headers, CML fields and formatting exactly as upstream; wrap only
    # allocation and numerical work for each supported spin representation.
    for columns,ending in ((2,'\n    case ( 2 )'),(3,'\n      if ( cml_p ) then'),(6,'\n      if ( cml_p ) then')):
        start=text.index(f'      allocate(qout({columns},na_u))')
        end=text.index(ending,start)
        text=text[:start]+'      if(.not.remag_derived)then\n'+text[start:end]+'      endif\n'+text[end:]
    text=replace_once(text,'    dqtot = 0._dp','    if(.not.remag_derived)dqtot = 0._dp')
    text=replace_once(text,'      dqtot = dqtot + qout(1, ia)',
        '      if(.not.remag_derived)dqtot = dqtot + qout(1, ia)')
    path.write_text(text)


def patch_orientations(source):
    """Reuse fixed crystal geometry across explicit independent magnetic directions."""
    template=Path(__file__).resolve().parents[1]/'native/remag_orientations.F90.in'
    if not template.is_file():template=Path('/opt/remag/native/remag_orientations.F90.in')
    (source/'Src/remag_orientations.F90').write_text(template.read_text())
    path=source/'Src/siesta.F';text=path.read_text()
    text=replace_once(text,'      use m_siesta_init\n','      use m_siesta_init\n      use m_remag_orientations\n')
    text=replace_once(text,'      integer :: istep','      integer :: istep, remag_direction')
    text=replace_once(text,'      call siesta_init()','      call siesta_init()\n      call remag_orientations_init()')
    start=text.index('C     Begin of coordinate relaxation iteration')
    end=text.index('      call siesta_end()')
    original=text[start:end]
    sweep='''      if (remag_direction_count.gt.0) then
        do remag_direction=1,remag_direction_count
          call remag_orientation_begin(remag_direction)
          istep=inicoor+remag_direction-1
          call timer('IterGeom',1)
          call siesta_forces(istep)
          relaxd=.false.
          call siesta_move(istep,relaxd)
          call timer('IterGeom',2)
          call siesta_analysis(relaxd)
          call remag_orientation_end(remag_direction)
        enddo
        call remag_orientations_restore()
      else
'''
    text=text[:start]+sweep+original+'      endif\n\n'+text[end:]
    path.write_text(text)
    path=source/'Src/state_init.F';text=path.read_text()
    text=replace_once(text,'      use m_new_dm,          only: new_dm\n', '''      use m_new_dm,          only: new_dm
      use m_remag_orientations, only:
     &     remag_reuse_orientation_geometry
''')
    start=text.index('!     Always (in case of auxiliary cell) check the new sparse pattern')
    end=text.index('#ifdef NCDF_4\n!     At this point the sparsity pattern',start)
    text=text[:start]+'''      if(remag_reuse_orientation_geometry())then
C       Fixed crystal: keep sparse/overlap/H0 storage and mesh identities.
C       Electronic mixing and the initial density still start a fresh cycle.
        auxchanged=.false.
        call mixers_history_init(scf_mixs)
        scf_mix=>scf_mixs(1)
        write(*,'(a)') 'REMAG: reuse fixed sparse geometry and overlap'
      else
'''+text[start:end]+'''      endif

'''+text[end:]
    path.write_text(text)

    path=source/'Src/setup_H0.F';text=path.read_text()
    text=replace_once(text,'      use parallel,  only : Node\n','''      use parallel,  only : Node
      use m_remag_orientations, only:
     &     remag_reuse_orientation_geometry
      use m_remag_dm, only: remag_geometry_changed
      use iso_c_binding, only: c_int
      use sys, only: die
''')
    text=replace_once(text,'      real(dp)           :: lo_time','''      real(dp)           :: lo_time
      real(dp),save :: retained_constants(6)=0._dp
      logical,save :: retained_ready=.false.
      interface
        integer(c_int) function remag_reuse_initial_mesh() bind(C)
          import c_int
        end function
        integer(c_int) function remag_reuse_pair_geometry() bind(C)
          import c_int
        end function
        integer(c_int) function remag_reuse_nonlocal_geometry() bind(C)
          import c_int
        end function
        integer(c_int) function remag_reuse_nonlocal_h0() bind(C)
          import c_int
        end function
      end interface''')
    text=replace_once(text,"      call timer('Setup_H0',1)",'''      call timer('Setup_H0',1)
      if(remag_reuse_orientation_geometry())then
        if(.not.retained_ready)
     &    call die('REMAG: no retained H0 geometry')
        g2max=retained_constants(1);Eions=retained_constants(2)
        Ena=retained_constants(3);DEna=retained_constants(4)
        Emm=retained_constants(5);Emeta=retained_constants(6)
        Eso=0._dp
C       Advance the electronic lifetime while retaining immutable geometry.
        call remag_geometry_changed()
        call remag_dm_check(remag_reuse_pair_geometry())
        call remag_dm_check(remag_reuse_nonlocal_geometry())
        call remag_dm_check(remag_reuse_nonlocal_h0())
        call remag_dm_check(remag_reuse_initial_mesh())
        write(*,'(a)') 'REMAG: reuse GPU pair/projector/SOC H0 geometry'
        call timer('Setup_H0',2)
        return
      endif''')
    anchor="      call timer('Setup_H0',2)"
    # There are now two exits. Save constants only after the original path.
    pos=text.rindex(anchor)
    text=text[:pos]+'''      retained_constants=[g2max,Eions,Ena,DEna,Emm,Emeta]
      retained_ready=.true.
'''+text[pos:]
    path.write_text(text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    patch(parser.parse_args().source)
