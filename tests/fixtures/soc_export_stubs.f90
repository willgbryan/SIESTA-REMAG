! Engineering interface doubles only. No electronic calculation is performed.
module test_mode
 implicit none
 character(32)::mode
 logical::written=.false.
end module
module precision
 use iso_c_binding,only:dp=>c_double
end module
module sys
contains
 subroutine die(message)
 character(*)::message
 print *,message
 error stop 1
 end subroutine
end module
module fdf
contains
 logical function fdf_get(key,default)
 use test_mode
 character(*)::key
 logical::default
 fdf_get=default
 if(key=='REMAG.WriteSOCMatrix')fdf_get=mode/='disabled'
 if(key=='Read-H-from-file')fdf_get=mode=='external'
 end function
end module
module parallel
 integer::Nodes=1
end module
module m_spin
 type spin_t
 logical::SO_offsite=.true.
 integer::H=8
 end type
 type(spin_t)::spin
end module
module class_dSpData2D
 use precision
 interface val
 module procedure val_d
 end interface
 type dSpData2D
 real(dp),pointer::v(:,:)=>null()
 end type
contains
 subroutine newdSpData2D(pattern,dim,distribution,x,name)
 integer::pattern,dim,distribution
 type(dSpData2D)::x
 character(*)::name
 allocate(x%v(pattern,dim))
 end subroutine
 function val_d(x) result(p)
 type(dSpData2D)::x
 real(dp),pointer::p(:,:)
 p=>x%v
 end function
 integer function spar(x)
 type(dSpData2D)::x
 spar=size(x%v,1)
 end function
 integer function dist(x)
 type(dSpData2D)::x
 dist=1
 end function
 subroutine delete(x)
 type(dSpData2D)::x
 deallocate(x%v)
 end subroutine
end module
module class_zSpData2D
 use precision
 interface val
 module procedure val_z
 end interface
 type zSpData2D
 complex(dp),pointer::v(:,:)=>null()
 end type
contains
 function val_z(x) result(p)
 type(zSpData2D)::x
 complex(dp),pointer::p(:,:)
 p=>x%v
 end function
end module
module sparse_matrices
 use class_dSpData2D,only:dSpData2D
 use class_zSpData2D,only:zSpData2D
 type(dSpData2D)::H_2D
 type(zSpData2D)::H_so_off_2D
 integer::S_1D=1
end module
module io_hsx_m
contains
 subroutine write_hsx(x,s,ef,q,temp,k,prec,extension)
 use precision
 use test_mode
 use class_dSpData2D,only:dSpData2D
 type(dSpData2D)::x
 integer::s,k,prec
 real(dp)::ef,q,temp
 character(*)::extension
 if(extension/='SOC.HSX'.or.prec/=dp)stop 2
 if(any(x%v(1,:)/=[1.,-1.,5.,6.,1.,-1.,7.,-8.]))stop 3
 if(ef/=0._dp.or.q/=2._dp.or.temp/=.01_dp)stop 4
 written=.true.
 end subroutine
end module
module kpoint_scf_m
 integer::kpoint_scf=1
end module
module atomlist
 use precision
 real(dp)::qtot=2._dp
end module
module siesta_options
 use precision
 real(dp)::temp=.01_dp
end module
module m_energies
 use precision
 real(dp)::ef=0._dp
end module
module files
 character(32)::slabel='remag'
end module
module m_remag_nonlocal_force
contains
 integer function remag_nonlocal_h0_enabled()
 use test_mode
 remag_nonlocal_h0_enabled=0
 if(mode=='device')remag_nonlocal_h0_enabled=1
 end function
 integer function remag_nonlocal_h0_poison_enabled()
 use test_mode
 remag_nonlocal_h0_poison_enabled=0
 if(mode=='poison')remag_nonlocal_h0_poison_enabled=1
 end function
end module
