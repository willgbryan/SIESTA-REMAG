program check
 use m_remag_soc_export
 use test_mode
 use precision
 use parallel
 use m_spin
 use sparse_matrices
 use, intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 call get_command_argument(1,mode)
 allocate(H_2D%v(1,8));H_2D%v=19._dp
 if(mode/='missing')then
   allocate(H_so_off_2D%v(1,4))
   H_so_off_2D%v(1,:)=[cmplx(3.,4.,dp),cmplx(1.,2.,dp),cmplx(5.,6.,dp),cmplx(7.,8.,dp)]
 endif
 if(mode=='parallel')Nodes=2
 if(mode=='onsite')spin%SO_offsite=.false.
 if(mode=='collinear')spin%H=2
 if(mode=='nonfinite')H_so_off_2D%v(1,1)=ieee_value(0._dp,ieee_quiet_nan)
 call remag_write_soc()
 if(mode=='disabled'.and.written)stop 5
 if(mode=='valid'.and..not.written)stop 6
 if(any(H_2D%v/=19._dp))stop 7
end program
