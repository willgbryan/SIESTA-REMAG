program reference
use precision, only: dp
implicit none
integer rows,atoms,species,components,fi,fo,i,j,k
integer,allocatable :: ids(:)
real(dp),allocatable :: z(:),orb(:,:),atom(:,:),total(:),ov(:,:),av(:,:),sv(:,:)
real(dp) summary(10),q(4),s(3)
character(2048) input,output
call get_command_argument(1,input)
call get_command_argument(2,output)
open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
read(fi) rows,atoms,species,components
allocate(ids(atoms),z(species),orb(components,rows),atom(components,atoms),total(components))
allocate(ov(5,rows),av(6,atoms),sv(5,species))
read(fi) ids,z,orb,atom,total
close(fi)
do i=1,rows
 call spnvec(components,orb(:,i),ov(1,i),ov(2,i),ov(3:5,i))
enddo
do i=1,atoms
 call spnvec(components,atom(:,i),av(2,i),av(3,i),av(4:6,i))
 av(1,i)=z(ids(i))-av(2,i)
enddo
q=0._dp
do i=1,species
 do j=1,atoms
  if(ids(j)/=i)cycle
  do k=1,components
   q(k)=q(k)+atom(k,j)
  enddo
 enddo
 call spnvec(components,q,sv(1,i),sv(2,i),sv(3:5,i))
enddo
call spnvec(components,total,summary(1),summary(2),summary(3:5))
summary(6)=0._dp
s=0._dp
do i=1,atoms
 summary(6)=summary(6)+av(1,i)
 s=s+av(4:6,i)
enddo
summary(7)=s(3)
if(components==4)summary(7)=sqrt(s(1)**2+s(2)**2+s(3)**2)
summary(8:10)=s
open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
write(fo) ov,av,sv,summary
close(fo)
end program
