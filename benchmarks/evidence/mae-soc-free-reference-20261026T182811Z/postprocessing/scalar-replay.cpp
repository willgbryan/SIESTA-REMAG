#include "frozen_potential_math.hpp"
extern "C" int split_many(int count,const double* off,double* scalar,double* soc) {
  for(int i=0;i<count;i++) if(!remag::mae::split_offsite_spin(off+8*i,scalar+8*i,soc+8*i)) return 1;
  return 0;
}
