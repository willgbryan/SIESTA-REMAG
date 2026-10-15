// SPDX-License-Identifier: GPL-3.0-or-later
// Host tetrahedron occupations for the SCF k grid (remag_tetrahedra.F90).
#include "tetrahedra.hpp"
#include "tetrahedron_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
// Kawamura et al. (2014): effective corner energies are W e over the 20
// stencil points below; weights return through the transpose. Rows sum to 1.
const int kFit[4][20]={
    {1440,0,30,0,-38,7,17,-28,-56,9,-46,9,-38,-28,17,7,-18,-18,12,-18},
    {0,1440,0,30,-28,-38,7,17,9,-56,9,-46,7,-38,-28,17,-18,-18,-18,12},
    {30,0,1440,0,17,-28,-38,7,-46,9,-56,9,17,7,-38,-28,12,-18,-18,-18},
    {0,30,0,1440,7,17,-28,-38,9,-46,9,-56,-28,17,7,-38,-18,12,-18,-18}};
int wrap(int i,int n) {i%=n;return i<0?i+n:i;}
// Six tetrahedra around the shortest main diagonal of a grid subcell, as
// integer offsets; for the optimized method each is extended to 20 points.
std::vector<int> offsets(int method,const double* cell,const int* n,int& shortest) {
    // Reciprocal vectors b_j (rows) with a_i . b_j = 2 pi delta_ij.
    const double* a=cell;double b[3][3];
    const double det=a[0]*(a[4]*a[8]-a[5]*a[7])-a[3]*(a[1]*a[8]-a[2]*a[7])+a[6]*(a[1]*a[5]-a[2]*a[4]);
    if(!(std::fabs(det)>0.)) throw std::invalid_argument("singular cell");
    for(int j=0;j<3;j++) {
        const double* u=a+3*((j+1)%3);const double* v=a+3*((j+2)%3);
        b[j][0]=2.*M_PI*(u[1]*v[2]-u[2]*v[1])/det;b[j][1]=2.*M_PI*(u[2]*v[0]-u[0]*v[2])/det;
        b[j][2]=2.*M_PI*(u[0]*v[1]-u[1]*v[0])/det;
    }
    const int diagonal[4][3]={{-1,1,1},{1,-1,1},{1,1,-1},{1,1,1}};
    double best=0.;
    if(shortest<0) for(int d=0;d<4;d++) {
        double l=0.;
        for(int x=0;x<3;x++) {double s=0.;for(int j=0;j<3;j++) s+=diagonal[d][j]*b[j][x]/n[j];l+=s*s;}
        if(d==0||l<best) {best=l;shortest=d;}
    }
    int start[3]={0,0,0},step[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    if(shortest<3) {start[shortest]=1;step[shortest][shortest]=-1;}
    const int count=method==3?20:4;std::vector<int> out;
    for(int i1=0;i1<3;i1++) for(int i2=0;i2<3;i2++) for(int i3=0;i3<3;i3++) {
        if(i2==i1||i3==i1||i3==i2) continue;
        int v[20][3];
        for(int x=0;x<3;x++) {
            v[0][x]=start[x];v[1][x]=v[0][x]+step[i1][x];v[2][x]=v[1][x]+step[i2][x];v[3][x]=v[2][x]+step[i3][x];
            for(int c=0;c<4;c++) {
                v[4+c][x]=2*v[c][x]-v[(c+1)%4][x];v[8+c][x]=2*v[c][x]-v[(c+2)%4][x];
                v[12+c][x]=2*v[c][x]-v[(c+3)%4][x];v[16+c][x]=v[(c+3)%4][x]-v[c][x]+v[(c+1)%4][x];
            }
        }
        for(int c=0;c<count;c++) for(int x=0;x<3;x++) out.push_back(v[c][x]);
    }
    return out;
}
}

namespace remag {
int tetra_grid(int method,const int* grid,const double* displacement,const double* cell,int nk,
               const double* kpoints,const double* weights,int trs,TetraGrid& out,int diagonal) {
    if(method<1||method>3||nk<1||!grid||!displacement||!cell||!kpoints||!weights||diagonal<-1||diagonal>3)
        return tetra_bad_input;
    for(int j=0;j<3;j++) if(grid[j]<1) return tetra_bad_input;
    const long total=long(grid[0])*grid[1]*grid[2];
    if(total>(1L<<28)) return tetra_bad_input;
    out=TetraGrid{};std::copy_n(grid,3,out.n);out.point.assign(total,-1);
    // Grid index of +k or -k: fractional coordinate f_j = k.a_j/(2 pi) = (i_j + d_j)/N_j.
    auto index=[&](const double* k,int sign,long& p) {
        p=0;
        for(int j=2;j>=0;j--) {
            const double f=sign*(k[0]*cell[3*j]+k[1]*cell[3*j+1]+k[2]*cell[3*j+2])/(2.*M_PI)*grid[j]-displacement[j];
            const double r=std::nearbyint(f);
            if(std::fabs(f-r)>1.e-6) return false;
            p=p*grid[j]+wrap(int(std::fmod(r,grid[j])),grid[j]);
        }
        return true;
    };
    std::vector<long> images(nk,0);
    for(int ik=0;ik<nk;ik++) for(int sign=1;sign>=(trs?-1:1);sign-=2) {
        long p;
        if(!index(kpoints+3*ik,sign,p)) return sign==1?tetra_off_grid:tetra_grid_incomplete;
        if(out.point[p]==ik) continue;
        if(out.point[p]>=0) return tetra_grid_conflict;
        out.point[p]=ik;images[ik]++;
    }
    for(long p=0;p<total;p++) if(out.point[p]<0) return tetra_grid_incomplete;
    for(int ik=0;ik<nk;ik++) if(std::fabs(weights[ik]-double(images[ik])/total)>1.e-10) return tetra_bad_weights;
    std::vector<int> v;
    try {v=offsets(method,cell,grid,diagonal);} catch(const std::exception&) {return tetra_bad_input;}
    out.diagonal=diagonal;
    out.corners=method==3?20:4;out.tetrahedra=int(6*total);out.stencil.resize(size_t(out.tetrahedra)*out.corners);
    size_t s=0;
    for(int i3=0;i3<grid[2];i3++) for(int i2=0;i2<grid[1];i2++) for(int i1=0;i1<grid[0];i1++)
        for(size_t c=0;c<v.size();c+=3)
            out.stencil[s++]=int(wrap(i1+v[c],grid[0])+grid[0]*(long(wrap(i2+v[c+1],grid[1]))+grid[1]*wrap(i3+v[c+2],grid[2])));
    return tetra_ok;
}

// Split [0,n) into `threads` contiguous blocks and run f(begin,end,thread) on each.
template<class F> void blocks(int threads,size_t n,F f) {
    threads=int(std::max<size_t>(1,std::min<size_t>(size_t(std::max(1,threads)),n)));
    std::vector<std::thread> pool;
    for(int t=1;t<threads;t++) pool.emplace_back(f,n*t/threads,n*(t+1)/threads,t);
    f(size_t(0),n/threads,0);
    for(auto& x:pool) x.join();
}

int tetra_occupations(const TetraGrid& grid,int method,int nspinor,int maxspn,int maxe,int ne,
                      const double* energies,double charge,double* occupations,double* fermi,bool average,bool given,
                      int threads) {
    if(method<1||method>3||grid.corners!=(method==3?20:4)||grid.tetrahedra<1||(nspinor!=1&&nspinor!=2)||
       maxspn<1||maxspn>2||(nspinor==1&&maxspn!=1)||ne<1||ne>maxe||!energies||!occupations||!fermi||
       !(charge>0.)||threads<1) return tetra_bad_input;
    const int nk=1+*std::max_element(grid.point.begin(),grid.point.end());
    const double full=nspinor==1?2.:1.,share=1./grid.tetrahedra;
    auto energy=[&](int band,int spin,int point) {return energies[size_t(maxe)*(spin+size_t(maxspn)*grid.point[point])+band];};
    // Sorted corner energies of every (spin, band, tetrahedron), with the
    // sorting permutation needed to return weights to their corners.
    const size_t items=size_t(maxspn)*ne*grid.tetrahedra,ntet=grid.tetrahedra;
    std::vector<double> sorted(4*items);std::vector<unsigned char> order(4*items);
    std::vector<double> lows(threads,HUGE_VAL),highs(threads,-HUGE_VAL);std::vector<int> bad(threads,0);
    blocks(threads,items,[&](size_t a,size_t b,int th) {
        for(size_t item=a;item<b;item++) {
            const int t=int(item%ntet),band=int((item/ntet)%ne),spin=int(item/(ntet*ne));
            const int* st=grid.stencil.data()+size_t(t)*grid.corners;
            double e[4];
            for(int i=0;i<4;i++) {
                if(method==3) {
                    double sum=0.;
                    for(int c=0;c<20;c++) sum+=kFit[i][c]*energy(band,spin,st[c]);
                    e[i]=sum/1260.;
                } else e[i]=energy(band,spin,st[i]);
                if(!std::isfinite(e[i])) bad[th]=1;
            }
            unsigned char* o=order.data()+4*item;
            for(int i=0;i<4;i++) o[i]=i;
            std::sort(o,o+4,[&](int x,int y) {return e[x]<e[y]||(e[x]==e[y]&&x<y);});
            for(int i=0;i<4;i++) sorted[4*item+i]=e[o[i]];
            lows[th]=std::min(lows[th],sorted[4*item]);highs[th]=std::max(highs[th],sorted[4*item+3]);
        }
    });
    if(*std::max_element(bad.begin(),bad.end())) return tetra_bad_input;
    const double low=*std::min_element(lows.begin(),lows.end()),high=*std::max_element(highs.begin(),highs.end());
    // Partial counts per block, summed in block order: deterministic for a given thread count.
    std::vector<double> partial(threads);
    auto states=[&](double ef) {
        std::fill(partial.begin(),partial.end(),0.);
        blocks(threads,items,[&](size_t a,size_t b,int th) {
            double n=0.;
            for(size_t item=a;item<b;item++) n+=tetra_count(sorted.data()+4*item,ef);
            partial[th]=n;
        });
        double n=0.;
        for(double x:partial) n+=x;
        return n*full*share;
    };
    const double capacity=full*maxspn*ne,tolerance=1.e-12*std::max(1.,charge);
    if(capacity<charge-tolerance) return tetra_too_few_states;
    double ef=given?*fermi:high;
    if(!given&&capacity>charge+tolerance) {
        // The occupied count is continuous and nondecreasing in ef.
        double a=low,b=high;bool found=false;
        for(int iteration=0;iteration<200&&!found;iteration++) {
            ef=.5*(a+b);const double n=states(ef);
            if(std::fabs(n-charge)<tolerance) found=true;
            else if(n<charge) a=ef; else b=ef;
            if(b-a<=1.e-15*std::max(1.,std::fabs(ef))) found=std::fabs(states(.5*(a+b))-charge)<1.e2*tolerance;
        }
        if(!found) return tetra_no_fermi;
    }
    // The highest band must lie above ef everywhere, or states are missing.
    if(capacity>charge+tolerance) for(int spin=0;spin<maxspn;spin++) for(int t=0;t<grid.tetrahedra;t++)
        if(sorted[4*((size_t(spin)*ne+ne-1)*grid.tetrahedra+t)]<ef) return tetra_top_band_occupied;
    const size_t size=size_t(maxe)*maxspn*nk;
    std::fill_n(occupations,size,0.);
    // Each block accumulates into its own copy (block 0 into the output), summed in block order.
    std::vector<std::vector<double>> copies(std::max(0,threads-1));
    blocks(threads,items,[&](size_t a,size_t b,int th) {
        double* out=occupations;
        if(th>0) {copies[th-1].assign(size,0.);out=copies[th-1].data();}
        for(size_t item=a;item<b;item++) {
            const int t=int(item%ntet),band=int((item/ntet)%ne),spin=int(item/(ntet*ne));
            double w[4],corner[4];
            tetra_weights(sorted.data()+4*item,ef,method==2,w);
            for(int i=0;i<4;i++) corner[order[4*item+i]]=w[i]*full*share;
            const int* st=grid.stencil.data()+size_t(t)*grid.corners;
            for(int c=0;c<grid.corners;c++) {
                double sum=0.;
                if(method==3) {for(int i=0;i<4;i++) sum+=kFit[i][c]*corner[i];sum/=1260.;}
                else sum=corner[c];
                out[size_t(maxe)*(spin+size_t(maxspn)*grid.point[st[c]])+band]+=sum;
            }
        }
    });
    for(auto& c:copies) if(!c.empty()) for(size_t i=0;i<size;i++) occupations[i]+=c[i];
    if(average) for(int ik=0;ik<nk;ik++) for(int spin=0;spin<maxspn;spin++) {
        // QE's grouping: bands within 1e-6 Ry of the group's first band share
        // its total weight; a group reaching the top band is left as it is.
        const double* e=energies+size_t(maxe)*(spin+size_t(maxspn)*ik);
        double* w=occupations+size_t(maxe)*(spin+size_t(maxspn)*ik);
        for(int b=0;b<ne;b++) {
            double sum=w[b];
            for(int c=b+1;c<ne;c++) {
                if(std::fabs(e[b]-e[c])<1.e-6) {sum+=w[c];continue;}
                for(int x=b;x<c;x++) w[x]=sum/(c-b);
                break;
            }
        }
    }
    *fermi=ef;
    return tetra_ok;
}
}

extern "C" int remag_tetra_occupations(int method,const int* grid,const double* displacement,const double* cell,int nk,
    const double* kpoints,const double* weights,int trs,int nspinor,int maxspn,int maxe,int ne,const double* energies,
    double charge,double* occupations,double* fermi,int* tetrahedra,int threads) {
    try {
        remag::TetraGrid g;
        int status=remag::tetra_grid(method,grid,displacement,cell,nk,kpoints,weights,trs,g);
        if(status) return status;
        if(tetrahedra) *tetrahedra=g.tetrahedra;
        return remag::tetra_occupations(g,method,nspinor,maxspn,maxe,ne,energies,charge,occupations,fermi,true,false,
                                        std::max(1,threads));
    } catch(const std::exception& e) {
        std::fprintf(stderr,"REMAG tetrahedra: %s\n",e.what());return remag::tetra_bad_input;
    }
}
