// SPDX-License-Identifier: GPL-3.0-or-later
// Host owner for site spin constraints. Fortran stages one trial per SCF.
// Every H built for that SCF carries the trial field (device cycle, or the
// host H that seeds or runs it); the final-density moments are recorded once
// before the optimizer observes them.
#include "spin_constraint.hpp"
#include "spin_constraint_math.hpp"
#include "spin_constraint_optimizer.hpp"
#include "execution.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void require(bool v,const char* m) {if(!v) throw std::runtime_error(m);}
struct Evaluation {int step,line,coarse;double penalty,free_energy,lagrangian,angle,torque,field;};
struct State {
    bool configured=false,retain=false,applied=false,recorded=false,finished=false;
    int sites=0,rows=0,step=0,status=0;uint64_t epoch=0;double penalty=0.; // rho of the pending trial
    remag::SpinConstraintSettings settings{};remag::SpinConstraintOptimizer optimizer;
    std::vector<int> site;std::vector<double> site_fields,targets,orbital_field,orbital_moments;
    std::vector<Evaluation> evaluations;
    ~State() {
        if(!configured) return;
        FILE* f=std::fopen("remag-spin-constraint.json","w");if(!f) return;
        std::fprintf(f,"{\"schema\":2,\"scope\":\"Site Mulliken spin constraints; one converged SCF per trial field\","
            "\"sites\":%d,\"direction\":\"%s\",\"line_search\":\"%s\",\"criteria\":%d,\"penalty_ry_per_muB2\":%.17g,"
            "\"converged\":%s,\"status\":%d,\"evaluations\":[",
            sites,settings.direction==2?"multipliers":settings.direction?"bfgs":"steepest",settings.line?"secant":"golden",
            settings.criteria,settings.penalty,optimizer.converged()?"true":"false",status);
        for(size_t i=0;i<evaluations.size();i++) {
            const auto& e=evaluations[i];
            std::fprintf(f,"%s{\"step\":%d,\"line\":%d,\"coarse\":%s,\"penalty\":%.17g,\"free_energy_ry\":%.17g,\"lagrangian_ry\":%.17g,"
                "\"max_angle_rad\":%.17g,\"max_torque_muB\":%.17g,\"max_field_ry\":%.17g}",
                i?",":"",e.step,e.line,e.coarse?"true":"false",e.penalty,e.free_energy,e.lagrangian,e.angle,e.torque,e.field);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
} state;
void current() {
    require(state.configured&&!state.finished&&state.step>0&&!state.status,"No pending spin-constraint trial");
}
int failure(const char* where,const std::exception& e) {
    std::fprintf(stderr,"REMAG spin constraint %s: %s\n",where,e.what());return -300;
}
// Partner orbital of a sparse entry, as hamiltonian.cu assembles it.
int partner(int orbital,int rows,const int* mapping) {return mapping?mapping[orbital-1]-1:(orbital-1)%rows;}
// Orbital moments of a sparse spinor density (row Mulliken, mulliken.F option 1).
std::vector<double> orbital_moments(int rows,int capacity,int spins,const int* num,const int* ptr,
        const double* overlap,const double* density) {
    std::vector<double> m(3*size_t(rows),0.);
    for(int r=0;r<rows;r++) for(int j=0;j<num[r];j++) {
        const size_t i=size_t(ptr[r])+j;require(i<size_t(capacity),"Spin-constraint entry outside the density");
        remag::spin_constraint_moment(spins,capacity,i,overlap[i],density,&m[3*size_t(r)]);
    }
    return m;
}
// With a penalty, the field of every H follows the density that built it.
void penalize(const std::vector<double>& moments) {
    for(int r=0;r<state.rows;r++) {
        const int s=state.site[r];double* f=&state.orbital_field[3*size_t(r)];
        if(s<0) {f[0]=f[1]=f[2]=0.;continue;}
        double m[3];remag::spin_constraint_site_moment(s,state.rows,state.site.data(),moments.data(),m);
        remag::spin_constraint_penalized_field(&state.site_fields[3*size_t(s)],&state.targets[3*size_t(s)],
            state.penalty,m,f);
    }
}
}
namespace remag {
bool spin_constraint_active() {return state.configured;}
SpinConstraintTrial spin_constraint_trial(int rows) {
    if(!state.configured) return {};
    current();require(rows==state.rows&&!state.recorded,"Spin-constraint field does not match this SCF");
    state.applied=true;
    return {state.sites,state.penalty,&state.site,&state.site_fields,&state.targets,&state.orbital_field};
}
void spin_constraint_record_moments(uint64_t epoch,const double* orbital_moments,int rows) {
    current();
    require(state.applied&&!state.recorded&&rows==state.rows&&orbital_moments&&epoch,"Spin-constraint moments without an applied field");
    state.orbital_moments.assign(orbital_moments,orbital_moments+3*size_t(rows));state.epoch=epoch;state.recorded=true;
}
}
// Integers: direction (0 steepest, 1 BFGS, 2 multipliers), line (0 golden, 1 secant),
//           criteria (1 angle, 2 torque, 3 both), maximum evaluations, retain geometry.
// Reals:    initial field, energy tolerance, slope ratio, angle tolerance (rad),
//           torque tolerance, minimum moment, maximum field, coarse angle (rad)
//           and torque tolerances (both zero without a coarse tier), penalty
//           rho (Ry/muB^2; 0 for the paper's plain Lagrangian). Ry units.
extern "C" int remag_spin_constraint_configure(int sites,const double* angles,int rows,const int* orbital_site,
        const int* options,const double* values) {
    try {
        require(!state.configured&&sites>0&&rows>0&&angles&&orbital_site&&options&&values,"Invalid spin-constraint configuration");
        auto& s=state.settings;s={};s.sites=sites;s.direction=options[0];s.line=options[1];s.criteria=options[2];
        s.maximum_evaluations=options[3];s.initial_field=values[0];s.energy_tolerance=values[1];s.slope_ratio=values[2];
        s.angle_tolerance=values[3];s.torque_tolerance=values[4];s.minimum_moment=values[5];s.maximum_field=values[6];
        s.coarse_angle_tolerance=values[7];s.coarse_torque_tolerance=values[8];s.penalty=values[9];
        require(options[4]==0||options[4]==1,"Invalid spin-constraint geometry retention");
        std::vector<remag::SpinConstraintSite> frames;state.targets.clear();
        for(int i=0;i<sites;i++) {
            require(std::isfinite(angles[2*i])&&std::isfinite(angles[2*i+1]),"Nonfinite spin-constraint direction");
            frames.push_back(remag::spin_constraint_site(angles[2*i],angles[2*i+1]));
            state.targets.insert(state.targets.end(),frames.back().target,frames.back().target+3);
        }
        std::vector<int> used(sites,0);
        for(int r=0;r<rows;r++) {
            require(orbital_site[r]>=-1&&orbital_site[r]<sites,"Invalid spin-constraint orbital site");
            if(orbital_site[r]>=0) used[orbital_site[r]]=1;
        }
        for(int u:used) require(u,"Spin-constraint site has no orbitals");
        require(state.optimizer.configure(s,frames.data())==remag::spin_constraint_ok,"Invalid spin-constraint settings");
        state.site.assign(orbital_site,orbital_site+rows);state.sites=sites;state.rows=rows;state.retain=options[4];
        state.configured=true;return 0;
    } catch(const std::exception& e) {return failure("configuration",e);}
}
// Stage the optimizer's pending trial for the next SCF.
extern "C" int remag_spin_constraint_begin(int step) {
    try {
        require(state.configured&&!state.finished&&!state.status&&step==state.step+1&&!remag::in_scf(),"Spin-constraint step out of order");
        state.step=step;state.applied=state.recorded=false;state.penalty=state.optimizer.penalty();
        state.site_fields.assign(3*size_t(state.sites),0.);state.optimizer.fields(state.site_fields.data());
        state.orbital_field.assign(3*size_t(state.rows),0.);
        for(int r=0;r<state.rows;r++) if(state.site[r]>=0)
            for(int k=0;k<3;k++) state.orbital_field[3*size_t(r)+k]=state.site_fields[3*size_t(state.site[r])+k];
        return 0;
    } catch(const std::exception& e) {return failure("begin",e);}
}
extern "C" int remag_spin_constraint_fields(int count,double* fields) {
    if(!state.configured||!fields||count!=3*state.sites||state.site_fields.size()!=size_t(count)) return -300;
    std::copy(state.site_fields.begin(),state.site_fields.end(),fields);return 0;
}
// Host-Hamiltonian producer: add the pending field to the complete host H,
// built from density (the field follows it when a penalty is set).
extern "C" int remag_spin_constraint_host_add(int rows,int capacity,int spins,const int* num,const int* ptr,
        const int* cols,const int* mapping,const double* overlap,const double* density,double* h) {
    try {
        current();
        require(rows==state.rows&&capacity>0&&(spins==4||spins==8)&&num&&ptr&&cols&&overlap&&density&&h&&!state.recorded,
            "Invalid host spin-constraint Hamiltonian");
        if(state.penalty>0.) penalize(orbital_moments(rows,capacity,spins,num,ptr,overlap,density));
        for(int r=0;r<rows;r++) for(int j=0;j<num[r];j++) {
            const size_t i=size_t(ptr[r])+j;require(i<size_t(capacity),"Host spin-constraint entry outside H");
            const int c=partner(cols[i],rows,mapping);require(c>=0&&c<rows,"Invalid host spin-constraint column");
            remag::spin_constraint_add(spins,capacity,i,overlap[i],&state.orbital_field[3*size_t(r)],&state.orbital_field[3*size_t(c)],h);
        }
        state.applied=true;return 0;
    } catch(const std::exception& e) {return failure("host Hamiltonian",e);}
}
extern "C" int remag_spin_constraint_host_moments(int rows,int capacity,int spins,const int* num,const int* ptr,
        const double* overlap,const double* density) {
    try {
        current();
        require(state.applied&&!state.recorded&&rows==state.rows&&capacity>0&&(spins==4||spins==8)&&num&&ptr&&overlap&&density,
            "Host spin-constraint moments without an applied field");
        state.orbital_moments=orbital_moments(rows,capacity,spins,num,ptr,overlap,density);
        state.epoch=remag::geometry_epoch();state.recorded=true;return 0;
    } catch(const std::exception& e) {return failure("host moments",e);}
}
// Observe the converged SCF of the pending trial. Returns 0, an optimizer
// status (>0, see spin_constraint_optimizer.hpp) or -300. On 0, *done is 1
// when the constraint holds, 2 when the coarse tier ends, else 0.
extern "C" int remag_spin_constraint_observe(double free_energy,int count,double* site_moments,double* lagrangian,int* done) {
    try {
        current();
        require(state.recorded&&state.epoch==remag::geometry_epoch()&&site_moments&&lagrangian&&done&&count==3*state.sites,
            "Spin-constraint observation without final moments");
        std::fill(site_moments,site_moments+count,0.);
        for(int r=0;r<state.rows;r++) if(state.site[r]>=0)
            for(int k=0;k<3;k++) site_moments[3*state.site[r]+k]+=state.orbital_moments[3*size_t(r)+k];
        // Augmented Lagrangian F + b.M + rho/2 |M_perp|^2 (the paper's L when rho=0).
        double value=free_energy;for(int i=0;i<count;i++) value+=state.site_fields[i]*site_moments[i];
        for(int s=0;s<state.sites;s++) {
            const double* m=site_moments+3*s;const double* t=&state.targets[3*size_t(s)];
            const double parallel=m[0]*t[0]+m[1]*t[1]+m[2]*t[2],square=m[0]*m[0]+m[1]*m[1]+m[2]*m[2];
            value+=.5*state.penalty*(square-parallel*parallel);
        }
        *lagrangian=value;const int line=state.optimizer.lines(),coarse=state.optimizer.coarse();
        state.status=state.optimizer.observe(value,site_moments,done);
        const auto& m=state.optimizer.measure();
        state.evaluations.push_back({state.step,line,coarse,state.penalty,free_energy,value,m.angle,m.torque,m.field});
        if(*done==1) state.finished=true;
        return state.status;
    } catch(const std::exception& e) {state.status=-300;return failure("observation",e);}
}
// Measures of the last observation: angle (rad), torque, field (Ry), evaluations, lines.
extern "C" int remag_spin_constraint_measure(double* v) {
    if(!state.configured||!v) return -300;
    const auto& m=state.optimizer.measure();
    v[0]=m.angle;v[1]=m.torque;v[2]=m.field;v[3]=state.optimizer.evaluations();v[4]=state.optimizer.lines();return 0;
}
extern "C" int remag_spin_constraint_retain_geometry() {return state.configured&&state.retain&&state.step>0;}
// 1 while the pending trial belongs to the coarse tier (cheaper SCF settings).
extern "C" int remag_spin_constraint_coarse() {return state.configured&&state.optimizer.coarse();}
// Penalty rho of the pending trial (Ry/muB^2); 0 without constraints.
extern "C" double remag_spin_constraint_penalty() {return state.configured?state.penalty:0.;}
