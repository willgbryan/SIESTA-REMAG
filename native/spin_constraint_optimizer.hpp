// SPDX-License-Identifier: GPL-3.0-or-later
// Outer multiplier search for site spin constraints, after Cuadrado, Pruneda,
// Garcia and Ordejon, J. Phys. Mater. 1, 015010 (2018): every trial field is
// one converged SCF, and the multipliers maximize
//   L(x) = F[D_x] + sum_s b_s(x).M_s[D_x],  b_s = x_2s e1_s + x_2s+1 e2_s,
// where e1_s,e2_s span the plane normal to the target direction t_s. Because
// D_x minimizes F + b.M and M is linear in D, L is concave in x and
// dL/dx = (e1_s.M_s, e2_s.M_s): the transverse moments. The paper's scheme is
// steepest ascent with bracketing and golden-section line maxima on L. Two
// opt-in variants use that exact gradient instead: a safeguarded secant line
// search on dL/dt and BFGS directions. The default is the method of
// multipliers (augmented Lagrangian): each SCF also carries rho/2 |M_perp|^2,
// which stiffens soft rotation modes so every trial converges, and the field
// update is b <- b + rho M_perp with no line search. The augmented dual has
// the same maximizer and gradient; its curvature is -(chi^-1 + rho)^-1, so
// one step converges at the rate 1/(1+rho chi). When an SCF does not reduce
// the transverse moment fourfold, rho grows fourfold (Bertsekas), up to 1e3x.
// An optional coarse tier runs the early trials with cheaper SCF settings to
// looser tolerances; the same multipliers are then re-evaluated with full
// settings and the search finishes from there. Host only; the caller runs every SCF.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace remag {
enum SpinConstraintDirection {spin_constraint_steepest=0,spin_constraint_bfgs=1,spin_constraint_multipliers=2};
enum SpinConstraintLine {spin_constraint_golden=0,spin_constraint_secant=1};
enum SpinConstraintCriterion {spin_constraint_angle=1,spin_constraint_torque=2};
// Status codes are sticky: once nonzero, observe refuses further work.
enum SpinConstraintStatus {
    spin_constraint_ok=0,spin_constraint_invalid=1,spin_constraint_nonfinite=2,
    spin_constraint_collapsed=3,spin_constraint_antiparallel=4,spin_constraint_field_limit=5,
    spin_constraint_evaluation_limit=6,spin_constraint_no_ascent=7,spin_constraint_order=8};
struct SpinConstraintSettings {
    int sites=0,direction=spin_constraint_steepest,line=spin_constraint_golden;
    int criteria=spin_constraint_angle,maximum_evaluations=60;
    double initial_field=0.;      // Ry: largest site-field change of the first trial
    double energy_tolerance=0.;   // Ry: golden bracket spread that ends a line
    double slope_ratio=0.1;       // secant: accept |dL/dt| <= ratio*dL/dt(0)
    double angle_tolerance=0.;    // rad, between M_s and t_s
    double torque_tolerance=0.;   // muB, |M_s x t_s|
    double minimum_moment=0.;     // muB
    double maximum_field=0.;      // Ry, any |b_s|
    double penalty=0.;            // Ry/muB^2, rho of the augmented Lagrangian (0: none)
    // Coarse tier, disabled when both are zero; otherwise looser than the above.
    double coarse_angle_tolerance=0.,coarse_torque_tolerance=0.;
};
struct SpinConstraintSite {double target[3],e1[3],e2[3];};
// Right-handed frame (e1,e2,t) = (theta hat, phi hat, r hat); smooth at the poles.
inline SpinConstraintSite spin_constraint_site(double theta,double phi) {
    const double st=std::sin(theta),ct=std::cos(theta),sp=std::sin(phi),cp=std::cos(phi);
    return {{st*cp,st*sp,ct},{ct*cp,ct*sp,-st},{-sp,cp,0.}};
}
// Worst site of the last observation. Collapsed or antiparallel moments are
// valid intermediate trials; they only fail a search that would end there.
struct SpinConstraintMeasure {double angle=0.,torque=0.,field=0.;bool collapsed=false,antiparallel=false;};

class SpinConstraintOptimizer {
public:
    // Valid settings and sites leave the first trial x=0 pending.
    int configure(const SpinConstraintSettings& settings,const SpinConstraintSite* sites) {
        const auto& s=settings;
        auto positive=[](double v){return std::isfinite(v)&&v>0.;};
        if(s.sites<1||s.sites>4096||!sites||s.direction<0||s.direction>2||(s.line!=0&&s.line!=1)||
           !std::isfinite(s.penalty)||s.penalty<0.||(s.direction==spin_constraint_multipliers&&!(s.penalty>0.))||
           s.criteria<1||s.criteria>3||s.maximum_evaluations<2||!positive(s.initial_field)||
           !positive(s.energy_tolerance)||!positive(s.slope_ratio)||s.slope_ratio>=1.||
           !positive(s.angle_tolerance)||!positive(s.torque_tolerance)||!(s.minimum_moment>=0.)||
           !std::isfinite(s.minimum_moment)||!positive(s.maximum_field)||s.initial_field>s.maximum_field)
            return status_=spin_constraint_invalid;
        // Golden section compares energies along a line, and a coarse SCF's
        // energy noise exceeds any useful bracket tolerance: the coarse tier
        // needs an update that uses only the gradient.
        const bool coarse=s.coarse_angle_tolerance!=0.||s.coarse_torque_tolerance!=0.;
        if(coarse&&((s.line!=spin_constraint_secant&&s.direction!=spin_constraint_multipliers)||!positive(s.coarse_angle_tolerance)||!positive(s.coarse_torque_tolerance)||
                    s.coarse_angle_tolerance<s.angle_tolerance||s.coarse_torque_tolerance<s.torque_tolerance))
            return status_=spin_constraint_invalid;
        for(int i=0;i<s.sites;i++) {
            const auto& f=sites[i];
            const double checks[6]={dot(f.target,f.target)-1.,dot(f.e1,f.e1)-1.,dot(f.e2,f.e2)-1.,
                                    dot(f.target,f.e1),dot(f.target,f.e2),dot(f.e1,f.e2)};
            for(double c:checks) if(!(std::fabs(c)<=1.e-12)) return status_=spin_constraint_invalid;
        }
        settings_=s;sites_.assign(sites,sites+s.sites);n_=2*s.sites;
        x_.assign(n_,0.);trial_.assign(n_,0.);base_g_.assign(n_,0.);d_.assign(n_,0.);
        inverse_.assign(size_t(n_)*n_,0.);for(int i=0;i<n_;i++)inverse_[size_t(i)*n_+i]=1.;
        scaled_=false;evaluations_=lines_=0;stage_=Stage::first;status_=spin_constraint_ok;converged_=false;coarse_=coarse;
        rho_=s.penalty;previous_torque_=0.;
        return status_;
    }
    int status() const {return status_;}
    bool converged() const {return converged_;}
    // True while trials belong to the coarse tier.
    bool coarse() const {return coarse_;}
    // Penalty rho for the pending trial (Ry/muB^2).
    double penalty() const {return rho_;}
    int evaluations() const {return evaluations_;}
    int lines() const {return lines_;}
    // Multipliers of the pending trial, two per site.
    const std::vector<double>& trial() const {return trial_;}
    // Site fields b_s in Ry, three per site, for the pending trial.
    void fields(double* b) const {for(int s=0;s<settings_.sites;s++)field(trial_.data(),s,b+3*s);}
    const SpinConstraintMeasure& measure() const {return measure_;}
    // Record one converged SCF at the pending trial. On success *done is 1 when
    // the constraint is met, 2 when the coarse tier ends (the same fields are
    // pending for a full evaluation), else 0 with the next fields pending.
    int observe(double lagrangian,const double* moments,int* done) {
        if(status_) return status_;
        if(!done||!moments||stage_==Stage::idle) return status_=spin_constraint_order;
        *done=0;
        if(!std::isfinite(lagrangian)) return status_=spin_constraint_nonfinite;
        for(int i=0;i<3*settings_.sites;i++) if(!std::isfinite(moments[i])) return status_=spin_constraint_nonfinite;
        evaluations_++;
        Point p{0.,lagrangian,0.,std::vector<double>(n_)};
        assess(moments,p.g);
        // A transverse-free state with a collapsed or antiparallel site is a
        // stationary point of the wrong kind: it satisfies g=0, not the target.
        if(measure_.torque<=settings_.torque_tolerance&&(measure_.collapsed||measure_.antiparallel))
            return status_=measure_.collapsed?spin_constraint_collapsed:spin_constraint_antiparallel;
        if(coarse_&&met(settings_.coarse_angle_tolerance,settings_.coarse_torque_tolerance)) {
            // A coarse SCF cannot certify the result: evaluate this trial again
            // and restart the line there, keeping BFGS curvature but no update.
            coarse_=false;x_=trial_;stage_=Stage::first;*done=2;return status_;
        }
        if(!coarse_&&met(settings_.angle_tolerance,settings_.torque_tolerance)) {
            converged_=true;stage_=Stage::idle;*done=1;return status_;
        }
        if(evaluations_>=settings_.maximum_evaluations) return status_=spin_constraint_evaluation_limit;
        if(settings_.direction==spin_constraint_multipliers) {
            // Method of multipliers: one gradient step of the rho this SCF used.
            for(int i=0;i<n_;i++)trial_[i]+=rho_*p.g[i];
            if(previous_torque_>0.&&measure_.torque>.25*previous_torque_) rho_=std::min(4.*rho_,1.e3*settings_.penalty);
            previous_torque_=measure_.torque;lines_++;x_=trial_;return check_field();
        }
        if(stage_==Stage::first) {begin_line(p,false);return propose();}
        p.t=t_;p.slope=dot(p.g,d_);
        return settings_.line==spin_constraint_golden?golden(p):secant(p);
    }
private:
    enum class Stage {first,expand,contract,refine,idle};
    struct Point {double t,value,slope;std::vector<double> g;};
    static constexpr double ratio=0.38196601125010515; // (3-sqrt(5))/2
    static constexpr double growth=1.6180339887498949;
    static double dot(const double* a,const double* b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
    double dot(const std::vector<double>& a,const std::vector<double>& b) const {
        double v=0.;for(int i=0;i<n_;i++)v+=a[i]*b[i];return v;
    }
    void field(const double* x,int s,double* b) const {
        const auto& f=sites_[s];
        for(int k=0;k<3;k++)b[k]=x[2*s]*f.e1[k]+x[2*s+1]*f.e2[k];
    }
    // Transverse gradient and the paper's angle/torque measures.
    void assess(const double* moments,std::vector<double>& g) {
        measure_={};
        for(int s=0;s<settings_.sites;s++) {
            const double* m=moments+3*s;const auto& f=sites_[s];
            const double parallel=dot(m,f.target);g[2*s]=dot(m,f.e1);g[2*s+1]=dot(m,f.e2);
            const double transverse=std::hypot(g[2*s],g[2*s+1]);
            if(!(std::sqrt(dot(m,m))>settings_.minimum_moment)) measure_.collapsed=true;
            if(!(parallel>0.)) measure_.antiparallel=true;
            measure_.angle=std::max(measure_.angle,std::atan2(transverse,parallel));
            measure_.torque=std::max(measure_.torque,transverse);
            double b[3];field(trial_.data(),s,b);measure_.field=std::max(measure_.field,std::sqrt(dot(b,b)));
        }
    }
    bool met(double angle_tolerance,double torque_tolerance) const {
        if(measure_.collapsed||measure_.antiparallel) return false;
        const bool angle=measure_.angle<=angle_tolerance,torque=measure_.torque<=torque_tolerance;
        switch(settings_.criteria) {
            case spin_constraint_angle: return angle;
            case spin_constraint_torque: return torque;
            default: return angle&&torque;
        }
    }
    double largest_site_step(const std::vector<double>& d) const {
        double v=0.;for(int s=0;s<settings_.sites;s++)v=std::max(v,std::hypot(d[2*s],d[2*s+1]));return v;
    }
    // Start a line from accepted point p (t=0 there). Steepest ascent keeps the
    // previous step's first-order gain; BFGS tries the full quasi-Newton step.
    void begin_line(const Point& p,bool update) {
        const double previous_gain=lines_?t_*dot(base_g_,d_):0.;
        if(update&&settings_.direction==spin_constraint_bfgs) {
            std::vector<double> s(n_),y(n_);
            for(int i=0;i<n_;i++){s[i]=p.t*d_[i];y[i]=base_g_[i]-p.g[i];}
            const double sy=dot(s,y),yy=dot(y,y);
            if(sy>1.e-12*std::sqrt(dot(s,s)*yy)) {
                if(!scaled_){
                    std::fill(inverse_.begin(),inverse_.end(),0.);
                    for(int i=0;i<n_;i++)inverse_[size_t(i)*n_+i]=sy/yy;
                    scaled_=true;
                }
                std::vector<double> hy(n_,0.);
                for(int i=0;i<n_;i++)for(int j=0;j<n_;j++)hy[i]+=inverse_[size_t(i)*n_+j]*y[j];
                const double yhy=dot(y,hy),r=1./sy;
                for(int i=0;i<n_;i++)for(int j=0;j<n_;j++)
                    inverse_[size_t(i)*n_+j]+=(1.+yhy*r)*r*s[i]*s[j]-r*(hy[i]*s[j]+s[i]*hy[j]);
            } else reset_inverse();
        }
        base_g_=p.g;base_value_=p.value;
        if(settings_.direction==spin_constraint_bfgs) {
            std::fill(d_.begin(),d_.end(),0.);
            for(int i=0;i<n_;i++)for(int j=0;j<n_;j++)d_[i]+=inverse_[size_t(i)*n_+j]*p.g[j];
            if(!(dot(d_,p.g)>0.)){reset_inverse();d_=p.g;}
        } else d_=p.g;
        slope0_=dot(d_,p.g);
        const double first=settings_.initial_field/largest_site_step(d_);
        double t=first;
        if(lines_&&settings_.direction==spin_constraint_steepest) t=previous_gain/slope0_;
        if(settings_.direction==spin_constraint_bfgs&&scaled_) t=1.;
        // A single trial never moves any site field by more than the limit.
        t_=std::min(t,settings_.maximum_field/largest_site_step(d_));
        a_={0.,p.value,slope0_,p.g};have_b_=have_c_=false;contractions_=0;
        lines_++;stage_=Stage::expand;
    }
    void reset_inverse() {
        std::fill(inverse_.begin(),inverse_.end(),0.);for(int i=0;i<n_;i++)inverse_[size_t(i)*n_+i]=1.;scaled_=false;
    }
    int propose() {
        for(int i=0;i<n_;i++)trial_[i]=x_[i]+t_*d_[i];
        return check_field();
    }
    int check_field() {
        for(int s=0;s<settings_.sites;s++) {
            double b[3];field(trial_.data(),s,b);
            if(!(std::sqrt(dot(b,b))<=settings_.maximum_field)) return status_=spin_constraint_field_limit;
        }
        return status_;
    }
    int accept(const Point& p) {
        for(int i=0;i<n_;i++)x_[i]+=p.t*d_[i];
        begin_line(p,true);return propose();
    }
    // Paper: bracket a maximum of the concave line L(t), then golden section
    // until the bracket's energy spread is below the tolerance.
    int golden(const Point& p) {
        const double tolerance=settings_.energy_tolerance;
        if(stage_==Stage::expand) {
            if(!have_b_) {
                if(p.value>=a_.value) {b_=p;have_b_=true;t_=b_.t+growth*(b_.t-a_.t);return propose();}
                c_=p;have_c_=true;stage_=Stage::contract;t_=a_.t+ratio*(c_.t-a_.t);return propose();
            }
            if(p.value<b_.value) {c_=p;have_c_=true;stage_=Stage::refine;}
            else {a_=b_;b_=p;t_=b_.t+growth*(b_.t-a_.t);return propose();}
        } else if(stage_==Stage::contract) {
            if(p.value>=a_.value) {b_=p;have_b_=true;stage_=Stage::refine;}
            else {
                // Concavity and dL/dt(0)>0 guarantee an ascent near t=0 unless
                // SCF noise exceeds the line's energy scale.
                if(++contractions_>40||c_.t-a_.t<=1.e-12*t_) return status_=spin_constraint_no_ascent;
                c_=p;t_=a_.t+ratio*(c_.t-a_.t);return propose();
            }
        } else {
            const bool right=p.t>b_.t;
            if(p.value>=b_.value) {(right?a_:c_)=b_;b_=p;}
            else (right?c_:a_)=p;
        }
        if(b_.value-std::min(a_.value,c_.value)<=tolerance||c_.t-a_.t<=1.e-10*b_.t) {
            if(b_.t==0.) return status_=spin_constraint_no_ascent;
            return accept(b_);
        }
        t_=(c_.t-b_.t>b_.t-a_.t)?b_.t+ratio*(c_.t-b_.t):b_.t-ratio*(b_.t-a_.t);
        return propose();
    }
    // Exact-gradient variant: dL/dt = d.g is decreasing on a concave line, so
    // bracket its sign change and use Illinois regula falsi with bisection.
    // While still ascending, extrapolate the zero of dL/dt from the last two
    // slopes (between 1.6x and 8x): trial fields far past the line maximum can
    // switch a moment onto another SCF branch, which no line search can undo.
    int secant(const Point& p) {
        const double accept_slope=settings_.slope_ratio*slope0_;
        if(std::fabs(p.slope)<=accept_slope) return accept(p);
        if(stage_==Stage::expand) {
            if(p.slope>0.) {
                double t=8.*p.t;
                if(a_.slope>p.slope) t=std::min(t,p.t+p.slope*(p.t-a_.t)/(a_.slope-p.slope));
                a_=p;t_=std::max(t,growth*p.t);return propose();
            }
            c_=p;have_c_=true;stage_=Stage::refine;
        } else {
            if(p.slope>0.) {a_=p;if(side_==1)c_.slope*=.5;side_=1;}
            else {c_=p;if(side_==-1)a_.slope*=.5;side_=-1;}
        }
        if(c_.t-a_.t<=1.e-12*c_.t) return status_=spin_constraint_no_ascent;
        double t=a_.t+a_.slope*(c_.t-a_.t)/(a_.slope-c_.slope);
        const double width=c_.t-a_.t;
        t_=std::min(std::max(t,a_.t+.1*width),c_.t-.1*width);
        return propose();
    }
    SpinConstraintSettings settings_{};
    std::vector<SpinConstraintSite> sites_;
    int n_=0,evaluations_=0,lines_=0,status_=spin_constraint_invalid,contractions_=0,side_=0;
    bool converged_=false,scaled_=false,have_b_=false,have_c_=false,coarse_=false;
    Stage stage_=Stage::idle;
    std::vector<double> x_,trial_,base_g_,d_,inverse_;
    double base_value_=0.,slope0_=0.,t_=0.,rho_=0.,previous_torque_=0.;
    Point a_{},b_{},c_{};
    SpinConstraintMeasure measure_{};
};
}
