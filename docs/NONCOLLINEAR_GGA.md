# Gradient-corrected functionals for noncollinear spin

Gradient-corrected (GGA) exchange-correlation functionals such as PBE are defined for collinear spin: a density of spin-up electrons and a density of spin-down electrons, and their gradients. When the magnetization changes direction in space, as it does with spin-orbit coupling or in a noncollinear magnet, a code has to decide how to evaluate such a functional. REMAG offers three choices through `REMAG.XC.NoncollinearGGA`. They differ in both the energy they evaluate and how its derivative, the exchange-correlation potential, is formed.

| Value | Energy | Potential |
| --- | --- | --- |
| `legacy` (default) | Collinear GGA in the local spin frame | Not the derivative of that energy |
| `local-frame` | The same energy, as Quantum ESPRESSO evaluates it | Its exact derivative |
| `multicollinear` | The multicollinear functional of Pu et al. (2023) | Its exact derivative |

The default keeps upstream SIESTA's results. All three run on the CPU path, with one process, and in REMAG's GPU exchange-correlation code, which serves the resident-grid and device-SCF settings. LDA calculations are unaffected by the choice, except that `multicollinear` reaches the same LDA result by a longer route.

## Why the upstream potential is inconsistent

In the local spin frame, each point of the mesh is treated as collinear along its own magnetization direction m̂. The densities along and against that direction are n± = (n ± |m|)/2, and their gradients are those of n and |m|. SIESTA 5.4.2's exchange-correlation library, libgridxc, evaluates this energy correctly. But its potential (`ggaxc.F90`, where `old_scheme` is fixed to true) leaves out how m̂ depends on the density matrix. The SCF then converges to a density that does not minimize the energy it reports.

The constrained-moment option shows this directly. When a site's moment is held at an angle θ by a field b, the derivative of the free energy is dF/dθ = −(b·e_θ)|M|, where e_θ points along increasing θ. On hcp cobalt, at a 6×6×3 k mesh and 500 K smearing, a central difference of three separately converged energies at 40°, 45° and 50° can be compared with the field at 45°. The finite difference itself underestimates the slope of a sin²θ curve by a factor sin 10°/(10° in radians) = 0.995, so a consistent calculation gives a ratio of 1.005. With LDA the ratio is 1.005. With `legacy` PBE it is 2.38. With `local-frame` PBE it is 1.004, and with `multicollinear` PBE 1.005.

The field-implied slope barely changes between the two PBE schemes, 250 against 245 μeV/rad. The energies are what move: their slope goes from 105 to 244 μeV/rad.

## Local frame, as in Quantum ESPRESSO

`local-frame` (`native/remag_xc.F90.in`) forms n± at every mesh point, runs libgridxc's collinear GGA on those two fields, and maps the two resulting potentials back onto SIESTA's four spin components with the chain rule at each point. That mapping supplies exactly the term `legacy` omits, so the potential is the derivative of the energy evaluated on the mesh.

Where |m| passes through zero, the label "along" can flip. Quantum ESPRESSO avoids this by measuring "along" against one fixed axis, but only when the moments it starts from are all parallel or antiparallel (`PW/src/compute_ux.f90`). REMAG makes the same choice at the first call. It fixes the axis when the magnetization on the mesh is collinear up to sign, meaning one direction carries 99% of the moment tensor. Otherwise, for example in a 120° antiferromagnet, it uses |m| itself and adds no magnetic potential where |m| vanishes. The fixed axis makes the energy slightly non-smooth at the few mesh points where the moment is nearly perpendicular to it. The change in H between SCF steps then flickers near 5×10⁻⁶ eV, so these runs use `SCF.H.Tolerance 1.e-5 eV`. Energies are stable to about 10⁻⁹ eV.

On hcp cobalt with PBE, `local-frame` brings SIESTA into agreement with Quantum ESPRESSO, where `legacy` gave about twice QE's anisotropy; the figures are in [the hcp cobalt benchmark](../benchmarks/spin-constraint-hcp-co/README.md).

## Multicollinear

The local-frame functional is a convention, not something derived from the exact functional. Locally it allows an exchange-correlation magnetic field perpendicular to m only through gradient terms, and in LDA none at all. The exact functional is known to have such a field locally, while its total torque on the system vanishes (Capelle, Vignale and Györffy, *PRL* **87**, 206403 (2001)). The local frame also needs a rule for where |m| = 0.

The multicollinear construction of Pu, Li, Zhang, Jiang, Gao, Xiao, Sun, Zhang and Shao, *Phys. Rev. Research* **5**, 013036 (2023), needs neither a frame nor an axis. For each direction e on the sphere it evaluates the collinear functional on n and the projection s = m·e, adds the integral of s times the derivative of the energy with respect to s, and averages over directions. The added term makes a collinear magnetization give back the collinear functional exactly. The result does not change under a rotation of all spins, is smooth where m = 0, and allows a perpendicular field locally. For LDA it is identical to the local frame.

REMAG evaluates it with the same libgridxc collinear GGA. With Ē(λ) the average of E_col[n, λ m·e] over directions, the multicollinear energy is d/dλ[λĒ(λ)] at λ = 1. The potential follows from d/dλ[λ V_n] and d/dλ[λ² e V_s], where V_n and V_s are half the sum and half the difference of the two collinear potentials. The energy uses only the λ = 1 calls. The λ derivatives use one-sided differences at λ = 1, 1−h and 1−2h with h = 10⁻⁴, never λ > 1, which could make n− negative. The potential is therefore the energy's derivative up to an error of order h². The directions are Gauss-Legendre nodes in cos θ times uniform azimuths, on one hemisphere only, since the average is the same for e and −e. `REMAG.XC.Multicollinear.Polar` sets the polar nodes per hemisphere (default 4), and `REMAG.XC.Multicollinear.Azimuth` the azimuths (default four times as many), so the default is 64 directions. Each direction costs three collinear evaluations per SCF step. Exchange-correlation is a small part of a SIESTA step, but at 64 directions on the CPU it is no longer negligible: on hcp cobalt it added about 45% to the time per step. On the GPU the 192 evaluations are mesh kernels inside the SCF graph.

On hcp cobalt at 6×6×3 and 500 K, `multicollinear` reproduces the checks one would ask of it:
- In LDA at 45° its free energy equals `local-frame`'s to 1×10⁻⁸ eV, as the theorem requires.
- Without spin-orbit coupling, where the moment stays collinear, it equals the collinear GGA (through `local-frame`) to 1×10⁻⁸ eV.
- 16, 64 and 256 directions give the same PBE energy to 1×10⁻⁸ eV.
- Its PBE torque identity ratio is 1.005.

In cobalt with PBE, it differs from `local-frame` by 1.4 μeV per cell at 45° and by 0.2% in the torque, so for this nearly collinear ferromagnet the choice hardly matters.

Being frame-free and consistent does not make the functional more accurate. Pu et al. do not claim that it is, and Gaul (arXiv:2605.31203, 2026) finds that its local torque pattern in chromium clusters disagrees qualitatively with exact-exchange calculations. Quantum ESPRESSO does not offer it, so it cannot be checked against QE. The reference implementation for molecules is the MCfun library used by PySCF.

## On the GPU

The GPU PBE plan (`native/pbe.cu`) evaluates `local-frame` and `multicollinear` with the same arithmetic as the CPU code: the channel densities n±, or the projections along each direction, go through the GPU's collinear PBE, and the potentials are mapped back point by point. The choice is made once, from the input option, before any plan is built. For `local-frame`, the fixed-axis decision uses the CPU rule. It is made from the first density on the mesh that has any magnetization, so the empty density used to warm up the SCF graph cannot fix it. The decision is then kept for the whole run, and the CPU and GPU codes share it, whichever decides first.

The check (`scripts/check_pbe_gpu.py`) compares the GPU results with the CPU construction built on pristine GridXC. It covers a moment that is collinear up to sign along a tilted axis, a noncollinear texture, a plane where the moment vanishes, a nearly unpolarized density and a left-handed cell, run directly and inside a CUDA graph. For `local-frame` the energies agree to 2×10⁻¹³ Ry and the potentials to 1.3×10⁻¹⁵ Ry, and the fixed-axis decision matches in every case. For `multicollinear`, at 16 and 64 directions, the energies agree to 2.2×10⁻¹¹ Ry, the potentials to 6.7×10⁻¹² Ry and the stress to 1.2×10⁻¹² Ry/Bohr³. These are larger than for the local frame because the λ differences divide rounding errors by 2h = 2×10⁻⁴; the check allows 2×10⁻⁸.

