"""One finite screening policy, deliberately distinct from precision MAE."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class Settings:
    orb_fmax_ev_ang: float = 0.02
    orb_stress_gpa: float = 0.1
    orb_steps: int = 400
    orb_seconds: float = 300
    dft_fmax_ev_ang: float = 0.04
    dft_stress_gpa: float = 0.5
    dft_steps: int = 30
    dft_seconds: float = 1800
    soc_seconds: float = 600
    total_dft_seconds: float = 3600
    max_scf: int = 100
    dm_tolerance: float = 1e-4
    h_tolerance_ev: float = 1e-3
    mixing_weight: float = 0.03
    mixing_history: int = 8
    mesh_ry: float = 300
    smearing_k: float = 300
    k_spacing_inv_ang: float = 0.35
    max_kpoints: int = 512
    max_atoms: int = 200
    minimum_js_t: float = 0.4
    axis_tolerance_deg: float = 1.0
    branch_rms_mub: float = 0.2
    branch_net_fraction: float = 0.05

    def __post_init__(self):
        integers = {'orb_steps', 'dft_steps', 'max_scf', 'mixing_history', 'max_kpoints', 'max_atoms'}
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            if name in integers and type(value) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.mixing_weight > 1 or self.branch_net_fraction >= 1:
            raise ValueError('Invalid mixing weight or branch fraction')
        if self.axis_tolerance_deg >= 90:
            raise ValueError('Axis tolerance must be below 90 degrees')

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        return cls(**data)

    def to_dict(self):
        return asdict(self)
