"""Requested-output corruption and spinor layout must fail closed."""
import struct

import numpy as np
import pytest

from siesta_remag.energy_density_output import read_tsde, compare_outputs
from siesta_remag.terminal_exports import requested_energy_exports
from siesta_remag.output import read_grid, grid_artifact


def test_tsde_spinor_components_and_truncation(tmp_path):
    path = tmp_path/'remag.TSDE'
    def record(values, dtype):
        data = np.asarray(values, dtype=dtype).tobytes()
        marker = struct.pack('<i', len(data))
        return marker+data+marker
    header = record([2, 8, 1, 1, 1], '<i4')+record([2, 1], '<i4')
    header += record([1, 2], '<i4')+record([2], '<i4')
    matrices = b''.join(record([s, s+.1], '<f8')+record([s+.2], '<f8') for s in range(12))
    path.write_bytes(header+matrices+record([-.2], '<f8'))
    result = read_tsde(path)
    assert result['DM'].shape == (8, 3) and result['EDM'].shape == (4, 3)
    assert result['EDM'][0, 2] == 8.2
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='Truncated/corrupt'):
        read_tsde(path)


def test_requested_files_and_unwritten_netcdf_are_rejected(tmp_path):
    nc = pytest.importorskip('netCDF4')
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir(); b.mkdir()
    requests = dict(tsde=False, netcdf=True)
    assert not all(c['passed'] for c in compare_outputs(a, b, requests).values())
    for folder in (a, b):
        with nc.Dataset(folder/'remag.nc', 'w') as file:
            file.createDimension('one', 1)
            file.createGroup('SPARSE')
            file.createVariable('nsc', 'i4', ('one',))  # deliberately unwritten
    checks = compare_outputs(a, b, requests)
    assert not checks['netcdf_read']['passed']
    assert 'unwritten' in checks['netcdf_read']['reason']
    assert not compare_outputs(a, b, dict(tsde=False, netcdf=False))['netcdf_not_requested']['passed']


def test_export_reasons_follow_actual_requests():
    text = 'Write.DM false\nTS.DE.Save true\nCDF.Save true\nWrite.DM.End.Of.Cycle false\n'
    assert requested_energy_exports(text, True) == 2
    assert requested_energy_exports(text, False) == 3
    assert requested_energy_exports(text.replace('Cycle false', 'Cycle true'), True) == 6
    with pytest.raises(ValueError, match='standalone'):
        requested_energy_exports(text+'%include other.fdf\n', True)


def test_netcdf_density_preserves_spin_axes_and_rejects_missing_values(tmp_path):
    nc=pytest.importorskip('netCDF4')
    path=tmp_path/'remag.nc'
    field=np.arange(2*3*4*5,dtype=float).reshape(2,3,4,5)/17
    cell=np.array([[2.,.1,.2],[.3,3.,.4],[.5,.6,4.]])
    with nc.Dataset(path,'w') as dataset:
        dataset.createDimension('xyz',3)
        variable=dataset.createVariable('cell','f8',('xyz','xyz'));variable.unit='Bohr';variable[:]=cell
        group=dataset.createGroup('GRID')
        for name,size in zip(('spin','nz','ny','nx'),field.shape):group.createDimension(name,size)
        variable=group.createVariable('Rho','f8',('spin','nz','ny','nx'));variable.unit='e/Bohr**3';variable[:]=field
        variable=group.createVariable('RhoDelta','f8',('spin','nz','ny','nx'));variable.unit='e/Bohr**3'
    assert grid_artifact(tmp_path/'remag.RHO')==path
    actual_cell,actual_field=read_grid(tmp_path/'remag.RHO')
    assert np.array_equal(actual_cell,cell) and np.array_equal(actual_field,field)
    with pytest.raises(ValueError,match='unwritten'):
        read_grid(tmp_path/'remag.DRHO')
