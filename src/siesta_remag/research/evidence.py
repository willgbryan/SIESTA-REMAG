"""Verification of saved electronic-structure evidence."""
from pathlib import Path

from ..corpus import sha256
from ..discovery.common import load


def verify_report(path, collection=None):
    """Verify a saved native or imported discovery report without running a calculation."""
    path = Path(path).resolve()
    root = path.parent
    report = load(path)
    if report.get('format') == 'REMAG-DISCOVERY-OBSERVATION/1':
        from .discovery_observation import verified_observation
        if collection is not None:
            raise ValueError('Discovery observations bind their original collection internally')
        return verified_observation(path)
    if report.get('format') != 'REMAG-EXPERIMENT-RESULT/1':
        raise ValueError('Expected a research calculation report')
    if report['source_sha256'] != report['request']['source_sha256']:
        raise ValueError('Calculation source binding differs')
    collected = None
    if collection is not None:
        collection=Path(collection).resolve()
        manifest=load(collection)
        if collection.name!='archive-manifest.json' or manifest.get('schema')!=1:
            raise ValueError('Expected the original compact archive manifest')
        if not path.is_relative_to(collection.parent):raise ValueError('Report is outside the collection')
        collected=manifest['files']
        if 'report.json' not in collected:raise ValueError('Collection lacks its root report binding')
        key=str(path.relative_to(collection.parent))
        if collected.get(key)!=sha256(path):raise ValueError('Report is not bound by the collection')
        top=load(collection.parent/'report.json')
        if (top['status']=='running' or top['status']!=manifest['status'] or
            top['invocation_sha256']!=manifest['invocation_sha256']):
            raise ValueError('Collection invocation differs')
        for name,expected in collected.items():
            artifact=(collection.parent/name).resolve()
            if not artifact.is_relative_to(collection.parent) or sha256(artifact)!=expected:
                raise ValueError('Collected artifact binding differs: '+name)
        prefix=path.parent.relative_to(collection.parent)
        for stage in report['stages']:
            if stage['accepted']:
                for filename in ('result.json','stdout.log','input.fdf','review.json'):
                    if str(prefix/stage['stage']/filename) not in collected:
                        raise ValueError('Collection lacks accepted stage evidence: '+filename)
                stage_path=prefix/stage['stage']
                if not any(Path(name).parent==stage_path and Path(name).match('fdf*.log') for name in collected):
                    raise ValueError('Collection lacks effective FDF evidence')
    omitted=[]
    for name, expected in report['artifacts'].items():
        artifact = (root/name).resolve()
        if not artifact.is_relative_to(root):raise ValueError('Calculation artifact escapes result directory')
        if not artifact.exists() and collected is not None:
            if str(artifact.relative_to(collection.parent)) in collected:
                raise ValueError('Collected artifact is missing: '+name)
            omitted.append(name)
        elif sha256(artifact) != expected:
            raise ValueError('Calculation artifact binding differs: '+name)
    report=dict(report,verification=dict(mode='compact collection' if collected is not None else 'full local artifacts',
        omitted_artifacts=omitted,collection_manifest_sha256=sha256(collection) if collection else None,
        scope='All collected and locally present report artifacts verified; omitted remote bytes are not reverified'))
    return report

