import pyarrow as pa

PYARROW_SCHEMA = pa.schema([
    ('CaloCellContainer_Cells', pa.list_(pa.struct([
        ('descriptor_link', pa.uint64()),
        ('deta', pa.float32()),
        ('dphi', pa.float32()),
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('phi', pa.float32()),
        ('tau', pa.float32())
    ]))),
    ('CaloClusterContainer_Clusters', pa.list_(pa.struct([
        ('cell_links', pa.list_(pa.uint64())),
        ('deta', pa.float32()),
        ('dphi', pa.float32()),
        ('e', pa.float32()),
        ('e0', pa.float32()),
        ('e1', pa.float32()),
        ('e2', pa.float32()),
        ('e233', pa.float32()),
        ('e237', pa.float32()),
        ('e277', pa.float32()),
        ('e2tsts1', pa.float32()),
        ('e3', pa.float32()),
        ('ehad1', pa.float32()),
        ('ehad2', pa.float32()),
        ('ehad3', pa.float32()),
        ('emaxs1', pa.float32()),
        ('emaxs2', pa.float32()),
        ('eratio', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('etot', pa.float32()),
        ('f0', pa.float32()),
        ('f1', pa.float32()),
        ('f2', pa.float32()),
        ('f3', pa.float32()),
        ('fracMax', pa.float32()),
        ('lambdaCenter', pa.float32()),
        ('lateralMom', pa.float32()),
        ('longitudinalMom', pa.float32()),
        ('phi', pa.float32()),
        ('reta', pa.float32()),
        ('rhad', pa.float32()),
        ('rhad1', pa.float32()),
        ('rphi', pa.float32()),
        ('secondLambda', pa.float32()),
        ('secondR', pa.float32()),
        ('seed_link', pa.int32()),
        ('weta2', pa.float32())
    ]))),
    ('CaloDetDescriptorContainer_Cells', pa.list_(pa.struct([
        ('bc_duration', pa.float32()),
        ('bcid_end', pa.int32()),
        ('bcid_start', pa.int32()),
        ('deta', pa.float32()),
        ('detector', pa.int32()),
        ('dphi', pa.float32()),
        ('e', pa.float32()),
        ('edep', pa.float32()),
        ('edep_per_bunch', pa.list_(pa.float32())),
        ('eta', pa.float32()),
        ('hash', pa.uint64()),
        ('phi', pa.float32()),
        ('pulse', pa.list_(pa.float32())),
        ('sampling', pa.int32()),
        ('tau', pa.float32()),
        ('tof', pa.list_(pa.float32())),
        ('z', pa.float32())
    ]))),
    ('CaloRingsContainer_Rings', pa.list_(pa.struct([
        ('cluster_link', pa.int32()),
        ('rings', pa.list_(pa.float32()))
    ]))),
    ('ElectronContainer_Electrons', pa.list_(pa.struct([
        ('cluster_link', pa.int32()),
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('phi', pa.float32()),
        ('isEM', pa.list_(pa.bool_())),
    ]))),
    ('EventInfoContainer_Events', pa.list_(pa.struct([
        ('avgmu', pa.float32()),
        ('eventNumber', pa.float32()),
        ('runNumber', pa.float32())
    ]))),
    ('SeedContainer_Seeds', pa.list_(pa.struct([
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('id', pa.int32()),
        ('phi', pa.float32())
    ]))),
    ('TruthParticleContainer_Particles', pa.list_(pa.struct([
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('pdgid', pa.int32()),
        ('phi', pa.float32()),
        ('px', pa.float32()),
        ('py', pa.float32()),
        ('pz', pa.float32()),
        ('seedid', pa.int32()),
        ('vx', pa.float32()),
        ('vy', pa.float32()),
        ('vz', pa.float32())
    ])))
])

STRUCTS = [
    'CaloCellContainer_Cells',
    'CaloClusterContainer_Clusters',
    'CaloDetDescriptorContainer_Cells',
    'CaloRingsContainer_Rings',
    'ElectronContainer_Electrons',
    'EventInfoContainer_Events',
    'EventSeedContainer_Seeds',
    'TruthParticleContainer_Particles'
]


def EventInfoContainer_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'avgmu': d.avgmu,
            'eventNumber': d.eventNumber,
            'runNumber': d.runNumber})
    return new_data


def CaloCellContainer_Cells_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'descriptor_link': d.descriptor_link,
            'deta': d.deta,
            'dphi': d.dphi,
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'phi': d.phi,
            'tau': d.tau})
    return new_data


def CaloClusterContainer_Clusters_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'cell_links': np.array(d.cell_links, dtype=np.uint64).tolist(),
            'deta': d.deta,
            'dphi': d.dphi,
            'e': d.e,
            'e0': d.e0,
            'e1': d.e1,
            'e2': d.e2,
            'e233': d.e233,
            'e237': d.e237,
            'e277': d.e277,
            'e2tsts1': d.e2tsts1,
            'e3': d.e3,
            'ehad1': d.ehad1,
            'ehad2': d.ehad2,
            'ehad3': d.ehad3,
            'emaxs1': d.emaxs1,
            'emaxs2': d.emaxs2,
            'eratio': d.eratio,
            'et': d.et,
            'eta': d.eta,
            'etot': d.etot,
            'f0': d.f0,
            'f1': d.f1,
            'f2': d.f2,
            'f3': d.f3,
            'fracMax': d.fracMax,
            'lambdaCenter': d.lambdaCenter,
            'lateralMom': d.lateralMom,
            'longitudinalMom': d.longitudinalMom,
            'phi': d.phi,
            'reta': d.reta,
            'rhad': d.rhad,
            'rhad1': d.rhad1,
            'rphi': d.rphi,
            'secondLambda': d.secondLambda,
            'secondR': d.secondR,
            # seed_link is an int32, but it is a list in the schema
            'seed_link': d.seed_link,
            'weta2': d.weta2})
    return new_data


def CaloDetDescriptorContainer_Cells_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'bc_duration': d.bc_duration,
            'bcid_end': d.bcid_end,
            'bcid_start': d.bcid_start,
            'deta': d.deta,
            'detector': d.detector,
            'dphi': d.dphi,
            'e': d.e,
            'edep': d.edep,
            'edep_per_bunch': np.array(d.edep_per_bunch).tolist(),
            'eta': d.eta,
            'hash': d.hash,
            'phi': d.phi,
            'pulse': np.array(d.pulse).tolist(),
            'sampling': d.sampling,
            'tau': d.tau,
            'tof': np.array(d.tof).tolist(),
            'z': d.z})
    return new_data


def CaloRingsContainer_Rings_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'cluster_link': d.cluster_link,
            'rings': np.array(d.rings).tolist()})
    return new_data


def ElectronContainer_Electrons_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'cluster_link': d.cluster_link,
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'phi': d.phi,
            'isEM': np.array(d.isEM).tolist()})
    return new_data


def SeedContainer_Seeds_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'id': d.id,
            'phi': d.phi})
    return new_data


def TruthParticleContainer_Particles_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'pdgid': d.pdgid,
            'phi': d.phi,
            'px': d.px,
            'py': d.py,
            'pz': d.pz,
            'seedid': d.seedid,
            'vx': d.vx,
            'vy': d.vy,
            'vz': d.vz})
    return new_data
