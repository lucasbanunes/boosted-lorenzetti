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
