# import os
# os.environ['PATH'] = '/root/workspaces/lorenzetti/lorenzetti/build/lib:/physics/geant/build:/physics/root/build/bin:/root/.vscode-server/bin/cd4ee3b1c348a13bafd8f9ad8060705f6d4b9cba/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/workspaces/lorenzetti/lorenzetti/build:/root/workspaces/lorenzetti/lorenzetti/build/executables:/root/workspaces/lorenzetti/lorenzetti/core/GaugiKernel/scripts:/root/workspaces/lorenzetti/lorenzetti/generator/scripts:/root/workspaces/lorenzetti/lorenzetti/geometry/DetectorATLASModel/scripts:/root/workspaces/lorenzetti/lorenzetti/scripts'
# LD_LIBRARY_PATH = [
#     '/root/workspaces/lorenzetti/lorenzetti/build/lib',
#     '/physics/geant/build/BuildProducts/lib',
#     '/physics/root/build/lib'
# ]
# os.environ['LD_LIBRARY_PATH'] = ':'.join(LD_LIBRARY_PATH)
# os.environ['PYTHONPATH'] = '/root/workspaces/lorenzetti/lorenzetti/build/python:/physics/root/build/lib:/physics/root/build/lib:/root/hep:/root/workspaces/lorenzetti/boosted-lorenzetti:/physics/pythia8/lib:/physics/hepmc/build/python/3.8.10'
# from pathlib import Path
# lzt_repo = Path(os.environ['LZT_REPO'])
# lzt_workspace = lzt_repo.parent.resolve()
# lzt_build_dir = lzt_repo / 'build'
# import ROOT
# lzt_lib_dir = lzt_build_dir / 'lib'
# for lib_dir in LD_LIBRARY_PATH:
#     lib_dir = Path(lib_dir)
#     for lib_path in lib_dir.glob('*.so'):
#         print(f'Loading {lib_path}')
#         ROOT.gInterpreter.Load(str(lib_path))
# ROOT.gSystem.Load('/root/workspaces/lorenzetti/lorenzetti/build/lib/liblorenzetti.so')
# Library
import ROOT
from pathlib import Path
ROOT.gSystem.Load('liblorenzetti')
from lzt_utils.dataset import LztDataset
from lzt_utils import INCLUDE_PATHS, LZT_UTILS_INCLUDE_PATHS
import mplhep
import matplotlib.pyplot as plt
ROOT.EnableImplicitMT(2)
plt.style.use(mplhep.style.ROOT)
print('Finished')

# for path in INCLUDE_PATHS:
#     ROOT.gInterpreter.AddIncludePath(path)
# cpp_file_extensions = [
#     '*.h', '*.hxx', '*.hpp', '*.hh'
# ]
# boosted_lzt_cpp = Path(LZT_UTILS_INCLUDE_PATHS[0])
# for file_ext in cpp_file_extensions:
#     print(f'Looking for files with extension {file_ext}')
#     for filename in boosted_lzt_cpp.rglob(file_ext):
#         if 'xAOD' in str(filename):
#             continue
#         print(f'Loading file {filename}')
#         text = filename.read_text()
#         ROOT.gInterpreter.Declare(text)

# for include_dir in INCLUDE_PATHS:
#     include_dir = Path(include_dir)
#     for file_ext in cpp_file_extensions:
#         for filename in include_dir.glob(file_ext):
#             print(f'Loading file {filename}')
#             text = filename.read_text()
#             ROOT.gInterpreter.Declare(text)

# ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<xAOD::EventSeed_t>", "EventInfo/EventSeedConverter.h")
# ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<xAOD::CaloCell_t>", "CaloCell/CaloCellConverter.h")
# ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<xAOD::EventInfo_t>", "EventInfo/EventInfoConverter.h")
# ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<xAOD::CaloDetDescriptor_t>", "CaloCell/CaloDetDescriptorConverter.h")
# ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<xAOD::TruthParticle_t>", "TruthParticle/TruthParticleConverter.h")


lzt_data = Path('/', 'root', 'ext_data', 'lorenzetti')
dataset_name = '2024_11_19_18_53_0000000000_jets'
dataset_path = lzt_data / dataset_name
output_dir = lzt_data / 'checks'
output_dir.mkdir(exist_ok=True, parents=True)
dataset = LztDataset.from_dir(dataset_path)

esd_rdf = dataset.get_esd_rdf(n_files=1)
esd_rdf.Describe()

esd_rdf.Define('CaloCluster_Clusters', "Lorenzetti::makeCaloCluster(EventSeedContainer_Seeds, CaloCellContainer_Cells)")
esd_rdf.Describe()

# options = ROOT.RDF.RSnapshotOptions()
# options.fCompressionLevel = 9
esd_rdf.Snapshot("CollectionTree",
                 "esd_to_aod.root")