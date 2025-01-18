#include <vector>
#include "Lorenzetti/CaloCluster.h"
#include "Lorenzetti/Event.h"
#include "Lorenzetti/CaloCell.h"
#include "Lorenzetti/xAOD.h"
#include <ROOT/RVec.hxx>

namespace Lorenzetti {

    CaloCluster_t makeCaloCluster(ROOT::RVec<xAOD::EventSeed_t> &event_seeds, ROOT::RVec<xAOD::CaloCell_t> &calo_cells){
        CaloCluster_t calo_cluster = {
            .e=0,
            .et=0,
            .eta=0,
            .phi=0,
            .deta=0,
            .dphi=0,
            .e0=0,
            .e1=0,
            .e2=0,
            .e3=0,
            .ehad1=0,
            .ehad2=0,
            .ehad3=0,
            .etot=0,
            .e233=0,
            .e237=0,
            .e277=0,
            .emaxs1=0,
            .emaxs2=0,
            .e2tsts1=0,
            .reta=0,
            .rphi=0,
            .rhad=0,
            .rhad1=0,
            .eratio=0,
            .f0=0,
            .f1=0,
            .f2=0,
            .f3=0,
            .weta2=0,
            .secondR=0,
            .lambdaCenter=0,
            .secondLambda=0,
            .fracMax=0,
            .lateralMom=0,
            .longitudinalMom=0
        };
        return calo_cluster;
    }


}