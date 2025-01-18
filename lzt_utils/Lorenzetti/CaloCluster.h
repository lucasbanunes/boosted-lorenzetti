#ifndef CALOCLUSTER_H
#define CALOCLUSTER_H

#include <vector>
#include <ROOT/RVec.hxx>
// #include "Lorenzetti/xAOD.h"
// #include "EventInfo/EventSeedConverter.h"
// #include "CaloCell/CaloCellConverter.h"


namespace Lorenzetti {

    struct CaloCluster_t{
        float e;
        float et;
        float eta;
        float phi;
        float deta;
        float dphi;
        float e0;
        float e1;
        float e2;
        float e3;
        float ehad1;
        float ehad2;
        float ehad3;
        float etot;
        float e233;
        float e237;
        float e277;
        float emaxs1;
        float emaxs2;
        float e2tsts1;
        float reta;
        float rphi;
        float rhad;
        float rhad1;
        float eratio;
        float f0;
        float f1;
        float f2;
        float f3;
        float weta2;
        float secondR;
        float lambdaCenter;
        float secondLambda;
        float fracMax;
        float lateralMom;
        float longitudinalMom;
        std::vector<int> cell_links;
    };

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
    };
}

#endif