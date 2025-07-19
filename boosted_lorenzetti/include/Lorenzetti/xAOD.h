#ifndef XAOD_H
#define XAOD_H

#include <vector>
#include <string>
#include <ROOT/RVec.hxx>

namespace xAOD{

    struct EventInfo_t{
        float runNumber;
        float eventNumber;
        float avgmu;
    };

    struct Seed_t{
        int id;
        float e;
        float et;
        float eta;
        float phi;
    };

    struct TruthParticle_t{
        int pdgid;
        int seedid;
        float e;
        float et;
        float eta;
        float phi;
        float px;
        float py;
        float pz;
        float vx; // vertex position x (prod_vx)
        float vy; // vertex position y
        float vz; // vertex position z
    };

    struct CaloCell_t{
        float e;
        float et;
        float tau; 
        float eta;
        float phi;
        float deta;
        float dphi;
        unsigned long int descriptor_link; // NOTE: lets use hash as link
    };

    struct CaloDetDescriptor_t{

        int sampling;
        int detector;
        float eta;
        float phi;
        float deta;
        float dphi;
        float e;
        float tau;
        float edep;
        int bcid_start;
        int bcid_end;
        float bc_duration;
        std::vector<float> pulse;
        std::vector<float> edep_per_bunch;
        std::vector<float> tof;
        unsigned long int hash;
        float z;
    };

    struct CaloRings_t{
        int cluster_link;
        std::vector<float> rings;
    };

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
        std::vector<unsigned long int> cell_links;
        int seed_link;
    };

    struct Electron_t{
        int cluster_link;
        float e;
        float et;
        float eta;
        float phi;
        std::vector<bool> isEM;
    };
}
#endif