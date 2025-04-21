#ifndef CALODET_H
#define CALODET_H

#include <vector>

namespace Lorenzetti{

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

}
#endif