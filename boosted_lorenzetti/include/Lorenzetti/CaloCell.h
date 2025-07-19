#ifndef CALOCELL_H
#define CALOCELL_H

namespace Lorenzetti{

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
}
#endif