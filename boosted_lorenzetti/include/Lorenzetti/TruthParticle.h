#ifndef TRUTHPARTICLE_H
#define TRUTHPARTICLE_H

namespace Lorenzetti{

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
}
#endif