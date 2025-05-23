#ifndef CALOCLUSTER_H
#define CALOCLUSTER_H

#include <vector>
#include <ROOT/RVec.hxx>


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
        std::vector<unsigned long int> cell_links;
        int seed_link;
    };
}

#endif