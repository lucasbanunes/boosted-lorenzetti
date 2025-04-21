#ifndef ELECTRON_H
#define ELECTRON_H

#include <vector>

namespace Lorenzetti{

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