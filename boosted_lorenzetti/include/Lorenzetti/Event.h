#ifndef EVENT_H
#define EVENT_H

namespace Lorenzetti {
    
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
}
#endif