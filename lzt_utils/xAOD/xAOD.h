// #ifndef XAOD_H
// #define XAOD_H


// #include <vector>
// #include <string>

// namespace xAOD{

//     struct EventInfo_t{
//         float runNumber;
//         float eventNumber;
//         float avgmu;
//     };

//     struct EventSeed_t{
//         int id;
//         float e;
//         float et;
//         float eta;
//         float phi;
//     };

//     struct TruthParticle_t{
//         int pdgid;
//         int seedid;
//         float e;
//         float et;
//         float eta;
//         float phi;
//         float px;
//         float py;
//         float pz;
//         float vx; // vertex position x (prod_vx)
//         float vy; // vertex position y
//         float vz; // vertex position z
//     };

//     struct CaloCell_t{
//         float e;
//         float et;
//         float tau; 
//         float eta;
//         float phi;
//         float deta;
//         float dphi;
//         int descriptor_link;
//     };

//     struct CaloDetDescriptor_t{
//         int sampling;
//         int detector;
//         float eta;
//         float phi;
//         float deta;
//         float dphi;
//         float e;
//         float tau;
//         float edep;
//         int bcid_start;
//         int bcid_end;
//         float bc_duration;
//         std::vector<float> pulse;
//         std::vector<float> edep_per_bunch;
//         std::vector<float> tof;
//         unsigned long int hash;
//         int cell_link;
//         float z;
//     };
// }
// #endif