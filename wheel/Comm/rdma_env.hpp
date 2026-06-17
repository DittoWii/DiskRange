#pragma once

// #include "comm/RDMA/CommManager.h"
// #include "comm/RDMA/CommManager.h"
// class comm_app::CommManager;

extern int r_server_id;
extern int r_server_num;

// class RDMAEnv {
//  public:
//   int serverId_;
//   int serverNum_;

//   // 构造函数
//   RDMAEnv() {}

//   // 初始化
//   void rdma_env_init() {
//     comm_app::DistConfig config;
//     config.machineNR = 3;
//     config.messagingThreadCount = 20;
//     comm_app::CommManager *comm_cm =
//     comm_app::CommManager::getInstance(config); serverId_ =
//     comm_cm->getMyNodeID(); serverNum_ = comm_cm->getClusterSize();
//   }
// };

// inline RDMAEnv &global_rdmaEnv() {
//   static RDMAEnv rdma_env;
//   return rdma_env;
// }
// inline int r_server_id { return global_rdmaEnv().serverId_; }
// inline int r_server_num { return global_rdmaEnv().serverNum_; }
