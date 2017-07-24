//
// Created by xx on 17/7/23.
//

#ifndef BPNN_CONFIG_H
#define BPNN_CONFIG_H

// 训练集名
#define TRAINNAME "all_train.list"

// 测试集名
#define TESTNAME "all_test.list"

// 网络名
#define NETNAME "BPNN.net"

// 随机产生器的种子
#define SEED 102194

// 选择阶段训练次数
#define SELETE   10

// 保存网络的周期(每训练几次保存网络)
#define SAVEDELTA 100

// 学习速率
#define LEARNRATE 0.3

// 冲量
#define IMPULSE 0.3

#endif //BPNN_CONFIG_H
