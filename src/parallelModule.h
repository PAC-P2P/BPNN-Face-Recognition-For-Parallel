//
// Created by xx on 17/7/23.
//

#ifndef BPNN_PARALLELMODULE_H
#define BPNN_PARALLELMODULE_H

#include <mpi.h>
#include <stdio.h>
#include "backprop.h"

int  selectBestNet(double sume[], int id, BPNN *net, int n_p);
void send_2d(double **arry,int rows,int cols,int where,int desnation);
void sendNet(BPNN*net, int id);
void recv_2d(double **arry,int rows,int cols,int id);
void recvNet(BPNN *net,int id);
void bpnn_adjust_weights_parallel(double **grad, int rows, int cols, double **w, double **oldw, double learning_rate, double momentum);
void reduce_main(double **a,double **b,int rows,int cols);
void grad_calculate(double *delta,int ndelta,double *ly,int nly,double **grad,int n);
void Bcast_2d(double **arry,int rows,int cols,int id);
void Bcast_Net(BPNN *net,int id);

#endif //BPNN_PARALLELMODULE_H
