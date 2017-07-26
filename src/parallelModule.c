//
// Created by xx on 17/7/23.
//

#include "parallelModule.h"


// 选择最佳网络
int  selectBestNet(double sume[], int id, BPNN *net, int n_p){

    int i;
    int k = 0;
    double  max_sumerr;

    MPI_Status status;

    // printf("\n-----------selectBestNet---111-----------\n");

    for(i = 1; i < n_p; i++)
    {
        MPI_Recv(&sume[i], 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
    }

    // printf("\n-----------selectBestNet---222--------------\n");

    //printf("id=%d\n",id);
    //print_1d(sume,n_p);

    max_sumerr = sume[0];

    for(i = 0; i < n_p; i++)
    {
        if(sume[i] > max_sumerr)
        {
            max_sumerr = sume[i];
            k = i;
        }
    }

    return k;
}

void send_2d(double **arry,int rows,int cols,int where,int desnation){

    int i;
    // printf("\nsende_2d--\n");

    for(i=0;i<rows;i++)
    {
        MPI_Send(*(arry+i),cols,MPI_DOUBLE,desnation,where,MPI_COMM_WORLD);
    }
}


// 发送网络
void sendNet(BPNN*net, int id){

    if(id == 0)
    {
        return;
    }

    // printf("\nsendNet--------\n");

    MPI_Send(&net->input_n, 1, MPI_INT, 0, id,MPI_COMM_WORLD);

    MPI_Send(&net->hidden_n, 1, MPI_INT, 0, id,MPI_COMM_WORLD);

    MPI_Send(&net->output_n, 1, MPI_INT, 0, id,MPI_COMM_WORLD);

    // MPI_Send(net->input_units,net->input_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    //MPI_Send(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    //MPI_Send(net->output_units,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    // MPI_Send(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    // MPI_Send(net->output_delta,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    //MPI_Send(net->target,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);
    // printf("\nsend_2d----\n");

    //·¢ËÍ¶þÎ¬Êý×é

    send_2d(net->input_weights, net->input_n+1, net->hidden_n+1, id, 0);

    send_2d(net->hidden_weights, net->hidden_n+1, net->output_n+1, id, 0);

    //send_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id,0);

    //send_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id,0);
    // printf("\nsendNet done--------\n");

}

// 接收二维数组
void recv_2d(double **arry,int rows,int cols,int id){
    //printf("\nrecv_2d--\n");
    int i;

    MPI_Status status;

    for(i=0;i<rows;i++){

        MPI_Recv(*(arry+i),cols,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);}

}

//  接收网络
void recvNet(BPNN *net,int id){

    if(id==0)
    {
        return;
    }

    // printf("\nrecv Net--------\n");

    MPI_Status status;

    MPI_Recv(&net->input_n, 1, MPI_INT, id, id, MPI_COMM_WORLD, &status);

    MPI_Recv(&net->hidden_n, 1, MPI_INT, id, id, MPI_COMM_WORLD, &status);

    MPI_Recv(&net->output_n, 1, MPI_INT, id, id, MPI_COMM_WORLD, &status);

    //MPI_Recv(net->input_units,net->input_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

    //MPI_Recv(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

    //MPI_Recv(net->output_units,net->output_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

    //MPI_Recv(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

    //MPI_Recv(net->output_delta,net->output_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

    //MPI_Recv(net->target,net->output_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);
    // printf("\nrecv_2d---------\n");

    //½ÓÊÕ¶þÎ¬Êý×é

    recv_2d(net->input_weights, net->input_n+1, net->hidden_n+1, id);

    recv_2d(net->hidden_weights, net->hidden_n+1, net->output_n+1, id);

    //recv_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id);

    //recv_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id);
    // printf("\nrecv Net done------\n");
}

// 调整权值
void bpnn_adjust_weights_parallel(double **grad, int rows, int cols, double **w, double **oldw, double learning_rate, double momentum)
{
    //printf("\n------adjust weight-----\n");
    double new_dw;
    int k, j;

    // 遍历输出层单元、隐藏层单元
    for (j = 1; j <=cols; j++)
    {
        // 遍历隐藏层单元、输入层单元
        for (k = 0; k <=rows; k++)
        {
            // 新的权值增量
            new_dw = ((learning_rate *  grad[k][j]) + (momentum * oldw[k][j]));
            //printf("\n------adjust weight 1-----\n");
            w[k][j] += new_dw;

            //保存权值增量，用于下次迭代时权值的更新（与冲量相乘，加到权值增量中）
            oldw[k][j] = new_dw;
        }
    }
//printf("\n------adjust done-----\n");
}

// 规约到主进程
void reduce_main(double **a,double **b,int rows,int cols){
    int i,j;
    for(i=0;i<rows;i++)
        for(j=0;j<cols;j++)
            MPI_Reduce(*(a+i)+j,*(b+i)+j,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    //printf("\n------reduce done------\n");
}

// 计算梯度
void grad_calculate(double *delta,int ndelta,double *ly,int nly,double **grad,int n)
{
    int k, j;

    ly[0] = 1.0;

    for (j = 1; j <= ndelta; j++) // ±éÀúÊä³ö²ãµ¥Ôª / Òþ²Ø²ãµ¥Ôª
    {
        for (k = 0; k <= nly; k++)  // ±éÀúÒþ²Ø²ãµ¥Ôª / ÊäÈë²ãµ¥Ôª
        {
            // ÐÂµÄÈ¨ÖµÔöÁ¿
            if(n==0)
            {
                grad[k][j]=delta[j] * ly[k];
            }
            else
            {
                grad[k][j]+=delta[j]*ly[k];
            }
            //if(n==0)
            // printf("%f   ",grad[k][j]);
        }
    }
}

//  广播二维数组
void Bcast_2d(double **arry,int rows,int cols,int id){
    int i;
    for(i=0;i<rows;i++)
        MPI_Bcast(*(arry+i),cols,MPI_DOUBLE,id,MPI_COMM_WORLD);
}

// 广播网络
void Bcast_Net(BPNN *net,int id){
    MPI_Bcast(&net->output_n,1,MPI_INT,id,MPI_COMM_WORLD);
    MPI_Bcast(&net->hidden_n,1,MPI_INT,id,MPI_COMM_WORLD);
    MPI_Bcast(&net->output_n,1,MPI_INT,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->input_units,net->input_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->output_units,net->output_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->output_delta,net->output_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->target,net->output_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //¹ã²¥¶þÎ¬Êý×é
    Bcast_2d(net->input_weights,net->input_n+1,net->hidden_n+1,id);
    Bcast_2d(net->hidden_weights,net->hidden_n+1,net->output_n+1,id);
    //Bcast_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id);
    //Bcast_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id);
}
