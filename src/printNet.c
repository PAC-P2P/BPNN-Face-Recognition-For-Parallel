//
// Created by xx on 17/7/23.
//

#include "printNet.h"


// 打印一维数组
void print_1d(double *a, int n){

    int i;
    for(i = 0; i < n; i++)
    {
        printf("%f ", a[i]);
    }
}

// 打印二维数组
void print_2d(double **a, int rows, int cols){

    int i,j;
    for(i = 0; i < rows; i++)
    {
        for(j = 0;j < cols; j++)
        {
            printf("%f ",a[i][j]);
        }
        printf("\n");
    }

}

// 打印网络
void printNet(BPNN *net, int id){

    printf("\n---------------------------------------------\n");
    printf("id =%d\n", id);

    printf("print net\n");

    printf("input_n:%d,hidden_n:%d,output_n:%d\n",net->input_n,net->hidden_n,net->output_n);

    //printf("input_uints\n");

    //print_1d(net->input_units,net->input_n);

    //printf("\nhidden_units\n");

    //print_1d(net->hidden_units,net->hidden_n);

    printf("\noutput_units\n");

    print_1d(net->output_units,net->output_n);

    //printf("\ninput_weights\n");

    //print_2d(net->input_weights,net->input_n+1,net->hidden_n+1);

    //printf("\nhidden_weights\n");

    //print_2d(net->hidden_weights,net->hidden_n+1,net->output_n+1);

    printf("\n---------------------------------------------\n");

}
