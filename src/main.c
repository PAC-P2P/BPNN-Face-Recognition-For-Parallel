#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <cstl/cmap.h>
#include "config.h"
#include "backprop.h"
#include "pgmimage.h"
#include "parallelModule.h"
#include "train.h"

int main(int argc, char *argv[]) {

	BPNN * net;
    IMAGELIST *trainlist, *testlist;
    int traintimes, seed, savedelta, list_errors;
    clock_t start, finish;
    int id;   							// 进程id
    int n_p;							// 进程数
	int k=0;
    double time;
    double  sume[1000];   				// 存储各进程误差  [可优化：获得进程数再定义]
    double **hidden_gobal_grad = NULL;	// 隐藏层全局梯度
	double **input_gobal_grad = NULL;	// 输出层全局梯度
	double **hidden_grad = NULL;		// 隐藏层权值梯度
	double **input_grad = NULL;			// 输入层权值梯度


    char netname[30] = NETNAME;
    char trainname[256] = TRAINNAME;
    char testname[256] = TESTNAME;

	seed = SEED;						// 种子
    savedelta = SAVEDELTA;   			// 保存网络的周期
    list_errors = 0;

    /*** Create imagelists ***/
    trainlist = imgl_alloc();
    testlist = imgl_alloc();

    // 创建存储用户名字的map
    map_t *map_user = create_map(char*,int);
    if (map_user == NULL) {
        printf("Failed to create map\n");
        exit(1);
    }

    // map初始化
    map_init(map_user);

    // MPI初始化
	MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&id);

    MPI_Comm_size(MPI_COMM_WORLD,&n_p);

    if(id == 0){
        printf("please input the times of train:\n");
        scanf("%d", &traintimes);

        /*** 开始计时 ***/
        //start = clock();
    }

    MPI_Bcast(&traintimes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    time=-MPI_Wtime();

    /*** If any train, test1, or test2 sets have been specified, then
     load them in. ***/
    if (trainname[0] != '\0')
    {
    	printf("\n-------load trainlist--------\n");
        imgl_load_images_from_textfile_map(trainlist, trainname, id, n_p, map_user);
    }
    if (testname[0] != '\0')
     {
     	printf("\n-------load testlist--------\n");
     	imgl_load_images_from_textfile_map(testlist, testname, id, n_p, map_user);
     }

    /*** 初始化神经网络包 ***/
    /*** Initialize the neural net package ***/
    bpnn_initialize(seed);

    /*** 显示训练集，测试集1，测试集2中图片数量 ***/
    /*** Show number of images in train, test1, test2 ***/
    printf("[%d] %d images in training set\n", id, trainlist->n);
    printf("[%d] %d images in test set\n", id, testlist->n);

    /*** If we've got at least one image to train on, go train the net ***/
    // 假如我们至少有1张图片来训练，那么就开始训练吧！
    //backprop_face(trainlist, testlist, traintimes, savedelta, netname, list_errors, map_user);

    // 选择阶段训练
    backprop_face_choose(trainlist, id, &net, SELETE, SAVEDELTA, sume, map_user);

    if(id == 0)
 	{
	 	k = selectBestNet(sume, id, net, n_p);
    }

	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	if(id == k)
	{
		//printNet(net,id);
		sendNet(net, k);
	}

	if(id == 0)
	{
		recvNet(net, k);
	}

	Bcast_Net(net, 0);

    //正式训练
    backprop_face_parallel(net,&input_grad,&hidden_grad,traintimes,LEARNRATE,IMPULSE,&input_gobal_grad,&hidden_gobal_grad,trainlist, testlist,id, map_user);

	/************** 预测结果 ****************************/
    // 输出测试集中每张图片的匹配情况
    printf("迭代结束后的匹配情况：\n\n");
    printf("测试集1：\n\n");
    result_on_imagelist(net, testlist, 0, map_user);

		time+=MPI_Wtime();
	MPI_Finalize();

	if(id == 0)
	{
		printf("\ntime=%f,k=%d\n",time,k);

	    // /*** 结束计时 ***/
    	// finish = clock();
   	// 	printf( "\nUse %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);
	}

    exit(0);
}
