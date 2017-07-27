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

		// IMAGE *iimg;
		BPNN * net;
    IMAGELIST *trainlist, *testlist;
    int traintimes, seed, savedelta, list_errors;
    clock_t start, finish;
    int id;   							// 进程id
    int n_p;							// 进程数
		int k=0;
		int perTimes= SELETE, i, iterateTimes, userNum, userNumMax, train_n, imagesize;
    double time;
		double totalCorrect = 0, correctRate = 0, error = 0;
		double correctRateSum, errorSum;
    double  sume[100];   				// 存储各进程误差  [可优化：获得进程数再定义]
    double **hidden_gobal_grad = NULL;	// 隐藏层全局梯度
		double **input_gobal_grad = NULL;	// 输出层全局梯度
		double **hidden_grad = NULL;		// 隐藏层权值梯度
		double **input_grad = NULL;			// 输入层权值梯度

    char netname[30] = NETNAME;
    char trainname[256] = TRAINNAME;
    char testname[256] = TESTNAME;

		seed = SEED;									// 种子
    savedelta = SAVEDELTA;   			// 保存网络的周期
    list_errors = 0;

    /*** Create imagelists ***/
    trainlist = imgl_alloc();
    testlist = imgl_alloc();

		/************** 创建map并加载训练集的人进map****************************/
    // // 创建存储用户名字的map
    // map_t *map_user = create_map(char*,int);
		//
		// if (map_user == NULL) {
    //     printf("Failed to create map_user\n");
    //     exit(1);
    // }

    // map初始化
    // map_init(map_user);

		// IMAGE分配空间
		// iimg = (IMAGE *) malloc ((unsigned) (sizeof (IMAGE)));

		/************** 加载训练集进map ****************************/
		// 加载训练集进map，同时获得第一张图片（大小）
		// imgl_load_images_from_textfile_map(iimg, trainname, map_user);

		// 获得图片大小，即网络输入结点数
		// imagesize = ROWS(iimg) * COLS(iimg);

		// 获取map中人数，即网络隐藏、输出结点数
		// userNum = map_size(map_user);

		/************** 存储map ****************************/
		//
		// char ** mapUserArr = MapArrayInit(userNum+1, USERNAMESIZE);
		//
		// MapSaveToArray(map_user, mapUserArr);
		//
		// map_destroy(map_user);

		/************** 开始并行****************************/

    // MPI初始化
		MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&id);

    MPI_Comm_size(MPI_COMM_WORLD,&n_p);

    if(id == 0){
        printf("please input the times of train:\n");
        scanf("%d", &traintimes);

				if(traintimes < perTimes || traintimes % perTimes != 0)
				{
					printf("训练次数应该大于 %d 次，且为 %d 的整数倍！\n",perTimes, perTimes);
					exit(-1);
				}

    }

		MPI_Bcast(&traintimes, 1, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		/*** 初始化神经网络包 ***/
		bpnn_initialize(seed+id*999);

		/************** 创建map并加载	训练集的人进map****************************/
		// 创建存储用户名字的map
		map_t *map_user = create_map(char*,int);

		if (map_user == NULL) {
			 printf("Failed to create map_user\n");
			 exit(1);
		}

		// map初始化
		map_init(map_user);

		/************** 加载训练集进map ****************************/
		// 加载训练集进map，同时获得第一张图片（大小）
		imgl_load_images_from_textfile_map(&imagesize, trainname, map_user);

		// 获取map中人数，即网络隐藏、输出结点数
		userNum = map_size(map_user);

		/************** 创建网络 ****************************/

		net=bpnn_create(imagesize,userNum,userNum);

		/************** 结束创建网络 ****************************/

		/************** 开始计时****************************/

    time=-MPI_Wtime();

		/************** 加载训练集、测试集 ****************************/
    if (trainname[0] != '\0')
    {
    	printf("\n-------load trainlist--------\n");
      imgl_load_images_from_textfile(trainlist, trainname, id, n_p);
    }
    if (testname[0] != '\0')
     {
     	printf("\n-------load testlist--------\n");
     	imgl_load_images_from_textfile(testlist, testname, id, n_p);
     }

  	/************** 显示训练集，测试集中图片数量****************************/
    printf("[%d] %d images in training set\n", id, trainlist->n);
    printf("[%d] %d images in test set\n", id, testlist->n);

		/************** 并行创建map ****************************/

		// map_t * map_user_parallel = create_map(char*,int);
		// if (map_user_parallel == NULL) {
    //     printf("Failed to create map\n");
    //     exit(1);
    // }
		// map_init(map_user_parallel);
		//
		// ArrayInsertToMap(mapUserArr, userNum+1, map_user_parallel);


		/************** 开始训练 ****************************/

		train_n = trainlist->n;

		iterateTimes = traintimes / perTimes;

		
		for (i = 1; i <= iterateTimes; i++) {

			// 选择阶段训练
	    backprop_face_choose(trainlist, train_n, id, net, perTimes, i, SAVEDELTA, &totalCorrect, map_user);

			if(id==0){
			    sume[0]=totalCorrect;
			}
			else{
			    MPI_Send(&totalCorrect,1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);
			}

			MPI_Barrier(MPI_COMM_WORLD);

			if(id == 0)
	 		{
		 		k = selectBestNet(sume, id, net, n_p);

				// if(i == iterateTimes) break;
	    }

			MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

			MPI_Barrier(MPI_COMM_WORLD);

			if(id == k)
			{
				sendNet(net, k);
			}

			if(id == 0)
			{
				recvNet(net, k);
			}

			Bcast_Net(net, 0);
		}

		/************** 结束训练 ****************************/

    //正式训练
    //backprop_face_parallel(net,&input_grad,&hidden_grad,traintimes,LEARNRATE,IMPULSE,&input_gobal_grad,&hidden_gobal_grad,trainlist, testlist,id, map_user);

	// if(id == k)
	// {
		/************** 预测结果 ****************************/
	    // 输出测试集中每张图片的匹配情况
	    printf("迭代结束后的匹配情况：\n\n");
	    printf("[%d] 测试集1：\n\n", id);
	    result_on_imagelist(net, testlist, 0, map_user, &correctRate, &error);
	// }

	// if (id == 0) {
		MPI_Reduce(&correctRate,&correctRateSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&error,&errorSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	// }

	// MPI_Barrier(MPI_COMM_WORLD);

	// MPI_Barrier(MPI_COMM_WORLD);

	/************** 结束计时****************************/
	time+=MPI_Wtime();


	if(id == 0)
	{
		printf("\nAccuracy rate of: %g%%  Average error: %g \n\n", correctRateSum/n_p, errorSum/n_p);

		printf("Used  %f  seconds\n\n", time);

		/************** 保存网络 ****************************/
		bpnn_save(net, netname);

		printf("\n");
	}

	/************** 释放内存 ****************************/

	MPI_Barrier(MPI_COMM_WORLD);
	imgl_free(trainlist);
	imgl_free(testlist);
	// bpnn_free(net);
	map_destroy(map_user);

	MPI_Finalize();

	// MapArrayDestroy(mapUserArr, userNum+1);

  exit(0);
}
