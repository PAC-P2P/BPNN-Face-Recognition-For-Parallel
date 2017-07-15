#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "pgmimage.h"
#include "backprop.h"
#include "imagenet.h"

#include <cstl/cmap.h>

// 选择阶段所训练的次数
#define SELETE   10

// 训练集名
#define TRAINNAME "all_train.list"

// 测试集名
#define TESTNAME "all_test.list"

// 训练种子
#define SEED 201600608+id*1000

// 学习速率
#define LEARNRATE 0.3

// 冲量
#define IMPULSE 0.3

void result_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user);

int evaluate_performance(BPNN *net, double *err);

int performance_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user);

void backprop_face_parallel(IMAGELIST *trainlist, int id, BPNN **net, double sume[], map_t *map_user);

int selectBestNet(double sume[], int id, BPNN *net, int n_p);

void send_2d(double **arry, int rows, int cols, int where, int desnation);

void sendNet(BPNN *net, int id);

void recvNet(BPNN *net, int id);

void recv_2d(double **arry, int rows, int cols, int id);

void printNet(BPNN *net, int id);

void print_1d(double *a, int n);

void print_2d(double **a, int rows, int cols);

void imgl_load_images_from_textfile_id(IMAGELIST *il, char *filename, int id, int n_p);

void imgl_load_images_from_textfile_id_user(IMAGELIST *il, char *filename, int id, int n, map_t *map_user);

void Bcast_2d(double **arry, int rows, int cols, int id);

void Bcast_Net(BPNN *net, int id);

void bpnn_train_parallel(BPNN *net, double ***input_grad, double ***hidden_grad, int epochs, double learning_rate,
                         double momentum,
                         double ***input_gobal_grad, double ***hidden_gobal_grad, IMAGELIST *trainlist, IMAGELIST *testlist, int id, map_t *map_user);

void bpnn_adjust_weights_parallel(double **grad, int rows, int cols, double **w, double **oldw, double learning_rate,
                                  double momentum);

void reduce_main(double **a, double **b, int rows, int cols);

void grad_calculate(double *delta, int ndelta, double *ly, int nly, double **grad, int n);


/**
        采用主从模式训练
**/

int main(int argc, char *argv[]) {

    int id;                         //进程id
    int n_p;                        //进程个数
    int epochs;                     //训练次数
    int epoch;                      //训练到第几次

    BPNN *net;

    IMAGELIST *trainlist = imgl_alloc();     //训练集
    IMAGELIST *testlist=imgl_alloc();	    //测试集

    char trainname[256] = TRAINNAME;      //存储图片路径的文件名
    char testname[256] = TESTNAME;

    double sume[1000];                       //存储各进程误差
    double **hidden_gobal_grad = NULL;        //隐藏层全局梯度
    double **input_gobal_grad = NULL;            //输出层全局梯度
    double **hidden_grad = NULL;                //隐藏层权值的梯度
    double **input_grad = NULL;                //输入层权值的梯度
    double time;
    int k = 0;
    int userNum = 0;

    // 创建存储用户名字的map
    map_t *map_user = create_map(char*,int);
    if (map_user == NULL) {
        printf("Failed to create map\n");
        exit(1);
    }

    // map初始化
    map_init(map_user);

    //创建映射需要指定两个函数，hashCode函数和equal函数。
//    MyHashMap * map_user = createMyHashMap(myHashCodeString, myEqualString);

//    //插入数据
//    for (int i=0; i<S; i++)
//    {
//        myHashMapPutData(map, strs[i], &data[i]);
//    }
//
//    //输出大小
//    printf("size=%d\n",myHashMapGetSize(map));


    //IMAGELIST *test1=NULL;
    //IMAGELIST *test2=NULL;
    // int savedelta;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Comm_size(MPI_COMM_WORLD, &n_p);

    // 输入训练次数
    if (id == 0) {
        printf("please input the times of train:\n");
        scanf("%d", &epochs);
    }

    //选择最优网络
    MPI_Bcast(&epochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //阻塞进程，使所有进程同步
    MPI_Barrier(MPI_COMM_WORLD);

    time = -MPI_Wtime();

    //各进程读取训练集和测试集
    if (trainname[0] != '\0'&&testname[0]!='\0'){
        imgl_load_images_from_textfile_id_user(trainlist, trainname, id, n_p, map_user);
        imgl_load_images_from_textfile_id_user(testlist, testname, id, n_p, map_user);
	}
    userNum = map_size(map_user);
    if(userNum == 0)
    {
        printf("The training set has no user data!\n");
        exit(-1);
    }

    // 初始化训练种子
    bpnn_initialize(SEED);

    // 选择阶段的并行训练
    backprop_face_parallel(trainlist, id, &net, sume, map_user);

    //MPI_Barrier(MPI_COMM_WORLD):

    // 获取最佳网络所在的进程k
    if (id == 0)
        k = selectBestNet(sume, id, net, n_p);

    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // k进程发送最佳网络
    if (id == k) {
        //printNet(net,id);
        sendNet(net, k);
    }

    // 主进程接收最佳网络
    if (id == 0)
        recvNet(net, k);

    // 广播网络到所有进程
    Bcast_Net(net, 0);
    //printNet(net,id);

    // 最佳网络的并行训练
    bpnn_train_parallel(net, &input_grad, &hidden_grad, epochs, LEARNRATE, IMPULSE, &input_gobal_grad, &hidden_gobal_grad, trainlist,
                        testlist, id, map_user);
    //printf("the addr of net: %o,id=%d\n",net,id);

    time += MPI_Wtime();
    MPI_Finalize();

    if (id == 0)
        printf("\ntime=%f,k=%d\n", time, k);
    //printNet(net,id);


}

/***
    训练开始
 ***/
void bpnn_train_parallel(BPNN *net, double ***input_grad, double ***hidden_grad, int epochs, double learning_rate,
                         double momentum,
                         double ***input_gobal_grad, double ***hidden_gobal_grad, IMAGELIST *trainlist, IMAGELIST *testlist, int id, map_t *map_user) {
    int i, j, k;
    double out_err, hid_err, sumerr;
    char netname[256]="BPNNnet";//bpnn网络名称

    printf("\n--------begin train parallel-------\n");
    printf("\n---epochs=%d\n", epochs);

    *input_gobal_grad = alloc_2d_dbl(net->input_n + 1, net->hidden_n + 1);
    *input_grad = alloc_2d_dbl(net->input_n + 1, net->hidden_n + 1);
    *hidden_gobal_grad = alloc_2d_dbl(net->hidden_n + 1, net->output_n + 1);
    *hidden_grad = alloc_2d_dbl(net->hidden_n + 1, net->output_n + 1);

    printf("\ntrain_n=%d,id=%d\n", trainlist->n, id);

    for (i = 1; i <= epochs; i++) {

        for (j = 0; j < trainlist->n; j++) {

            load_input_with_image(trainlist->list[i], net);
            load_target(trainlist->list[i], net, map_user);
            //printf("\n------1--------\n");

            bpnn_feedforward(net);

            //if(j==0)
            //printNet(net,id);

            bpnn_output_error(net->output_delta, net->target, net->output_units, net->output_n, &out_err);
            bpnn_hidden_error(net->hidden_delta, net->hidden_n, net->output_delta, net->output_n, net->hidden_weights,
                              net->hidden_units, &hid_err);
            grad_calculate(net->output_delta, net->output_n, net->hidden_units, net->hidden_n, *hidden_grad, j);
            grad_calculate(net->hidden_delta, net->hidden_n, net->input_units, net->input_n, *input_grad, j);
    		//printNet(net,id);
            //if(j==0)
            //{print_2d(*hidden_grad,net->hidden_n+1,net->output_n+1);}
            //printf("\n------2------\n");
        }

        reduce_main(*input_grad, *input_gobal_grad, net->input_n + 1, net->hidden_n + 1);
        reduce_main(*hidden_grad, *hidden_gobal_grad, net->hidden_n + 1, net->output_n + 1);
        //if(id==0)
        //{print_2d(*hidden_gobal_grad,net->hidden_n+1,net->output_n+1);}

        printf("\n------2------\n");

        Bcast_2d(*input_gobal_grad, net->input_n + 1, net->hidden_n + 1, 0);
        Bcast_2d(*hidden_gobal_grad, net->hidden_n + 1, net->output_n + 1, 0);
        //printf("\n-------id=%d------\n",id);
        //print_2d(*hidden_gobal_grad,net->input_n+1,net->hidden_n+1);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("\n------3------\n");

        bpnn_adjust_weights_parallel(*input_gobal_grad, net->input_n, net->hidden_n, net->input_weights,
                                     net->input_prev_weights, learning_rate, momentum);
        printf("\n--------4------\n");

        bpnn_adjust_weights_parallel(*hidden_gobal_grad, net->hidden_n, net->output_n, net->hidden_weights,
                                     net->hidden_prev_weights, learning_rate, momentum);
        printf("\n--------end train parallel   1------\n");

        //测试网络
	    performance_on_imagelist(net, testlist, 0, map_user);

        //保存网络
	    bpnn_save(net, netname);
    }
    printf("\n--------end train parallel------\n");

    /************** 预测结果 ****************************/

    // 输出测试集中每张图片的匹配情况
    printf("Matching at the end of the iteration: \n\n");
    printf("Test set 1 : \n\n");
    result_on_imagelist(net, testlist, 0, map_user);

    /** Save the trained network **/
    if (epochs > 0) {
        bpnn_save(net, netname);
    }
}

/***
    调正权值
***/
void bpnn_adjust_weights_parallel(double **grad, int rows, int cols, double **w, double **oldw, double learning_rate,
                                  double momentum) {
    printf("\n------adjust weight-----\n");
    double new_dw;
    int k, j;

    for (j = 1; j <= cols; j++) // 遍历输出层单元 / 隐藏层单元
    {
        for (k = 0; k <= rows; k++)  // 遍历隐藏层单元 / 输入层单元
        {
            // 新的权值增量
            new_dw = ((learning_rate * grad[k][j]) + (momentum * oldw[k][j]));
            //printf("\n------adjust weight 1-----\n");
            w[k][j] += new_dw;
            oldw[k][j] = new_dw;  // 保存权值增量，用于下次迭代时权值的更新(与冲量相乘加到权值增量中)
        }
    }
//printf("\n------adjust done-----\n");
}

/***
    规约到主进程
***/
void reduce_main(double **a, double **b, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            MPI_Reduce(*(a + i) + j, *(b + i) + j, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    printf("\n------reduce done------\n");
}


/***
    计算梯度
***/

void grad_calculate(double *delta, int ndelta, double *ly, int nly, double **grad, int n) {
    int k, j;

    ly[0] = 1.0;

    for (j = 1; j <= ndelta; j++) // 遍历输出层单元 / 隐藏层单元
    {
        for (k = 0; k <= nly; k++)  // 遍历隐藏层单元 / 输入层单元
        {
            // 新的权值增量
            if (n == 0)
                grad[k][j] = delta[j] * ly[k];
            else
                grad[k][j] += delta[j] * ly[k];
            //if(n==0)
            // printf("%f   ",grad[k][j]);
        }
    }
}


/***
    广播网络
***/
void Bcast_Net(BPNN *net, int id) {
    MPI_Bcast(&net->output_n, 1, MPI_INT, id, MPI_COMM_WORLD);
    MPI_Bcast(&net->hidden_n, 1, MPI_INT, id, MPI_COMM_WORLD);
    MPI_Bcast(&net->output_n, 1, MPI_INT, id, MPI_COMM_WORLD);
    //MPI_Bcast(net->input_units,net->input_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->output_units,net->output_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->output_delta,net->output_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //MPI_Bcast(net->target,net->output_n+1,MPI_DOUBLE,id,MPI_COMM_WORLD);
    //广播二维数组
    Bcast_2d(net->input_weights, net->input_n + 1, net->hidden_n + 1, id);
    Bcast_2d(net->hidden_weights, net->hidden_n + 1, net->output_n + 1, id);
    //Bcast_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id);
    //Bcast_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id);
}

/***
    广播二维数组
***/
void Bcast_2d(double **arry, int rows, int cols, int id) {
    int i;
    for (i = 0; i < rows; i++)
        MPI_Bcast(*(arry + i), cols, MPI_DOUBLE, id, MPI_COMM_WORLD);
}


/***
	打印网络
***/

void printNet(BPNN *net, int id) {

    printf("\n---------------------------------------------\n");
    printf("id =%d\n", id);

    printf("print net\n");

    printf("input_n:%d,hidden_n:%d,output_n:%d\n", net->input_n, net->hidden_n, net->output_n);

    //printf("input_uints\n");

    //print_1d(net->input_units,net->input_n);

    printf("\nhidden_units\n");

    print_1d(net->hidden_units, net->hidden_n);

    printf("\noutput_units\n");

    print_1d(net->output_units, net->output_n);

    //printf("\ninput_weights\n");

    //print_2d(net->input_weights,net->input_n+1,net->hidden_n+1);

    printf("\nhidden_weights\n");

    print_2d(net->hidden_weights, net->hidden_n + 1, net->output_n + 1);

    printf("\n---------------------------------------------\n");

}


/***
    打印一位数组
***/

void print_1d(double *a, int n) {

    int i;
    for (i = 0; i < n; i++)
        printf("%f ", a[i]);
}


/***
    打印二维数组
***/
void print_2d(double **a, int rows, int cols) {

    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            printf("%f ", a[i][j]);
        printf("\n");
    }
}

/***
	选择神经网络
***/
int selectBestNet(double sume[], int id, BPNN *net, int n_p) {

    int i;

    int k = 0;                                              //最优神经网络进程id号

    double min_sumerr;

    MPI_Status status;                      //MPI_Recv的参数

    for (i = 1; i < n_p; i++)

        MPI_Recv(&sume[i], 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

    printf("id=%d\n", id);
    print_1d(sume, n_p);

    min_sumerr = sume[0];

    for (i = 0; i < n_p; i++)
        if (sume[i] < min_sumerr) {
            min_sumerr = sume[i];
            k = i;
        }

    return k;
}


/***
        发送网络
***/

void sendNet(BPNN *net, int id) {
    if (id == 0)
        return;
    printf("\nsendNet--------\n");
    MPI_Send(&net->input_n, 1, MPI_INT, 0, id, MPI_COMM_WORLD);

    MPI_Send(&net->hidden_n, 1, MPI_INT, 0, id, MPI_COMM_WORLD);

    MPI_Send(&net->output_n, 1, MPI_INT, 0, id, MPI_COMM_WORLD);

    // MPI_Send(net->input_units,net->input_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    //MPI_Send(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    //MPI_Send(net->output_units,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    // MPI_Send(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    // MPI_Send(net->output_delta,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

    //MPI_Send(net->target,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);
    printf("\nsend_2d----\n");

    //发送二维数组

    send_2d(net->input_weights, net->input_n + 1, net->hidden_n + 1, id, 0);

    send_2d(net->hidden_weights, net->hidden_n + 1, net->output_n + 1, id, 0);

    //send_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id,0);

    //send_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id,0);
    printf("\nsendNet done--------\n");

}


/***
       发送二维数组
***/
void send_2d(double **arry, int rows, int cols, int where, int desnation) {
    printf("\nsende_2d--\n");
    int i;

    for (i = 0; i < rows; i++)

        MPI_Send(*(arry + i), cols, MPI_DOUBLE, desnation, where, MPI_COMM_WORLD);

}


/***

	接收网络
***/
void recvNet(BPNN *net, int id) {
    if (id == 0)
        return;
    printf("\nrecv Net--------\n");
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
    printf("\nrecv_2d---------\n");

    //接收二维数组

    recv_2d(net->input_weights, net->input_n + 1, net->hidden_n + 1, id);

    recv_2d(net->hidden_weights, net->hidden_n + 1, net->output_n + 1, id);

    //recv_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id);

    //recv_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id);
    printf("\nrecv Net done------\n");
}


/***
        接收二维数组
***/

void recv_2d(double **arry, int rows, int cols, int id) {
    printf("\nrecv_2d--\n");
    int i;

    MPI_Status status;

    for (i = 0; i < rows; i++) {

        MPI_Recv(*(arry + i), cols, MPI_DOUBLE, id, id, MPI_COMM_WORLD, &status);
    }

}

/***

 加载图片集

 参数：
	il	        图片集
	filename	文件名
	id	        进程号
    n           进程个数

***/
void imgl_load_images_from_textfile_id(IMAGELIST *il, char *filename, int id, int n) {

    IMAGE *iimage;
    FILE *fp;

    int i = 0;
    char buf[2000];
    int id_temp;

    id_temp = id;

    if (filename[0] == '\0') {
        printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Invalid file '%s'\n", filename);
    }

    if ((fp = fopen(filename, "r")) == NULL) {
        printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Couldn't open '%s'\n", filename);
    }

    while (fgets(buf, 1999, fp) != NULL) {

        if (i == id_temp) {

            imgl_munge_name(buf);
            printf("Loading '%s'...    id =%d\n", buf, id);
            fflush(stdout);

            if ((iimage = img_open(buf)) == 0) {
                printf("Couldn't open '%s'\n", buf);
            } else {
                imgl_add(il, iimage);
                printf("done\n");
            }

            fflush(stdout);
            id_temp = id_temp + n;
        }

        i++;
    }

    fclose(fp);
}

/***

 加载图片集

 参数：
	il	        图片集
	filename	文件名
	id	        进程号
    n           进程个数

***/
void imgl_load_images_from_textfile_id_user(IMAGELIST *il, char *filename, int id, int n, map_t *map_user) {

    IMAGE *iimage;
    FILE *fp;

    int i = 0;
    char buf[2000], userid[40];
    int id_temp;
    int i_userNum = 0;
    size_t mapSize = 0;

    id_temp = id;

    if (filename[0] == '\0') {
        printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Invalid file '%s'\n", filename);
    }

    if ((fp = fopen(filename, "r")) == NULL) {
        printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Couldn't open '%s'\n", filename);
    }

//    MyHashMapEntryIterator * it = createMyHashMapEntryIterator(map_user);

//    map_iterator_t iterator;

    while (fgets(buf, 1999, fp) != NULL) {

        if (i == id_temp) {

            imgl_munge_name(buf);
            printf("Loading '%s'...    id =%d\n", buf, id);
            fflush(stdout);

            // 获取每个用户的名字
            sscanf(buf, "%*[^/]/%*[^/]/%[^/]", userid);

            // 获取map中元素个数
            mapSize = map_size(map_user);

            // 插入
            *(int *)map_at(map_user,userid) = i_userNum+1;

            if(mapSize < map_size(map_user))
            {
                // 插入成功
                i_userNum++;
            }

//            myHashMapPutData(map_user, userid, i_userNum + 1);

            if ((iimage = img_open(buf)) == 0) {
                printf("Couldn't open '%s'\n", buf);
            } else {
                imgl_add(il, iimage);
                printf("done\n");
            }

            fflush(stdout);
            id_temp = id_temp + n;
        }

        i++;
    }

    fclose(fp);
}

/***

 参数：
	trainlist     	训练集
	id		  		进程号
	net		  		BPNN网络
	sume[]	  		存储误差的数组

 ***/
void backprop_face_parallel(IMAGELIST *trainlist, int id, BPNN **net, double sume[], map_t *map_user) {

    int i, j;
    int imagesize;
    int train_n;
    double out_err, hid_err, sumerr;

    int userNum = map_size(map_user);

    IMAGE *iimage;
    train_n = trainlist->n;

    printf("train_n:%d,id=%d\n", train_n, id);

    iimage = trainlist->list[0];

    imagesize = ROWS(iimage) * COLS(iimage);

    *net = bpnn_create(imagesize, userNum, userNum);

    //printNet(*net,id);
    for (i = 1; i <= SELETE; i++) {
        sumerr = 0;
        for (j = 0; j < train_n; j++) {

            load_input_with_image(trainlist->list[i], *net);
            load_target(trainlist->list[i], *net, map_user);
            bpnn_train(*net, LEARNRATE, IMPULSE, &out_err, &hid_err);
            sumerr += (out_err + hid_err);
        }
        // printf("sumerr: %g ,id=:%d\n", sumerr,id);
    }
    //printNet(*net,id);
    if (id == 0) {
        sume[0] = sumerr;
    } else {
        MPI_Send(&sumerr, 1, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
    }
}


int evaluate_performance(BPNN *net, double *err)
{
  bool flag = true; // 样例匹配成功为true
  
  *err = 0.0;
  double delta;
  
  // 计算输出层均方误差之和
  for (int j = 1; j <= net->output_n; j++) 
  {
    delta = net->target[j] - net->output_units[j];
    *err += (0.5 * delta * delta);
  }
  
  
  for (int j = 1; j <= net->output_n; j++) {
    /*** If the target unit is on... ***/
    if (net->target[j] > 0.5) {
      if (net->output_units[j] > 0.5) {
        /*** If the output unit is on, then we correctly recognized me! ***/
      } else /*** otherwise, we didn't think it was me... ***/
      {
        flag = false;
      }
    } else /*** Else, the target unit is off... ***/
    {
      if (net->output_units[j] > 0.5) {
        /*** If the output unit is on, then we mistakenly thought it was me ***/
        flag = false;
      } else {
        /*** else, we correctly realized that it wasn't me ***/
      }
    }
  }

  if (flag)
    return 1;
  else
    return 0;
}

/*** Computes the performance of a net on the images in the imagelist. ***/
/*** Prints out the percentage correct on the image set, and the
     average error between the target and the output units for the set. ***/
int performance_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user)
{
  double err, val;
  int i, n, j, correct;

  err = 0.0;
  correct = 0;
  n = il->n;  // n：图片集中图片张数
  if (n > 0) {
    // 遍历图片列表中每张图片
    for (i = 0; i < n; i++) {

      /*** Load the image into the input layer. **/
      load_input_with_image(il->list[i], net);

      /*** Run the net on this input. **/
      bpnn_feedforward(net);

      /*** Set up the target vector for this image. **/
      load_target(il->list[i], net, map_user);

      /*** See if it got it right. ***/
      if (evaluate_performance(net, &val)) {
        //匹配成功，计数器加1
        correct++;
      } 
      else if (list_errors) 
      {
        printf("%s", NAME(il->list[i]));

        // for (j = 1; j <= net->output_n; j++) 
        // {
        //   printf("%.3f ", net->output_units[j]);
        // }
        printf("\n");
      }
      err += val; // 列表中所有图片 输出层 均方误差之和
    }

    err = err / (double)n;  // 列表中所有图片 输出层 均方误差之和 的平均数

    if (!list_errors)
      /* bthom==================================
         this line prints part of the ouput line
         discussed in section 3.1.2 of homework
          */
      // 输出 匹配准确率 和 误差
      printf("%g%%  %g \n", ((double)correct / (double)n) * 100.0, err);
  } else {
    if (!list_errors)
      printf("0.0 0.0 ");
  }
  return 0;
}

// 评估图片集的匹配情况
void result_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user)
{
    double err, val;
    int i, n, j, correct;

    err = 0.0;
    correct = 0;

    n = il->n; // 图片集元素个数

    if (n > 0) {
        for (i = 0; i < n; i++) {
            /*** Load the image into the input layer. **/
            // 装载图片到输入层
            load_input_with_image(il->list[i], net);

            /*** Run the net on this input. **/
            // 在此输入的基础上运行这个网络
            bpnn_feedforward(net);

            /*** Set up the target vector for this image. **/
            // 设置目标向量
            load_target(il->list[i], net, map_user);

            // 输出图片的名称
            printf("Picture name: %s\n", NAME(il->list[i]));

            int map_userNum = map_size(map_user), i_flag_num = 0, i_flag_i = 0;

            // map迭代器
            map_iterator_t iterator;

            for(int i = 1; i  <= map_userNum; ++i)
            {
                printf("--output_units-->> %f\n", net->output_units[i]);
                if(net->output_units[i] > 0.5)
                {
                    // 统计输出权值大于0.5的输出单元个数和索引
                    i_flag_num ++;
                    i_flag_i = i;
                }
            }

            if(1 == i_flag_num)
            {
                // 遍历map
                for (iterator = map_begin(map_user); !iterator_equal(iterator, map_end(map_user)); iterator = iterator_next(iterator)) {

                    if(i_flag_i == *(int *) pair_second((const pair_t *) iterator_get_pointer(iterator)))
                    {
                        printf("He is --> %s \n", (char *) pair_first((const pair_t *) iterator_get_pointer(iterator)));
                    }
                }
            }
            else
            {
                printf("I do not know who he is...\n");
            }

            /*** See if it got it right. ***/
            if (evaluate_performance(net, &val)) {
                correct++;
                printf("Yes\n");
            } else {
                printf("No\n");
            }

            printf("\n");

            err += val;
        }

        err = err / (double)n;

        // 输出 匹配准确率 和 平均误差
        if (!list_errors)
            printf("Accuracy rate of: %g%%  Average error: %g \n\n",
                   ((double)correct / (double)n) * 100.0, err);
    } else {
        if (!list_errors)
            printf("0.0 0.0 ");
    }
    return;
}