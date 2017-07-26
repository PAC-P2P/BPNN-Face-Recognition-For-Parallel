//
// Created by xx on 17/7/23.
//

#include "train.h"
#include "printNet.h"
#include "parallelModule.h"
//
// int backprop_face(IMAGELIST *trainlist, IMAGELIST *test1list, int epochs, int savedelta,
//                   char *netname, int list_errors, map_t *map_user) {
//     IMAGE *iimg;
//     BPNN *net;
//     int train_n, epoch, i, imgsize;
//     double out_err, hid_err, sumerr;
//
//     int userNum = map_size(map_user);
//
//     train_n = trainlist->n;
//
//     /*** Read network in if it exists, otherwise make one from scratch ***/
//     if ((net = bpnn_read(netname)) == NULL) {
//         if (train_n > 0) {
//             printf("Creating new network '%s'\n", netname);
//             iimg = trainlist->list[0];
//             imgsize = ROWS(iimg) * COLS(iimg);
//             /* bthom ===========================
//               make a net with:
//                 imgsize inputs, peopleNum hiden units, and peopleNum output unit
//           【图片规模】的输入层单元个数，【训练集总人数】个隐藏层单元，【训练集总人数】个输出层单元
//                 */
//             net = bpnn_create(imgsize, userNum, userNum);
//         } else {
//             printf("Need some images to train on!\n");
//             return -1;
//         }
//     }
//
//     if (epochs > 0) {
//         /*** 训练进行中（epochs次） ***/
//         printf("Training underway (going to %d epochs)\n", epochs);
//         /*** 每epochs次保存网络 ***/
//         printf("Will save network every %d epochs\n", savedelta);
//         fflush(stdout);
//     }
//
//     /*** 迭代前输出测试表现 ***/
//     /*** Print out performance before any epochs have been completed. ***/
//     printf("\n迭代前：\n");
//     printf("训练集误差和：0.0\n");
//     printf("评估训练集的表现： ");
//     performance_on_imagelist(net, trainlist, 0, map_user);
//     printf("评估测试集1的表现：");
//     performance_on_imagelist(net, test1list, 0, map_user);
//     //printf("评估测试集2的表现：");
//     //performance_on_imagelist(net, test2list, 0, map_user);
//     printf("\n");
//     fflush(stdout);
//     if (list_errors) {
//         printf("\n训练集中的这些图片分类失败:\n");
//         performance_on_imagelist(net, trainlist, 1, map_user);
//         printf("\n测试集1中的这些图片分类失败:\n");
//         performance_on_imagelist(net, test1list, 1, map_user);
//         //printf("\n测试集2中的这些图片分类失败:\n");
//         //performance_on_imagelist(net, test2list, 1, map_user);
//     }
//
//     /************** 开始训练！ ****************************/
//     /************** Train it *****************************/
//     for (epoch = 1; epoch <= epochs; epoch++) {
//
//         // 输出迭代次数
//         printf("Iteration number: %d \n", epoch);
//         fflush(stdout);
//
//         sumerr = 0.0;
//         for (i = 0; i < train_n; i++) {
//
//             /** Set up input units on net with image i **/
//             // 用训练集中图片i来设置输入层单元
//             load_input_with_image(trainlist->list[i], net);
//
//             /** Set up target vector for image i **/
//             // 为图片i设置目标向量
//             load_target(trainlist->list[i], net, map_user);
//
//             /** Run backprop, learning rate 0.3, momentum 0.3 **/
//             /** 运行反向传播算法，学习速率0.3，冲量0.3 **/
//             bpnn_train(net, LEARNRATE, IMPULSE, &out_err, &hid_err);
//
//             sumerr += (out_err + hid_err);  // 训练集中所有图片作为输入，网络的 输出层 和 隐藏层 的误差之和
//         }
//         printf("训练集误差和: %g \n", sumerr);
//
//         // 评估测试集，测试集1，测试集2 的表现
//         /*** Evaluate performance on train, test, test2, and print perf ***/
//         printf("评估训练集的表现： ");
//         performance_on_imagelist(net, trainlist, 0, map_user);
//         printf("评估测试集1的表现：");
//         performance_on_imagelist(net, test1list, 0, map_user);
//         //printf("评估测试集2的表现：");
//         //performance_on_imagelist(net, test2list, 0, map_user);
//         printf("\n");
//         fflush(stdout);
//
//         /*** Save network every 'savedelta' epochs ***/
//         if (!(epoch % savedelta)) {
//             bpnn_save(net, netname);
//         }
//     }
//     printf("\n");
//     fflush(stdout);
//     /************** 迭代结束 ****************************/
//
//     /************** 预测结果 ****************************/
//
//     // 输出测试集中每张图片的匹配情况
//     printf("迭代结束后的匹配情况：\n\n");
//     printf("测试集1：\n\n");
//     result_on_imagelist(net, test1list, 0, map_user);
//     //printf("测试集2：\n\n");
//     //result_on_imagelist(net, test2list, 0, map_user);
//
//     /** Save the trained network **/
//     if (epochs > 0) {
//         bpnn_save(net, netname);
//     }
//     return 0;
// }

void backprop_face_choose(IMAGELIST *trainlist, int train_n, int id, BPNN *net, int epochs, int iterateTimes, int savedelta, double *totalCorrect, map_t *map_user)
{

    int i, epoch, size = (iterateTimes-1)*epochs;
    double out_err, hid_err, sumerr;

    // printf("[%d] Number of training sets: %d\n",id, train_n);

    // printf("Creating new network '%s'\n", netname);



    // if (epochs > 0) {
    //     /*** 训练进行中（epochs次） ***/
    //     printf("Training underway (going to %d epochs)\n", epochs);
    //     /*** 每epochs次保存网络 ***/
    //     printf("Will save network every %d epochs\n", savedelta);
    //     fflush(stdout);
    // }

    /*** 迭代前输出测试表现 ***/
    /*** Print out performance before any epochs have been completed. ***/
    // printf("\n迭代前：\n");
    // printf("训练集误差和：0.0\n");
    // printf("评估训练集的表现： ");
    // performance_on_imagelist(net, trainlist, 0, map_user);
    // printf("评估测试集1的表现：");
    // performance_on_imagelist(net, testlist, 0, map_user);
    //printf("评估测试集2的表现：");
    //performance_on_imagelist(net, test2list, 0, map_user);
    // printf("\n");
    // fflush(stdout);
    // if (list_errors) {
    // printf("\n训练集中的这些图片分类失败:\n");
    // performance_on_imagelist(net, trainlist, 1, map_user);
    // printf("\n测试集1中的这些图片分类失败:\n");
    // performance_on_imagelist(net, testlist, 1, map_user);
    //printf("\n测试集2中的这些图片分类失败:\n");
    //performance_on_imagelist(net, test2list, 1, map_user);
    // }

    /************** 开始训练！ ****************************/
    /************** Train it *****************************/

    for(epoch = 1;epoch <= epochs; epoch++){
        sumerr = 0;
        for (i = 0; i < train_n; i++) {
            /** Set up input units on net with image i **/
            // 用训练集中图片i来设置输入层单元
            load_input_with_image(trainlist->list[i], net);
            /** Set up target vector for image i **/
            // 为图片i设置目标向量
            load_target(trainlist->list[i], net, map_user);
            /** Run backprop, learning rate LEARNRATE, momentum IMPULSE **/
            /** 运行反向传播算法，学习速率LEARNRATE，冲量IMPULSE **/
            bpnn_train(net, LEARNRATE, IMPULSE, &out_err, &hid_err);
            // 训练集中所有图片作为输入，网络的 输出层 和 隐藏层 的误差之和
            sumerr += (out_err + hid_err);
        }

        // 每个进程的总误差
        // *totalSumerr += sumerr;

        // 输出迭代次数
        printf("Iteration number: %d \n", epoch+size);
        fflush(stdout);

        printf("[%d] 训练集误差和: %g \n", id, sumerr);

        // 评估测试集，测试集1，测试集2 的表现
        /*** Evaluate performance on train, test, test2, and print perf ***/

        printf("[%d] 评估训练集的表现： ", id);
        performance_on_imagelist(net, trainlist, 0, map_user, totalCorrect);

        // printf("评估测试集1的表现：");
        // performance_on_imagelist(net, testlist, 0, map_user);
        //printf("评估测试集2的表现：");
        //performance_on_imagelist(net, test2list, 0, map_user);
        printf("\n");
        fflush(stdout);

        /*** Save network every 'savedelta' epochs ***/
        // if (!(epoch % savedelta)) {
        //     bpnn_save(net, netname);
        // }
        // printf("sumerr: %g ,id=:%d\n", sumerr,id);

    }
    printf("\n");
    fflush(stdout);
    //printNet(net,id);

/************** 迭代结束 ****************************/

}

/***
    训练开始
 ***/
// void backprop_face_parallel(BPNN *net, double ***input_grad, double ***hidden_grad, int epochs, double learning_rate, double momentum,
//                          double ***input_gobal_grad, double ***hidden_gobal_grad, IMAGELIST *trainlist, IMAGELIST *testlist, int id, map_t *map_user)
// {
//     int i, k, epoch;
//     double out_err, hid_err, sumerr;
//     char netname[256] = NETNAME;//bpnn网络名称
//
//     printf("\n--------begin train parallel-------\n");
//     printf("\n---epochs=%d\n", epochs);
//
//     printf("评估训练集的表现： ");
//     performance_on_imagelist(net, trainlist, 0, map_user, NULL);
//     //测试网络
//     printf("评估测试集的表现：");
//     performance_on_imagelist(net, testlist, 0, map_user, NULL);
//
//     *input_gobal_grad = alloc_2d_dbl(net->input_n + 1, net->hidden_n + 1);
//     *input_grad = alloc_2d_dbl(net->input_n + 1, net->hidden_n + 1);
//     *hidden_gobal_grad = alloc_2d_dbl(net->hidden_n + 1, net->output_n + 1);
//     *hidden_grad = alloc_2d_dbl(net->hidden_n + 1, net->output_n + 1);
//
//     printf("\ntrain_n=%d,id=%d\n", trainlist->n, id);
//
//     for (epoch = 1; epoch <= epochs; epoch++) {
//
//         // 输出迭代次数
//         printf("Iteration number: %d \n", epoch);
//         fflush(stdout);
//
//         // printf("\n--------[printNet]-------\n");
//         // printNet(net,id);
//
//         sumerr = 0.0;
//         for (i = 0; i < trainlist->n; i++) {
//
//             load_input_with_image(trainlist->list[i], net);
//
//             load_target(trainlist->list[i], net, map_user);
//
//             bpnn_feedforward(net);
//
//             //if(j==0)
//             // printNet(net,id);
//
//             bpnn_output_error(net->output_delta, net->target, net->output_units, net->output_n, &out_err);
//             bpnn_hidden_error(net->hidden_delta, net->hidden_n, net->output_delta, net->output_n, net->hidden_weights,
//                               net->hidden_units, &hid_err);
//
//             grad_calculate(net->output_delta, net->output_n, net->hidden_units, net->hidden_n, *hidden_grad, i);
//             grad_calculate(net->hidden_delta, net->hidden_n, net->input_units, net->input_n, *input_grad, i);
//
//             //printNet(net,id);
//             //if(j==0)
//             //{print_2d(*hidden_grad,net->hidden_n+1,net->output_n+1);}
//             //printf("\n------2------\n");
//
//             // printf("\n--------[printNet]-------\n");
//             // printNet(net,id);
//
//             /*** Adjust input and hidden weights. ***/
//             // 调整权值
//             // bpnn_adjust_weights(net->output_delta, net->output_n, net->hidden_units, net->hidden_n, net->hidden_weights, net->hidden_prev_weights, learning_rate, momentum);
//             // bpnn_adjust_weights(net->hidden_delta, net->hidden_n, net->input_units, net->input_n, net->input_weights, net->input_prev_weights, learning_rate, momentum);
//
//
//             sumerr += (out_err + hid_err);
//
//             // if(i >= 20) break;
//         }
//
//         printf("训练集误差和: %g \n", sumerr);
//
//         reduce_main(*input_grad, *input_gobal_grad, net->input_n + 1, net->hidden_n + 1);
//         reduce_main(*hidden_grad, *hidden_gobal_grad, net->hidden_n + 1, net->output_n + 1);
//         // if(id==0)
//         // {print_2d(*hidden_gobal_grad,net->hidden_n+1,net->output_n+1);}
//
//         //printf("\n------2------\n");
//
//         Bcast_2d(*input_gobal_grad, net->input_n + 1, net->hidden_n + 1, 0);
//         Bcast_2d(*hidden_gobal_grad, net->hidden_n + 1, net->output_n + 1, 0);
//         //printf("\n-------id=%d------\n",id);
//         //print_2d(*hidden_gobal_grad,net->input_n+1,net->hidden_n+1);
//
//         MPI_Barrier(MPI_COMM_WORLD);
//         //printf("\n------3------\n");
//
//         bpnn_adjust_weights_parallel(*input_gobal_grad, net->input_n, net->hidden_n, net->input_weights,
//                                      net->input_prev_weights, learning_rate, momentum);
//         //printf("\n--------4------\n");
//         //
//         bpnn_adjust_weights_parallel(*hidden_gobal_grad, net->hidden_n, net->output_n, net->hidden_weights,
//                                      net->hidden_prev_weights, learning_rate, momentum);
//         // printf("\n--------end train parallel   1------\n");
//
//         printf("评估训练集的表现： ");
//         performance_on_imagelist(net, trainlist, 0, map_user, NULL);
//         //测试网络
//         // printf("评估测试集的表现：");
//         // performance_on_imagelist(net, testlist, 0, map_user);
//
//     }
//     printf("\n--------end train parallel------\n");
//
//     if(0 == id)
//     {
//         //保存网络
//         bpnn_save(net, netname);
//
//         /************** 预测结果 ****************************/
//
//         // 输出测试集中每张图片的匹配情况
//         printf("Matching at the end of the iteration: \n\n");
//         printf("Test set 1 : \n\n");
//         result_on_imagelist(net, testlist, 0, map_user);
//
//         /** Save the trained network **/
//         if (epochs > 0) {
//             bpnn_save(net, netname);
//         }
//     }
// }
