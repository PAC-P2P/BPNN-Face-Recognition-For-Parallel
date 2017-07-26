//
// Created by xx on 17/7/23.
//

#include "evaluate.h"


// 评估表现
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
int performance_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user, double *totalCorrect)
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
    if(NULL != totalCorrect)
    {
      *totalCorrect = (double)correct;
    }

    return 0;
}

// 评估图片集的匹配情况
void result_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors, map_t *map_user, double *correctRate, double *error)
{
    double val;
    int i, n, j, correct;

    *error = 0.0;
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

            /******************* 识别具体是哪个人. ******************/

            // size_t map_userNum = map_size(map_user), i_flag_num = 0, i_flag_i = 0;
            //
            // // map迭代器
            // map_iterator_t iterator;
            //
            // for(size_t i = 1; i  <= map_userNum; ++i)
            // {
            //     // printf("--output_units-->> %f\n", net->output_units[i]);
            //     if(net->output_units[i] > 0.5)
            //     {
            //         // 统计输出权值大于0.5的输出单元个数和索引
            //         i_flag_num ++;
            //         i_flag_i = i;
            //     }
            // }
            //
            // if(1 == i_flag_num)
            // {
            //     // 遍历map
            //     for (iterator = map_begin(map_user); !iterator_equal(iterator, map_end(map_user)); iterator = iterator_next(iterator)) {
            //
            //         if(i_flag_i == *(int *) pair_second((const pair_t *) iterator_get_pointer(iterator)))
            //         {
            //             printf("He is 【%s】 \n", (char *) pair_first((const pair_t *) iterator_get_pointer(iterator)));
            //         }
            //     }
            // }
            // else
            // {
            //     printf("I do not know who he is...\n");
            // }

            /******************* 评估表现 ******************/

            /*** See if it got it right. ***/
            if (evaluate_performance(net, &val)) {
                correct++;
                printf("Yes\n");
            } else {
                printf("No\n");
            }

            // printf("\n");

            *error += val;
        }

        *error = *error / (double)n;

        *correctRate = ((double)correct / (double)n) * 100.0;

        // 输出 匹配准确率 和 平均误差
        // if (!list_errors)
        //     printf("Accuracy rate of: %g%%  Average error: %g \n\n", *correctRate, *error);
    } else {
        if (!list_errors)
            printf("0.0 0.0 ");
    }
    return;
}
