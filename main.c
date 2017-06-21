#include"mpi.h"

#include<stdio.h>

#include"backprop.h"

#include"pgmimage.h"

#include<stdlib.h>

#include<string.h>

#define SELETE   10       //ѡ��׶���ѵ���Ĵ���

void backprop_face_parallel(IMAGELIST *trainlist,int id,BPNN **net,double sume[]);

int selectBestNet(double sume[],int id,BPNN *net,int n_p);

void send_2d(double **arry,int rows,int cols,int where,int desnation);

void sendNet(BPNN*net,int id);

void recvNet(BPNN *net,int id);

void recv_2d(double **arry,int rows,int cols,int id);

void printNet(BPNN *net,int id);

void print_1d(double *a,int n);

void print_2d(double **a,int rows,int cols);

void imgl_load_images_from_textfile_id(IMAGELIST *il, char *filename,int id,int n_p);

void Bcast_2d(double **arry,int rows,int cols,int id);

void Bcast_Net(BPNN *net,int id);

void bpnn_train_parallel(BPNN *net,double ***input_grad,double ***hidden_grad,int epochs,double learning_rate,double momentum,
						double ***input_gobal_grad,double ***hidden_gobal_grad,IMAGELIST *trainlist,int id);

void bpnn_adjust_weights_parallel(double **grad,int rows,int cols,double **w,double **oldw,double learning_rate,double momentum);

void reduce_main(double **a,double **b,int rows,int cols);

void grad_calculate(double *delta,int ndelta,double *ly,int nly,double **grad,int n);



/**
        ��������ģʽѵ��

**/





int main(int argc,char * argv[]){

	int id;                                                                                //����id

	int n_p;                                                                              //���̸���

	BPNN * net;

	int epochs;                                                                          //ѵ������

	int epoch;                                                                           //ѵ�����ڼ���

	char trainname[256]="all_train.list";                                               //�洢ͼƬ·�����ļ���

	IMAGELIST *il=imgl_alloc();                                                        //ѵ����

	//IMAGELIST *test1=NULL;

	//IMAGELIST *test2=NULL;

      // int savedelta;

	double  sume[1000];                                                               //�洢���������

	double **hidden_gobal_grad=NULL;								    //���ز�ȫ���ݶ�

	double **input_gobal_grad=NULL;								    //�����ȫ���ݶ�    

	double **hidden_grad=NULL;									    //���ز�Ȩֵ���ݶ�

	double **input_grad=NULL;									     //�����Ȩֵ���ݶ�

	double time;

	int k=0;

	MPI_Init(&argc,&argv);

      MPI_Comm_rank(MPI_COMM_WORLD,&id);

      MPI_Comm_size(MPI_COMM_WORLD,&n_p);

      if(id==0){
	      printf("please input the times of train:\n");
		scanf("%d",&epochs);
	}
         //ѡ����������

	MPI_Barrier(MPI_COMM_WORLD);									//�������̣�ʹ���н���ͬ��

	time=-MPI_Wtime();

	bpnn_initialize(201600608+id*1000);															

      imgl_load_images_from_textfile_id(il, trainname,id,n_p);                         //�����̶�ȡѵ����
	
      backprop_face_parallel(il,id,&net,sume);
	
	//MPI_Barrier(MPI_COMM_WORLD):
	if(id==0)
	 	k=selectBestNet(sume,id,net,n_p);
	
	MPI_Bcast(&k,1,MPI_INT,0,MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	
	if(id==k)
	{
	//printNet(net,id);
	sendNet(net,k);
	}

	if(id==0)
	   recvNet(net,k);
	Bcast_Net(net,0);
	//printNet(net,id);
	bpnn_train_parallel(net,&input_grad,&hidden_grad,epochs,0.3,0.3,&input_gobal_grad,&hidden_gobal_grad,il,id);
	//printf("the addr of net: %o,id=%d\n",net,id);
	
	time+=MPI_Wtime();
	MPI_Finalize();

	if(id==0)
	printf("\ntime=%f,k=%d\n",time,k);
	//printNet(net,id);
	

}
/***      ѵ����ʼ	***/
void bpnn_train_parallel(BPNN *net,double ***input_grad,double ***hidden_grad,int epochs,double learning_rate,double momentum,
						double ***input_gobal_grad,double ***hidden_gobal_grad,IMAGELIST *trainlist,int id){
	int i,j,k;
	double out_err, hid_err, sumerr;
	printf("\n--------begin train parallel-------\n");
	printf("\n---epochs=%d\n",epochs);
	*input_gobal_grad=alloc_2d_dbl(net->input_n+1,net->hidden_n+1);
	*input_grad=alloc_2d_dbl(net->input_n+1,net->hidden_n+1);
	*hidden_gobal_grad=alloc_2d_dbl(net->hidden_n+1,net->output_n+1);
	*hidden_grad=alloc_2d_dbl(net->hidden_n+1,net->output_n+1);
	printf("\ntrain_n=%d,id=%d\n",trainlist->n,id);
	for(i=1;i<=epochs;i++){
	for(j=0;j<trainlist->n;j++){
		load_input_with_image(trainlist->list[i], net);
		load_target(trainlist->list[i],net);
		//printf("\n------1--------\n");
		bpnn_feedforward(net);
		//if(j==0)
		//printNet(net,id);
		bpnn_output_error(net->output_delta, net->target, net->output_units, net->output_n, &out_err);
 		bpnn_hidden_error(net->hidden_delta, net->hidden_n, net->output_delta, net->output_n, net->hidden_weights, net->hidden_units, &hid_err);
		grad_calculate(net->output_delta,net->output_n,net->hidden_units,net->hidden_n,*hidden_grad,j);
		grad_calculate(net->hidden_delta,net->hidden_n,net->input_units,net->input_n,*input_grad,j);	
		//if(j==0)
		//{print_2d(*hidden_grad,net->hidden_n+1,net->output_n+1);}
		//printf("\n------2------\n");	
	}	
	reduce_main(*input_grad,*input_gobal_grad,net->input_n+1,net->hidden_n+1);
	reduce_main(*hidden_grad,*hidden_gobal_grad,net->hidden_n+1,net->output_n+1);
	//if(id==0)
	//{print_2d(*hidden_gobal_grad,net->hidden_n+1,net->output_n+1);}
	printf("\n------2------\n");
	Bcast_2d(*input_gobal_grad,net->input_n+1,net->hidden_n+1,0);
	Bcast_2d(*hidden_gobal_grad,net->hidden_n+1,net->output_n+1,0);
	//printf("\n-------id=%d------\n",id);
	//print_2d(*hidden_gobal_grad,net->input_n+1,net->hidden_n+1);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("\n------3------\n");
	bpnn_adjust_weights_parallel(*input_gobal_grad,net->input_n,net->hidden_n,net->input_weights, net->input_prev_weights, learning_rate, 		momentum);
	printf("\n--------4------\n");
	bpnn_adjust_weights_parallel(*hidden_gobal_grad,net->hidden_n,net->output_n,net->hidden_weights, net->hidden_prev_weights, learning_rate, 		momentum);
	printf("\n--------end train parallel   1------\n");
	}
	printf("\n--------end train parallel------\n"); 	
}
/***     ����Ȩֵ   ***/
void bpnn_adjust_weights_parallel(double **grad,int rows,int cols,double **w,double **oldw,double learning_rate,double momentum)
{
  printf("\n------adjust weight-----\n");
  double new_dw;
  int k, j;
  
  for (j = 1; j <=cols; j++) // ��������㵥Ԫ / ���ز㵥Ԫ
  {
    for (k = 0; k <=rows; k++)  // �������ز㵥Ԫ / ����㵥Ԫ
    {
      // �µ�Ȩֵ����
      new_dw = ((learning_rate *  grad[k][j]) + (momentum * oldw[k][j]));
      //printf("\n------adjust weight 1-----\n");
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;  // ����Ȩֵ�����������´ε���ʱȨֵ�ĸ���(�������˼ӵ�Ȩֵ������)
    }
  }
//printf("\n------adjust done-----\n");
}

/***     ��Լ��������		***/
void reduce_main(double **a,double **b,int rows,int cols){
	int i,j;
	for(i=0;i<rows;i++)
		for(j=0;j<cols;j++)
		MPI_Reduce(*(a+i)+j,*(b+i)+j,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	printf("\n------reduce done------\n");
}



/***	�����ݶ�	***/

void grad_calculate(double *delta,int ndelta,double *ly,int nly,double **grad,int n)
{
  int k, j;

  ly[0] = 1.0;
  
  for (j = 1; j <= ndelta; j++) // ��������㵥Ԫ / ���ز㵥Ԫ
  {
    for (k = 0; k <= nly; k++)  // �������ز㵥Ԫ / ����㵥Ԫ
    {
      // �µ�Ȩֵ����
	if(n==0)
      grad[k][j]=delta[j] * ly[k];
	else
	grad[k][j]+=delta[j]*ly[k];
       //if(n==0)
       // printf("%f   ",grad[k][j]);
    }
  }	
}


/***	�㲥����
		  ***/
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
	//�㲥��ά����
	Bcast_2d(net->input_weights,net->input_n+1,net->hidden_n+1,id);
	Bcast_2d(net->hidden_weights,net->hidden_n+1,net->output_n+1,id);
	//Bcast_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id);
	//Bcast_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id);
}
		
/***	�㲥��ά����		
			***/
void Bcast_2d(double **arry,int rows,int cols,int id){
	int i;
	for(i=0;i<rows;i++)
	MPI_Bcast(*(arry+i),cols,MPI_DOUBLE,id,MPI_COMM_WORLD);
}



/***��ӡ����

	***/

void printNet(BPNN *net,int id){

	printf("\n---------------------------------------------\n");
	printf("id =%d\n",id);

	printf("print net\n");

	printf("input_n:%d,hidden_n:%d,output_n:%d\n",net->input_n,net->hidden_n,net->output_n);

	//printf("input_uints\n");

	//print_1d(net->input_units,net->input_n);

	printf("\nhidden_units\n");

	print_1d(net->hidden_units,net->hidden_n);

	printf("\noutput_units\n");

	print_1d(net->output_units,net->output_n);

	//printf("\ninput_weights\n");

	//print_2d(net->input_weights,net->input_n+1,net->hidden_n+1);

	printf("\nhidden_weights\n");

	print_2d(net->hidden_weights,net->hidden_n+1,net->output_n+1);

	printf("\n---------------------------------------------\n");

}



/***	��ӡһλ����***/

void print_1d(double *a,int n){

	int i;

	for(i=0;i<n;i++)

		printf("%f ",a[i]);}



/***	��ӡ��ά���� ***/

void print_2d(double **a,int rows,int cols){

	int i,j;

	for(i=0;i<rows;i++){

		for(j=0;j<cols;j++)

			printf("%f ",a[i][j]);

	    	printf("\n");

	}

}

/***

	ѡ��������

			***/

int  selectBestNet(double sume[],int id,BPNN *net,int n_p){

      int i;

      int k=0;                                              //�������������id��

      double  min_sumerr;

      MPI_Status status;                      //MPI_Recv�Ĳ���


      for(i=1;i<n_p;i++)

           MPI_Recv(&sume[i],1,MPI_DOUBLE,i,i,MPI_COMM_WORLD,&status);

	printf("id=%d\n",id);
	print_1d(sume,n_p);

      min_sumerr=sume[0];

      for(i=0;i<n_p;i++)
         if(sume[i]<min_sumerr)
	      { 
		min_sumerr=sume[i];
		k=i;}

		    return k;

}





/***

        ��������

                        ***/

void sendNet(BPNN*net,int id){
	if(id==0)
	   return;
	  printf("\nsendNet--------\n");
        MPI_Send(&net->input_n,1,MPI_INT,0,id,MPI_COMM_WORLD);

        MPI_Send(&net->hidden_n,1,MPI_INT,0,id,MPI_COMM_WORLD);

        MPI_Send(&net->output_n,1,MPI_INT,0,id,MPI_COMM_WORLD);

       // MPI_Send(net->input_units,net->input_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

        //MPI_Send(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

       //MPI_Send(net->output_units,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

      // MPI_Send(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

      // MPI_Send(net->output_delta,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);

       //MPI_Send(net->target,net->output_n+1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);
	printf("\nsend_2d----\n");

        //���Ͷ�ά����

        send_2d(net->input_weights,net->input_n+1,net->hidden_n+1,id,0);

        send_2d(net->hidden_weights,net->hidden_n+1,net->output_n+1,id,0);

        //send_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id,0);

        //send_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id,0);
	  printf("\nsendNet done--------\n");

}





/***

       ���Ͷ�ά����

        ***/

void send_2d(double **arry,int rows,int cols,int where,int desnation){
	  printf("\nsende_2d--\n");
        int i;

        for(i=0;i<rows;i++)

        MPI_Send(*(arry+i),cols,MPI_DOUBLE,desnation,where,MPI_COMM_WORLD);

}





/***

	��������
		***/

void recvNet(BPNN *net,int id){
	if(id==0)
	  return;
	printf("\nrecv Net--------\n");
	MPI_Status status;

	MPI_Recv(&net->input_n,1,MPI_INT,id,id,MPI_COMM_WORLD,&status);

	MPI_Recv(&net->hidden_n,1,MPI_INT,id,id,MPI_COMM_WORLD,&status);

	MPI_Recv(&net->output_n,1,MPI_INT,id,id,MPI_COMM_WORLD,&status);

	//MPI_Recv(net->input_units,net->input_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

	//MPI_Recv(net->hidden_units,net->hidden_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

	//MPI_Recv(net->output_units,net->output_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

	//MPI_Recv(net->hidden_delta,net->hidden_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

	//MPI_Recv(net->output_delta,net->output_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);

	//MPI_Recv(net->target,net->output_n+1,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);
	printf("\nrecv_2d---------\n");

	//���ն�ά����

	recv_2d(net->input_weights,net->input_n+1,net->hidden_n+1,id);

	recv_2d(net->hidden_weights,net->hidden_n+1,net->output_n+1,id);

	//recv_2d(net->input_prev_weights,net->input_n+1,net->hidden_n+1,id);

	//recv_2d(net->hidden_prev_weights,net->hidden_n+1,net->output_n+1,id);
	 printf("\nrecv Net done------\n");
}





/***
        ���ն�ά����

			***/

void recv_2d(double **arry,int rows,int cols,int id){
	printf("\nrecv_2d--\n");
	int i;

	MPI_Status status;

	for(i=0;i<rows;i++){

	MPI_Recv(*(arry+i),cols,MPI_DOUBLE,id,id,MPI_COMM_WORLD,&status);}

}

/***
		����ͼƬ��
	������
	il	ͼƬ��
	filename	�ļ���
	id	���̺�	
      n     ���̸���                                                       ***/

void imgl_load_images_from_textfile_id(IMAGELIST *il, char *filename,int id,int n){

	IMAGE *iimage;

	FILE *fp;

	int i=0;

	char buf[2000];

	int id_temp;

	id_temp=id;


  	if (filename[0] == '\0') {

    	printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Invalid file '%s'\n", filename);

  	}



  	if ((fp = fopen(filename, "r")) == NULL) {

    	printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Couldn't open '%s'\n", filename);

  	}



  	while (fgets(buf, 1999, fp) != NULL) {

    	if(i==id_temp){

    	imgl_munge_name(buf);

    	printf("Loading '%s'...    id =%d\n", buf,id);  fflush(stdout);

    	if ((iimage = img_open(buf)) == 0)

    	{

      	printf("Couldn't open '%s'\n", buf);

    	}

    	else

    	{

      	imgl_add(il, iimage);

      	printf("done\n");

   	}

    	fflush(stdout);

    	id_temp=id_temp+n;

    	}

    	i++;

 	 }



  	fclose(fp);

}

/***   ������
	trainlist     ѵ����
	id		  ���̺�	
	net		  BPNN����
	sume[]	  �洢��������
									***/
void backprop_face_parallel(IMAGELIST *trainlist,int id,BPNN **net,double sume[])

{

        int i,j;

        IMAGE *iimage;

        int imagesize;

        int train_n;

        double out_err, hid_err, sumerr;

        train_n=trainlist->n;

        printf("train_n:%d,id=%d\n",train_n,id);

        iimage=trainlist->list[0];

        imagesize=ROWS(iimage)*COLS(iimage);

        *net=bpnn_create(imagesize, 20, 20);
	   //printNet(*net,id);
        for(i=1;i<=SELETE;i++){
		     sumerr=0;
                for (j = 0; j < train_n; j++) {



                        load_input_with_image(trainlist->list[i], *net);



                        load_target(trainlist->list[i], *net);


                        bpnn_train(*net, 0.3, 0.3, &out_err, &hid_err);



                        sumerr += (out_err + hid_err);
                }

               // printf("sumerr: %g ,id=:%d\n", sumerr,id);

        }
		//printNet(*net,id);
	if(id==0){
	    sume[0]=sumerr;
	}else{
             MPI_Send(&sumerr,1,MPI_DOUBLE,0,id,MPI_COMM_WORLD);
	}
}

