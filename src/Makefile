OBJ		= main.o backprop.o imagenet.o pgmimage.o evaluate.o parallelModule.o printNet.o train.o
BPNN:$(OBJ)
	mpicc -o BPNN  $(OBJ) -std=c99 -lcstl -fopenmp -lm
main.o:main.c
	mpicc -c main.c -std=c99 -fopenmp -o main.o
backprop.o:backprop.c
	mpicc -c backprop.c -std=c99 -lm -fopenmp -o backprop.o
imagenet.o:imagenet.c
	mpicc -c imagenet.c -std=c99 -lcstl -o imagenet.o
pgmimage.o:pgmimage.c
	mpicc -c pgmimage.c -std=c99 -lcstl -o pgmimage.o
evaluate.o:evaluate.c
	mpicc -c evaluate.c -std=c99 -lcstl -o evaluate.o
parallelModule.o:parallelModule.c
	mpicc -c parallelModule.c -std=c99 -fopenmp -o parallelModule.o
printNet.o:printNet.c
	mpicc -c printNet.c -std=c99 -o printNet.o
train.o:train.c
	mpicc -c train.c -std=c99 -lcstl -fopenmp -o train.o
clean:
	rm -rf *.o
