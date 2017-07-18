OBJ		= main.o backprop.o imagenet.o pgmimage.o
BPNN:$(OBJ)
	mpicc -o BPNN  $(OBJ) -std=c99 -lcstl -lstdc++ -fopenmp -lm
main.o:main.c
	mpicc -c main.c -std=c99 -lcstl -lstdc++ -fopenmp -lm -o main.o
backprop.o:backprop.c
	mpicc -c backprop.c -std=c99 -lcstl -lstdc++ -fopenmp -lm -o backprop.o
imagenet.o:imagenet.c
	mpicc -c imagenet.c -std=c99 -lcstl -lstdc++ -fopenmp -lm -o imagenet.o
pgmimage.o:pgmimage.c
	mpicc -c pgmimage.c -std=c99 -lcstl -lstdc++ -fopenmp -lm -o pgmimage.o
clean:
	rm -rf *.o
