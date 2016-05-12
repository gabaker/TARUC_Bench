all:
	gcc -std=gnu99 -fopenmp -D INNER -O3 test_bw.c -o inner
	gcc -std=gnu99 -fopenmp -O3 test_bw.c -o outer

clean:
	rm inner outer
