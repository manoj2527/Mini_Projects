#include <pthread.h>
#include <stdio.h>

#define NTHREADS 10
pthread_mutex_t mutexsum[10];
long int counter[10];

//	The thread starter function
void *myThread(void *a)
{
	int co = (int)a;
	for(int i=0; i<1000; i++){
		pthread_mutex_lock (&mutexsum[co]);
		counter[co]++;
		pthread_mutex_unlock (&mutexsum[co]);
	}
}

int main()
{
	// Create space for pthread variables
	pthread_t tid[NTHREADS];
	for(int i=0;i<10;i++){
		pthread_mutex_init(&mutexsum[i], NULL);
		counter[i]=0;
	}
	for(int i=0; i<NTHREADS; i++)
	{
		//	Create a thread with default attributes and no arguments
		pthread_create(&tid[i], NULL, myThread, (void *)i);
	}

	for(int i=0;i<1000;i++){
		for(int j=0;j<10;j++){
			pthread_mutex_lock (&mutexsum[j]);
			counter[j]++;
			pthread_mutex_unlock (&mutexsum[j]);
		}
	}
	for(int i=0; i<NTHREADS; i++)
	{
		//	Wait for all threads to finish
		pthread_join(tid[i], NULL);
	}

	for(int i=0;i<10;i++){
		printf("Counter[%d]: %ld \n",i, counter[i]);
	}

	return 0;
}
