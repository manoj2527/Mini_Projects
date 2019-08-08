#include <pthread.h>
#include <stdio.h>

#define NTHREADS 100
pthread_mutex_t mutexsum;
long int counter = 0;

//	The thread starter function
void *myThread()
{
	for(int i=0; i<1000; i++){
		pthread_mutex_lock (&mutexsum);
		counter++;
		pthread_mutex_unlock (&mutexsum);
	}
}

int main()
{
	// Create space for pthread variables
	pthread_t tid[NTHREADS];
	pthread_mutex_init(&mutexsum, NULL);
	for(int i=0; i<NTHREADS; i++)
	{
		//	Create a thread with default attributes and no arguments
		pthread_create(&tid[i], NULL, myThread, NULL);
	}

	for(int i=0; i<NTHREADS; i++)
	{
		//	Wait for all threads to finish
		pthread_join(tid[i], NULL);
	}

	printf("Counter: %ld \n", counter);

	return 0;
}
