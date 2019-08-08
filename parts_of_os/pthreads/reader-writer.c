#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <time.h>



struct read_write_lock
{
	// If required, use this strucure to create
	// reader-writer lock related variables
	int writers;
	int readers;
	int wait;
	pthread_mutex_t mutex;
	pthread_cond_t read;
	pthread_cond_t write;

}rwlock;

long int data = 0;			//	Shared data variable


void InitalizeReadWriteLock(struct read_write_lock * rw)
{
	rw->writers = 0;
	rw->readers = 0;
	rw->wait = 0;
	pthread_mutex_init(&rw->mutex, NULL);
   	pthread_cond_init (&rw->read, NULL);
   	pthread_cond_init (&rw->write, NULL);
}


// The pthread based reader lock
void ReaderLock(struct read_write_lock * rw)
{
	pthread_mutex_lock (&rw->mutex);
	while(rw->writers>0 || rw->wait>0){
		pthread_cond_wait(&rw->read,&rw->mutex);
	}
	rw->readers++;
	pthread_mutex_unlock (&rw->mutex);
}	

// The pthread based reader unlock
void ReaderUnlock(struct read_write_lock * rw)
{
	pthread_mutex_lock (&rw->mutex);
	rw->readers--;
	if(rw->readers==0)pthread_cond_signal(&rw->write);
	pthread_mutex_unlock (&rw->mutex);
}

// The pthread based writer lock
void WriterLock(struct read_write_lock * rw)
{
	pthread_mutex_lock (&rw->mutex);
	rw->wait++;
	while(rw->writers>0||rw->readers>0){
		pthread_cond_wait(&rw->write,&rw->mutex);
	}
	rw->wait--;
	rw->writers++;
	pthread_mutex_unlock (&rw->mutex);	
}

// The pthread based writer unlock
void WriterUnlock(struct read_write_lock * rw)
{
	pthread_mutex_lock (&rw->mutex);
	rw->writers--;
	if(rw->writers==0){
		pthread_cond_signal(&rw->write);                                                                                                                                  
		if(rw->wait<=0)pthread_cond_broadcast(&rw->read);
	}
	pthread_mutex_unlock (&rw->mutex);
}

//	Call this function to delay the execution by 'delay' ms
void delay(int delay)
{
	struct timespec wait;

	if(delay <= 0)
		return;

	wait.tv_sec = delay / 1000;
	wait.tv_nsec = (delay % 1000) * 1000 * 1000;
	nanosleep(&wait, NULL);
}

// The pthread reader start function
void *ReaderFunction(void *args)
{
	//	Delay the execution by arrival time specified in the input
	int *p = (int *)args;
	delay(p[0]);
	//	....
	
	//  Get appropriate lock
	//	Display  thread ID and value of the shared data variable
	ReaderLock(&rwlock);
	printf("Reader %d, data: %ld\n", p[1],data);
	delay(1);
	free(args);
	ReaderUnlock(&rwlock);
	//
    //  Add a dummy delay of 1 ms before lock release  
	//	....
}

// The pthread writer start function
void *WriterFunction(void *args)
{
	//	Delay the execution by arrival time specified in the input
	int *p = (int *)args;
	delay(p[0]);
	//	....
	//
	//  Get appropriate lock
	//	Increment the shared data variable
	//	Display  thread ID and value of the shared data variable
	WriterLock(&rwlock);
	data++;
	printf("Writer %d, data: %ld\n",p[1], data);
	delay(1);
	free(args);
	WriterUnlock(&rwlock);
    //  Add a dummy delay of 1 ms before lock release  
	//	....
}

int main(int argc, char *argv[])
{
	pthread_t *threads;
	struct argument_t *arg;
	
	long int reader_number = 0;
	long int writer_number = 0;
	long int thread_number = 0;
	long int total_threads = 0;	
	
	int count = 0;			// Number of 3 tuples in the inputs.	Assume maximum 10 tuples 
	int rws[10];			
	int nthread[10];
	int delay[10];

	//	Verifying number of arguments
	if(argc<4 || (argc-1)%3!=0)
	{
		printf("reader-writer <r/w> <no-of-threads> <thread-arrival-delay in ms> ...\n");		
		printf("Any number of readers/writers can be added with repetitions of above mentioned 3 tuple \n");
		exit(1);
	}

	//	Reading inputs
	for(int i=0; i<(argc-1)/3; i++)
	{
		char rw[2];
		strcpy(rw, argv[(i*3)+1]);

		if(strcmp(rw, "r") == 0 || strcmp(rw, "w") == 0)
		{
			if(strcmp(rw, "r") == 0)
				rws[i] = 1;					// rws[i] = 1, for reader
			else
				rws[i] = 2;					// rws[i] = 2, for writer
			
			nthread[i] = atol(argv[(i*3)+2]);		
			delay[i] = atol(argv[(i*3)+3]);

			count ++;						//	Number of tuples
			total_threads += nthread[i];	//	Total number of threads
		}
		else
		{
			printf("reader-writer <r/w> <no-of-threads> <thread-arrival-delay in ms> ...\n");
			printf("Any number of readers/writers can be added with repetitions of above mentioned 3 tuple \n");
			exit(1);
		}
	}
	threads=(pthread_t *)malloc(total_threads * sizeof(pthread_t ));
	int tn = -1;
	InitalizeReadWriteLock(&rwlock);
	//	Create reader/writer threads based on the input and read and write.
	for(int i=0;i<count;i++){
		if(rws[i]==1){
			for(long int j=0;j<nthread[i];j++){
				tn++;
				reader_number++;
				int *temp = (int *)malloc(2*(sizeof(int)));
				temp[0] = delay[i];
				temp[1] = reader_number;
				pthread_create(&threads[tn], NULL, ReaderFunction, (void *)temp);
			}
		}
		else
		{
			for(long int j=0;j<nthread[i];j++){
				tn++;
				writer_number++;
				int *temp = (int *)malloc(2*(sizeof(int)));
				temp[0] = delay[i];
				temp[1] = writer_number;
				pthread_create(&threads[tn], NULL, WriterFunction, (void *)temp);
			}	
		}
	}
	for(int i=0; i<total_threads; i++)
	{
		//	Wait for all threads to finish
		pthread_join(threads[i], NULL);
	}
	//  Clean up threads on exit

	exit(0);
}
