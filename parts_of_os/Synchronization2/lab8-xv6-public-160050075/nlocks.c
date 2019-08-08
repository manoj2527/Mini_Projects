#include "types.h"
#include "stat.h"
#include "user.h"

void mutex_lock(int index)
{
	/*
		Todo
	*/
	acquire_mutex_spinlock(index);
	while(get_mutex_value(index)==1){
		cond_wait(index,index);
	}
	set_mutex_value(index,1);
	release_mutex_spinlock(index);
}

void mutex_unlock(int index)
{
	/*
		Todo
	*/
	acquire_mutex_spinlock(index);
	set_mutex_value(index,0);
	cond_signal(index);
	release_mutex_spinlock(index);
}


int main()
{
	int ret;


	init_counters();

	for(int i=0;i<10;i++){
		ret = fork();
		if(ret == 0){
			for(int j=0;j<1000;j++){
				mutex_lock(i);
				set_var(i, get_var(i)+1);
				mutex_unlock(i);
			}
			exit();
		}
	}

	for(int i=0;i<1000;i++){
		for(int j=0;j<10;j++){
			mutex_lock(j);
			set_var(j, get_var(j)+1);
			mutex_unlock(j);	
		}
	}

	for(int i=0;i<10;i++){
		wait();
	}

	for(int i=0;i<10;i++){
		int val = get_var(i);
		printf(1, "data at array index %d is %d\n", i,val);
	}
	exit();
}

