#include "types.h"
#include "stat.h"
#include "user.h"
char b[1024];


int getpusz()
{
	/* todo */
	uint ps = 4096;
	uint pa;
	int flags;
	int count = 0;
	uint x  = (0x80000000)/(4096);
	for(int i=0;i<x;i++){
		if(get_va_to_pa(i*ps, &pa, &flags))count++;
	}
	return count*ps;
}
int getpksz()
{
	/* todo */
	int p = 0x80000000;
	int end = 0xFFFFFFFF; 
	int ps = 4096;
	uint pa;
	int flags;
	int count = 0;
	int x  = (end-p)/ps;
	for(int i=0;i<=x;i++){
		if(get_va_to_pa(p+i*ps, &pa, &flags))count++;
	}
	return count*ps;
}

int
main(int argc, char *argv[])
{
	char *buf;

	printf(1, "\ngetpsz: %d bytes \n", getpsz());
	printf(1,"getpusz: %d bytes \n",getpusz());
	printf(1,"getpksz: %d bytes\n",getpksz());


	buf=sbrk(4096);
	buf[0]='\0';
	printf(1, "\ngetpsz: %d bytes \n", getpsz());
	printf(1,"getpusz: %d bytes \n",getpusz());
	printf(1,"getpksz: %d bytes\n",getpksz());

	
	buf=sbrk(4096*7);
	printf(1, "\ngetpsz: %d bytes \n", getpsz());
	printf(1,"getpusz: %d bytes \n",getpusz());
	printf(1,"getpksz: %d bytes\n",getpksz());

	buf=sbrk(1);
	printf(1, "\ngetpsz: %d bytes \n", getpsz());
	printf(1,"getpusz: %d bytes \n",getpusz());
	printf(1,"getpksz: %d bytes\n",getpksz());

	buf=sbrk(2);
	printf(1, "\ngetpsz: %d bytes \n", getpsz());
	printf(1,"getpusz: %d bytes \n",getpusz());
	printf(1,"getpksz: %d bytes\n",getpksz());


	exit();
}