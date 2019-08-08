#include "types.h"
#include "x86.h"
#include "defs.h"
#include "date.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "proc.h"

int
sys_fork(void)
{
  return fork();
}

int
sys_exit(void)
{
  exit();
  return 0;  // not reached
}

int
sys_wait(void)
{
  return wait();
}

int
sys_kill(void)
{
  int pid;

  if(argint(0, &pid) < 0)
    return -1;
  return kill(pid);
}

int
sys_getpid(void)
{
  return myproc()->pid;
}

int
sys_sbrk(void)
{
  int addr;
  int n;

  if(argint(0, &n) < 0)
    return -1;
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

int
sys_sleep(void)
{
  int n;
  uint ticks0;

  if(argint(0, &n) < 0)
    return -1;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(myproc()->killed){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}



int sys_signal(){
  char* p;
  if(argptr(0,&p,32)<0)return -1;
  myproc()->handler = p;
  myproc()->default_handler = 1;
  return 0;
}

int sys_signal_send(){
  int p;
  int sig;
  if(argint(0, &p) < 0)
    return -1;
  if(argint(1, &sig) < 0)
    return -1;
  help_send(p,sig);
  return 0;
}

int sys_signal_ret(){
  *(myproc()->tf) = *(myproc()->temptf);
  return 0;
}


int sys_setprio(){
  int p;
  if(argint(0, &p) < 0)
    return -1;
  myproc()->prio = p;
  return 1;
}

int sys_getprio(){
  return myproc()->prio;
}

int sys_fork2(){
  int p;
  if(argint(0, &p) < 0)
    return -1;
  return help_fork(p);
}

// return how many clock tick interrupts have occurred
// since start.
int
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}
