#include<cuda.h>
#include<cuda_runtime.h>
#include<numa.h> //all numa calls

#include<unistd.h> //sysconf, usleep
#include<sched.h>

//#include<linux/getcpu.h>
#include<hwloc.h>
#include<omp.h>
#include<iostream>
#include<string>



int main (int argc, char *argv[]) {

   long  numCores = sysconf( _SC_NPROCESSORS_CONF );

   //unsigned int currCPU = 0;
   //unsigned int currNode = 0;

   if (numa_available() < 0) {
      std::cout << "This system does not support the NUMA API...exiting" << std::endl;
   }


   cpu_set_t CPUs;
   CPU_ZERO(&CPUs);
   sched_getaffinity(0, sizeof(CPUs), &CPUs);

   int currCore = sched_getcpu(); 
   std::cout << "Num CPU Cores:" << numCores << std::endl;
   std::cout << "sched_getcpu(): " << currCore << std::endl;

   int count = 0;
   for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(i, &CPUs))
         count++;
   }


   //int totalCPUs = 0;
   //hwloc_topology_t topo;

   currCore = sched_getcpu(); 
   std::cout << "Main Program thread: " << currCore << std::endl;

   std::cout << "numa_num_task_nodes: " << numa_num_task_nodes() << std::endl;
   std::cout << "numa_max_node:" << numa_max_node() << std::endl;
   std::cout << "numa_max_possible_node:" << numa_max_possible_node() << std::endl;
   std::cout << "numa_num_possible_nodes:" << numa_num_possible_nodes() << std::endl;
   
   std::cout << "numa_num_configured_cpus: " << numa_num_configured_cpus() << std::endl;
   std::cout << "numa_num_configured_nodes: " << numa_num_configured_nodes() << std::endl;
   std::cout << "numa_get_mems_allowed: " << numa_get_mems_allowed() << std::endl;
   std::cout << "numa_num_task_cpus: " << numa_num_task_cpus() << std::endl;
   std::cout << "numa_preferred: " << numa_preferred() << std::endl;
   
   std::cout << "numa_distance 0 => 1: " << numa_distance(0, 1) << std::endl;

   int OmpMaxThreads = 16;//omp_get_max_threads();
   omp_set_num_threads(OmpMaxThreads);
   std::cout << "Max Omp Threads: " << OmpMaxThreads << std::endl;

   for (int i = 0; i < numa_num_configured_cpus(); ++i) {
      std::cout << "Physical core : " << i << " NUMA node: " << numa_node_of_cpu(i) << std::endl;
   }

   count = 0;
   while(1) {

      //numa_run_on_node(count % 2);
      std::cout << "RUN ON NUMA NODE: " << (count % 2) << std::endl;
      #pragma omp parallel
      {

         cpu_set_t cpuset;
         CPU_ZERO(&cpuset);

         int thread_ID = omp_get_thread_num();
         int node = (count % 2) == 0 ? 0 : 1;
         
         if (thread_ID < 8) {
            CPU_SET(thread_ID + node * 8, &cpuset);
         } else {
            CPU_SET(thread_ID + node * 8 + 8, &cpuset);
         }
         pthread_t thread = pthread_self();
         int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
         
         if (s != 0)
            std::cout << "Problem in setaffinity function: ThreadID: " << omp_get_thread_num() << "Core:" << sched_getcpu() << std::endl;

         long long loop = 0;
         while(loop < 1000000000)
            loop++;

         #pragma omp critical
         {
            bitmask *numa_mask = numa_allocate_cpumask();
            numa_node_to_cpus(0, numa_mask);
            std::cout << "Numa CPU mask " << numa_all_nodes_ptr->maskp << std::endl;

            std::cout << "CPU_MEM_PREF/Sched_Core/OMP_Thread_Count/OMP_Thread_ID: ";
            std::cout << numa_preferred() << " " << sched_getcpu() << " " << omp_get_num_threads() << " " << omp_get_thread_num();
            std::cout << std::endl;
         }
      }
   }
   return 0;
}
