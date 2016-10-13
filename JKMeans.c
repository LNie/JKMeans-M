#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>


int main(int argc,char *argv[])
{
    // Read parameters
	FILE *parafile;
	parafile = fopen(argv[1],"r");
    if(parafile==NULL){
        printf("Cannot open parameter file: %s.\n",argv[1]);
		return(1);
	}
	
	int fscanfRet;
	// Open log file
	char logFilename[500];
	fscanfRet = fscanf(parafile,"Log filename: %s\n",logFilename);
	if(fscanfRet!=1){
		printf("Cannot read log filename in the parameter file.\n");
		return(1);
	}
	FILE *logfile;
    logfile = fopen(logFilename, "w+");
    if(logfile==NULL){
        printf("Cannot open file log.txt.\n");
        fclose(logfile);
        return(1);
	}else{
	    printf("Start.\n");
	    fprintf(logfile,"Start.\n");
	    fflush(logfile);
	}
	
	// Read output filename
	char outputFilename[500];
	fscanfRet = fscanf(parafile,"Ouput filename: %s\n",outputFilename);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read the output filemane in the parameter file.\n");
		return(1);
	}
	
	long maxIter;
	fscanfRet = fscanf(parafile,"Maximal iteration: %ld\n",&maxIter);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read maximal iteration in the parameter file.\n");
		return(1);
	}
	fprintf(logfile,"Maximal iteration: %ld\n",maxIter);
	
	double lambda;
	fscanfRet = fscanf(parafile,"Lambda: %lf\n",&lambda);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read lambda in the parameter file.\n");
		return(1);
	}
	fprintf(logfile,"Lambda: %lf\n",lambda);
	
	long numSamples;
	fscanfRet = fscanf(parafile,"Number of samples: %ld\n",&numSamples);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read the number of samples in the parameter file.\n");
		return(1);
	}
	fprintf(logfile,"Number of samples: %ld\n",numSamples);
	
	long numFeatures;
	fscanfRet = fscanf(parafile,"Number of features: %ld\n",&numFeatures);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read the number of features in the parameter file.\n");
		return(1);
	}
	fprintf(logfile,"Number of features: %ld\n",numFeatures);
	
	long numClusters;
	fscanfRet = fscanf(parafile,"Number of clusters: %ld\n",&numClusters);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read the number of clusters in the parameter file.\n");
		return(1);
	}
	fprintf(logfile,"Number of clusters: %ld\n",numClusters);
	
	long numTasks;
	fscanfRet = fscanf(parafile,"Number of tasks: %ld\n",&numTasks);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read the number of tasks in the parameter file.\n");
		return(1);
	}
	fprintf(logfile,"Number of tasks: %ld\n",numTasks);
	fflush(logfile);
	

    // Variables
	double esp = 1.0e-10;
    long i,j;
    long task;
    int CFlag = 1;
    long iter = 0;
    time_t rawtime;
    struct tm * timeinfo;
	long tmpValue;
    double cost;
    double preCost=INFINITY;
    double *dataSet[numTasks];
    long *initLabel;
    long *label;
    double *costSet[numTasks];
	long *labelSet[numTasks];
	long *preLabelSet[numTasks];
	long *nextLabelSet[numTasks];
	double *prior;
	double initPrior = 1.0/numClusters;
	
	// Loading initial labels
	char initFilename[500];
	fscanfRet = fscanf(parafile,"Initial filename: %s\n",initFilename);
	if(fscanfRet!=1){
		fprintf(logfile,"Cannot read the initial filemane in the parameter file.\n");
		return(1);
	}
	initLabel = (long*)malloc(sizeof(long)*numSamples);
    if(loadInit(initLabel,initFilename,numSamples,logfile)==1){
        return(1);
    }


    // Loading data
	char *fileSet[numTasks];
	for(task=0;task<numTasks;task++){
		fileSet[task] = (char*)malloc(sizeof(char)*500);
		tmpValue = -1;
		fscanfRet = fscanf(parafile,"%ld: %s\n",&tmpValue,fileSet[task]);
		if(fscanfRet!=2||tmpValue!=task){
			printf("%d %ld %ld",fscanfRet,tmpValue,task);
			fprintf(logfile,"Cannot read the data filemane in the parameter file.\n");
			return(1);
		}
	}
    #pragma omp parallel for
	for(task=0;task<numTasks;task++){
    	dataSet[task] = (double*)malloc(sizeof(double)*numSamples*numFeatures);
    	if(loadData(dataSet[task],fileSet[task],numSamples,numFeatures,logfile)==1){
            CFlag = 0;
    	}
	}
	if(CFlag==0){
        return(1);
	}

    // Initialization
    for(task=0;task<numTasks;task++){
        labelSet[task] = (long*)malloc(sizeof(long)*numSamples);
        memcpy(labelSet[task],initLabel,sizeof(long)*numSamples);
        preLabelSet[task] = (long*)malloc(sizeof(long)*numSamples*numFeatures);
    	nextLabelSet[task] = (long*)malloc(sizeof(long)*numSamples*numFeatures);
    	costSet[task] = (double*)malloc(sizeof(double));
    }
    free(initLabel);
	prior = (double*)malloc(sizeof(double)*numSamples*numClusters);

    // Print start time
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    fprintf (logfile,"Current local time and date: %s", asctime(timeinfo));
    fflush(logfile);

    // Main loop
    while(CFlag){
		
        // Calculate prior
        iter++;
        for(i=0;i<numSamples;i++){
            for(j=0;j<numClusters;j++){
                prior[numClusters*i+j] = initPrior;
            }
        }
        if(iter>1){
            for(task=0;task<numTasks;task++){
                label = labelSet[task];
                for(i=0;i<numSamples;i++){
                    prior[numClusters*i+label[i]-1] += 1;
                }
            }
        }
        for(i=0;i<numSamples;i++){
            for(j=0;j<numClusters;j++){
                prior[numClusters*i+j] = log(prior[numClusters*i+j]);
            }
        }

        // Update
        #pragma omp parallel for
        for(task=0;task<numTasks;task++){
            kker(dataSet[task],costSet[task],labelSet[task],nextLabelSet[task],prior,lambda,numSamples,numFeatures,numClusters,iter);
        }

        // Calculate the total cost
        cost = 0;
        for(task=0;task<numTasks;task++){
            cost += *costSet[task];
        }

        // Check termination
        time (&rawtime);
        timeinfo = localtime (&rawtime);
		fprintf(logfile,"Iteration: %ld; Cost: %f; Time: %s \n",iter,cost,asctime(timeinfo));
        fflush(logfile);
        if(cost>preCost){
            for(task=0;task<numTasks;task++){
                free(labelSet[task]);
                labelSet[task] = preLabelSet[task];
            }
            fprintf(logfile,"Stop and rollback.\n");
            CFlag = 0;
        }else if(preCost-cost<esp){
            fprintf(logfile,"Stop at a local minimum.\n");
            CFlag = 0;
        }else if(iter>maxIter){
            CFlag = 0;
            fprintf(logfile,"Stop at the maximal iter.\n");
        }else{
            preCost = cost;
            for(task=0;task<numTasks;task++){
                label = preLabelSet[task];
                preLabelSet[task] = labelSet[task];
                labelSet[task] = nextLabelSet[task];
                nextLabelSet[task] = label;
            }
        }

    }

    //Save results
    if(saveResults(labelSet,outputFilename,numTasks,numSamples,logfile)==1){
        fclose(logfile);
        return(1);
    }else{
        fclose(logfile);
        return(0);
    }
}

int loadInit(long *initLabel,char *initFilename,long numSamples,FILE *logfile){
    long i;
    int fscanfRet;
    FILE *fp;

    fprintf(logfile,"%s\n",initFilename);
    fp = fopen(initFilename,"r");
    if(fp==NULL){
        fprintf(logfile,"Cannot open: %s.\n",initFilename);
        return(1);
    }else{
        for(i=0;i<numSamples;i++){
            fscanfRet = fscanf(fp,"%ld",&initLabel[i]);
            if(fscanfRet!=1){
                fprintf(logfile,"The data in the initial file is not correct.\n");
                return(1);
            }
        }
        fclose(fp);
    }
    return(0);
}

int loadData(double *data,char *filename,long numSamples,long numFeatures,FILE *logfile){
    long i,j;
    int fscanfRet;
    FILE *fp;

    fprintf(logfile,"%s\n",filename);
    fp = fopen(filename, "r");
    if(fp==NULL){
        fprintf(logfile,"Cannot open: %s.\n",filename);
        return(1);
    }else{
        for(i=0;i<numSamples;i++){
            for(j=0;j<numFeatures;j++){
                fscanfRet = fscanf(fp, "%lf", &data[numFeatures*i+j]);
                if(fscanfRet!=1){
                    fprintf(logfile,"The data in %s is not correct.\n",filename);
                    return(1);
                }
            }
        }
        fclose(fp);
    }
    return(0);
}

int saveResults(long *labelSet[],char *outputFilename,long numTasks,long numSamples,FILE *logfile){
    long i;
    long task;
    int fscanfRet;
    long *label;
    FILE *fp;

    fp = fopen (outputFilename, "w+");
    if(fp==NULL){
        fprintf(logfile,"Cannot open: %s.\n",outputFilename);
        return(1);
    }else{
        for(task=0;task<numTasks;task++){
            label = labelSet[task];
            for(i=0;i<numSamples;i++){
                fprintf(fp, "%ld\t", label[i]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    return(0);
}

int kker(double *data,double *cost,long *label,long *nextLabel,double *prior,double lambda,long numSamples,long numFeatures,long numClusters,long iter){

    long i,j,k;
	double tmpValue;
	long tmpIndex;
    double *centroids;
	double *dists;
	long *memCounter;

    // Initialization
    centroids = (double*)malloc(sizeof(double)*numClusters*numFeatures);
	memCounter = (long*)malloc(sizeof(long)*numClusters);
	dists =  (double*)malloc(sizeof(double)*numSamples*numClusters);
	memset(centroids,0,sizeof(double)*numClusters*numFeatures);
	memset(memCounter,0,sizeof(long)*numClusters);

	// Update centroids
	for(i=0;i<numSamples;i++){
		if(label[i]>0){
			memCounter[label[i]-1] += 1;
			for(j=0;j<numFeatures;j++){
                centroids[numFeatures*(label[i]-1)+j] += data[numFeatures*i+j];
			}
		}
	}
	for(i=0;i<numClusters;i++){
		for(j=0;j<numFeatures;j++){
            centroids[numFeatures*i+j] /= memCounter[i];
		}
	}

	// Update distances
	for(i=0;i<numSamples;i++){
		for(j=0;j<numClusters;j++){
            dists[numClusters*i+j] = -lambda*prior[numClusters*i+j];
			for(k=0;k<numFeatures;k++){
			    tmpValue = data[numFeatures*i+k]-centroids[numFeatures*j+k];
                dists[numClusters*i+j] += tmpValue*tmpValue;
			}
		}
	}

	// Calculate cost
	if(iter>1){
        *cost = 0;
		for(i=0;i<numSamples;i++){
            *cost += dists[numClusters*i+label[i]-1];
		}
	}
	else{
		*cost = INFINITY;
	}

	// Update labels
	for(i=0;i<numSamples;i++){
		tmpIndex = 0;
		tmpValue = dists[numClusters*i];
		for(j=1;j<numClusters;j++){
			if(tmpValue>dists[numClusters*i+j]){
				tmpValue = dists[numClusters*i+j];
				tmpIndex = j;
			}
		}
		nextLabel[i] = tmpIndex+1;
	}

	free(centroids);
	free(dists);
	free(memCounter);
	return(0);
}




