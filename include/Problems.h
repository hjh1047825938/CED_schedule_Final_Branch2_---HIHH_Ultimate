#ifndef _PROBLEMS_H
#define _PROBLEMS_H

#include <cmath>
#include <ctime>
#include <random>
#include <vector>
#include <map>
#include <iostream>
#include <limits>
#include <numeric>
#include <algorithm>
using namespace std;

// Forward declaration
struct Workspace;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define QN 23 //DBm

double randnorm(double miu, double score);
typedef struct
{
    double Computation;
    double Communication;
    vector<int> Precedence;
    vector<int> Interact;
    vector<int> Start_Pre;
    vector<int> End_Pre;
    int Job_Constraints;
    vector<int> AvailEdgeServerList;
}CETask;

double CED_Schedule(const double* var, Workspace& ws, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time, double** EtoD_Distance, double** DtoD_Distance, vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices, vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad, vector<int>* DeviceLoad, vector<int>* CETask_coDevice, map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST, double* CE_ET);

#endif // _PROBLEMS_H
