#ifndef _MULTIMETHOD_H
#define _MULTIMETHOD_H

#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "Population.h"
#include "Problems.h"
#include "Workspace.h"
using namespace std;

#define TNN 1000

typedef double (*FF) (const double* var, Workspace& ws, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time, double** EtoD_Distance, double** DtoD_Distance, vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevice, vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad, vector<int>* DeviceLoad, vector<int>* CETask_coDevice, map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST, double* CE_ET);

class MultiMet : public Population<double>
{
public:
	MultiMet(int psize, int nn, double lb, double ub, int c_num, int e_num, int d_num, int ce_tnum, int m_jnum, int m_optnum, FF evaluate, const std::filesystem::path& data_dir = ".", const std::string& data_file = "data_matrix_100.txt");
	~MultiMet();

public:
	FF EvaluFunc;
    std::filesystem::path DataDir;  // Data directory for loading files
    std::string DataFileName;       // Data file name
    double Pini;                    // Heuristic init probability
    Workspace workspace;             // Reusable workspace for fitness evaluation
    uint64_t eval_count;             // Total evaluation calls since last reset
    
    //Prob
    int Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum;
    CETask* CETask_Property;
    double* MTask_Time;
    double** EtoD_Distance;
    double** DtoD_Distance;
    vector<int>* AvailDeviceList;
    double* EnergyList;
    
    vector<int>* CloudDevices;
    vector<int>* EdgeDevices;
    vector<int>* CloudLoad;
    vector<int>* EdgeLoad;
    vector<int>* DeviceLoad;
    vector<int>* CETask_coDevice;
    map<int, double>* Edge_Device_comm;
    double** ST;
    double** ET;
    double* CE_ST;
    double* CE_ET;
    

	//PSO
	double** ibest;
	double* ibest_fit;
	double **velocity;
	double ac1, ac2;
	double *AC1, *AC2, *AW;
	int OArow;
	int **OA;

	//ACO                      //int tao_size;
	double **ant_tao;

	//ABCA
	int *trial;
	double *pr;

	//VNS
	int *neigh;

	//CMAES parameter
	bool CMAisdone;

	int lambda;
	int mu;
	double mucov;
	double mueff;
    double* weights;
	double damps;
	double cs;
    double ccumcov;
    double ccov;
	double* xstart;
	double* stddev;
	double diagonalCov;
	//CMAES 

	//! Step size.
	double sigma;
	//! Mean x vector, "parent".
	double* xmean;

	//! Sorting index of sample population.
	int* index;

	double chiN;
	//! Lower triangular matrix: i>=j for C[i][j].
	double** C;
	//! Matrix with normalize eigenvectors in columns.
	double** B;
	//! Axis lengths.
	double* rgD;
	//! Anisotropic evolution path (for covariance).
	double* pcc;
	//! Isotropic evolution path (for step length).
	double* ps;
	//! Last mean.
	double* xold;
	//! B*D*z.
	double* BDz;
	//! Temporary (random) vector used in different places.
	double* tempRandom;

	//PBILc
    double num_of_elite;
    double *PB_center;
    double *PB_sigma;
    
    //BatA
    double *BAT_r;
    double *BAT_A;
    double **BAT_v;

	//FA
	double *I;		//Intensity
	int	*Index;		//sort of fireflies according to fitness values
	double alpha;
    
    //BA
    int ne, nre, stlim;
    double ngh_decay, ngh_origin;
    int* ngh_decay_count;
    double* ngh;
    
    //meme number
    int *Ind_meme;
    double **SubDecBase;
    
    // Rotated-ring migration (Step 7)
    int nSubpop;           // Number of subpopulations (default 8)
    int Dispara;           // Neighbor distance parameter (starts at 1)
    int nCircle;           // Migration interval in generations (default 5)
    double pElitist;       // Probability of sending gbest vs random (default 0.8)
    double** subpop_gbest; // Best individual per subpopulation
    double* subpop_gbest_fit;
    bool migrationEnabled; // Whether migration is active

public:
	inline double randnorm(double miu, double score);
	void pop_update(int p_start, int p_end);
	void pop_better_update(int p_start, int p_end);
	void Initial();
    void Evaluation(bool s, int p_start, int p_end);             //s = 0: evalu pop, s = 1: evalu newpop
    double Eval(const double* var);
    void ResetEvalCount() { eval_count = 0; }
    uint64_t GetEvalCount() const { return eval_count; }
    void SetPini(double pini);
    
    //Prob
    void show_result(double* results);
	
	//GA
	void select(int p_start, int p_end);
	void crossover(double pc, int p_start, int p_end);
	void xover(int, int);
	void mutate(double pm, int p_start, int p_end);
	void GA(double pc, double pm, int p_start, int p_end);
	void NGA(double pc, double pm, double mdis, int p_start, int p_end);
	void LGA(double pc, double pm, double step, int nst, int p_start, int p_end);
	void VAGA(int p_start, int p_end);
	void EAGA(int p_start, int p_end);

	//PSO
	void PSO(double w, double c1, double c2, double max_ve, int p_start, int p_end);
	void CPSO(double w, double c1, double c2, double max_ve, int p_start, int p_end);
	void Subgradient(double *theta, int q, double c_step, double *subgrad);
	void SPSO(double w, double c1, double c2, double max_ve, int Gen, int MaxGen, int p_start, int p_end);
	void Cauchy_mutation(double* pp, int Gen, int MaxGen);
	void CMPSO(double w, double c1, double c2, double max_ve, int Gen, int MaxGen, int p_start, int p_end);
	void APSO_1(double c1, double c2, double max_ve, int Gen, int MaxGen, int p_start, int p_end);
	void APSO_2(double c1, double c2, double max_ve, int p_start, int p_end);
	void APSO_3(double max_ve, int p_start, int p_end);
	void APSO_4(double max_ve, int Gen, int MaxGen, int p_start, int p_end);
	void APSO_5(double max_ve, int Gen, int MaxGen, int p_start, int p_end);
	void CreateOA();
	void Orthogonal_P(double *P0, double w, double c, int ppn);
	void OLPSO(double w, double c, double max_ve, int p_start, int p_end);

	//HS
	void newpop_worst_best(int& w, int& b, int p_start, int p_end);
	void HS(double srate, double trate, double bw, int p_start, int p_end);

	//ACO
	void path_finding(double epsl, int p_start, int p_end);
	void phe_updating(int p_start, int p_end);
	void ACO(double epsl, int p_start, int p_end);
	
	//COA
	void chaos(double **incpop, int chaos_n);
	void COA(int chaos_n, int p_start, int p_end);

	//DE
	void differential_mutate(double F, int S, int p_start, int p_end);
	void differential_crossover(double cr, int p_start, int p_end);
	void DE(double F, int S, double cr, int p_start, int p_end);
	
	//GDE - Gbest-centric Differential Evolution
	void GDE(double Pmu, int n_centric, int p_start, int p_end);

	//CSA
	void levy_cuckoo(int p_start, int p_end);
	void nest_discover(double pa, int p_start, int p_end);
	void CSA(double pa, int p_start, int p_end);

	//ABCA
	void EmployedBee(int pn, double *tmp, double **pp);
	void OnlookerBee(int p_start, int p_end);
	void ScoutBee(int limit, int p_start, int p_end);
	void ABCA(int limit, int p_start, int p_end);

	//POA
	void plant_growth(int p_start, int p_end);
	void POA(int p_start, int p_end);

	//ILS
	void localsearch(double *pp, int nst);
	void ILS(int nst, int p_start, int p_end);

	//VNS
	void VNS(int nst, int p_start, int p_end);

	//GRASP
	void GRASP(double alfa, int nst, int p_start, int p_end);

	//PBILc
    void PBILC(int p_start, int p_end, double learn_rate);
    
    //BATA
    void BATA(int gen, int p_start, int p_end);
	
	//FA
	void FA(double gama, double alpha, double betamin,int gen, int p_start, int p_end);

	//sorting selection
	void pop_heap_adjust(int s, int len);
	void pop_heap_sort(int len);
	void newpop_heap_adjust(int s, int len);
	void newpop_heap_sort(int len);

	//niche
	void pop_niche(double mdis, int p_start, int p_end);
	//local
	void newpop_localstep(double step, int nst, int p_start, int p_end);
	//variance
	double pop_random_variance(int p_start, int p_end);
	//entropy
	double pop_random_entropy(int subn, int p_start, int p_end);

	//CMA-ES
	void CMAES(int gen, int p_start, int p_end);
	void CMAES_parametersetting();
	void CMAES_initial();
	void CMAES_sampleGenerate(int firstrun, int p_start, int p_end);
	void updateEigensystem();
	void eigen(double* diag, double** Q, double* rgtmp);
	void householder(double** V, double* d, double* e);
	void ql(double* d, double* e, double** V);
	double* CMAES_updateDistribution(int gen);
	void sortIndex(const double* rgFunVal, int* iindex, int n);
	void adaptC2(const int hsig, int gen);
    
    //BA
    void BA(int p_start, int p_end);
    void NeighborFlowerPatch(int nr, int point);
    
    //Discrete Gradient
    void Direct_Change(int ChangeSize);
    
    //local search operators
    void newpop_bit_climbing(int popi, int L, double scale);
    void newpop_simplex(int popi, int L, int stepn, double scale);
    void newpop_box_complex(int popi, int L, int stepN, double scale);
    void newpop_powell(int popi, int L, int stepn, double scale);
    
    
    //memetic adaptive selection
    void meme_selection(int popi, int X, double scale, int Iter);
    void meme_random_walk(double scale);
    void meme_simple_random(double scale);
    void meme_randperm(double scale);
    void meme_inheritance(double scale);
    void meme_subprob_decomposition(int Gen, int MaxG, int kk, double scale);
    void meme_biasd_roulette(int Gen, int MaxG, double scale);
    
    // Rotated-ring migration (Step 7)
    void InitMigration(int nG, int nC, double pE);  // Initialize migration parameters
    void RingMigration(int gen);                     // Perform migration at generation gen
    void UpdateSubpopBest();                         // Update subpopulation best individuals
    int GetSubpopWorst(int subpop_idx);             // Get index of worst individual in subpop
};
#endif
