
/**********************************
 *
 * Header Files(test)
 *
 *********************************/

#include <math.h>			// for sin()
#include <stdlib.h>			// for rand()
#include <time.h>			// for clock()
#include <stdio.h>			// for printf()
#include <float.h>			// for DBL_EPSILON

/**********************************
 *
 * Macros
 *
 *********************************/

#define SQR(x)					( (x)*(x) )
#define CUB(x)					( (x)*(x)*(x) )
#define SMALLER(x,y)			( (x)<(y) ? (x) : (y) )	// some C compilers support MIN(,)
#define MAX(x,y)				( (x)>(y) ? (x) : (y) )
#define RAND_IN(low,high)		( (low) + (double)((high) - (low))*random()/RAND_MAX )
#define PIN(min,x,max)			( (x)<(min) ? (min) : ((x)>(max)?(max):(x)) )

/**********************************
 *
 * Constants
 *
 *********************************/

#define PI 3.14159265358979323846

// DIM = 2 * N_PEN (non-autonomous), DIM = 2 * N_PEN + 1 (autonomous)
enum {N_PEN = 8, SUB_PEN = N_PEN - 1, DIM = 2 * N_PEN, N_EQN = DIM + SQR(DIM)};

enum {homo=1, dRandom, alternate, linear, sineWave, cosineWave, oddManOut, oddMenOut, altQuad, altLin};
enum {euler=1, RuKu4=4, RuKu5=5, RuKu45=45};

/**********************************
 *
 * Global Types
 *
 *********************************/

typedef enum {false=0, true} boolean;

/**********************************
 *
 * Global Variables
 *
 *********************************/

// RK5Lyap //

double x[N_EQN], temp_x[N_EQN];
double dx_dt[N_EQN];

double tDiverge;						// full lyap parameters
int nDiverge;

double xError[N_EQN];
double xScale[N_EQN];

// RK5 //

double ak2[N_EQN];
double ak3[N_EQN];
double ak4[N_EQN];
double ak5[N_EQN];
double ak6[N_EQN];
double yTemp[N_EQN];

double dtTry, dtMin, dtMax, error, errorNow;  // Runge-Kutta parameters

double  // Cash-Karp parameters
a2 = 0.2,
a3 = 0.3,
a4 = 0.6,
a5 = 1.0,
a6 = 0.875,

b21 = 0.2,
b31 = 3.0/40.0,
b32 = 9.0/40.0,
b41 = 0.3,
b42 = -0.9,
b43 = 1.2,
b51 = -11.0/54.0,
b52 = 2.5,
b53 = -70.0/27.0,
b54 = 35.0/27.0,
b61 = 1631.0/55296.0,
b62 = 175.0/512.0,
b63 = 575.0/13824.0,
b64 = 44275.0/110592.0,
b65 = 253.0/4096.0,

c1 = 37.0/378.0,
c3 = 250.0/621.0,
c4 = 125.0/594.0,
c6 = 512.0/1771.0,

dc1 = 37.0/378.0 - 2825.0/27648.0,
dc3 = 250.0/621.0 - 18575.0/48384.0,
dc4 = 125.0/594.0 - 13525.0/55296.0,
dc6 = 512.0/1771.0 - 0.25,

dc5 = -277.0/14336.0;

// --- //

double config_vars_lf[100];				// for ReadConfigFile
long config_vars_ld[100];
char config_vars_s[100];

// changed from MAX_P to N_PEN
double angle[N_PEN];					// pendulum variables
	double relAngle[N_PEN];
double angVel[N_PEN];
double deltaAngle[N_PEN];				// for lyapunov calculation
double deltaAngVel[N_PEN];
double length[N_PEN];
	// double lengthConcrete[] = {0.6757200531046353,-0.7362197049750343,0.6199930620301959,-0.5834526338254545,-0.4763369923222063,-0.827581408219945,0.1254343137817411,-0.4967499137654523,-0.167046764140333,0.4418886366469826,-0.5354069691292523,0.5382548357283888,-0.1133576285576094,-0.4640693655094912,0.995284596244046,0.6912669517395205,0.891491475120218,0.1970533767512513,-0.3733194536416622,-0.5429567976763218,0.2018900353508102,-0.7014172579606328,-0.3462041671682707,0.464965212914051,-0.1778951844773588,0.680875546332852,-0.3083559312469257,0.2274615277780828,0.07848940684281994,-0.6210328934820081,-0.01392864084462905,0.655262676576992};
	double lengthConcrete[] =  {-0.923680800820958, -0.4241968519089583, -0.9667549239890083, -0.6495725724332089, -0.7192532140737757, -0.7255940359152313, -0.21910952173388007, -0.17088574747445137, -0.6106219809556754, -0.3831722924545655, 0.7420807554281152, -0.6984837344323753, 0.44107083280440174, 0.9383460082468607, 0.49699494830205593, 0.08322167761573732, 0.44753619915948883, 0.8907380293732725, 0.7551040213917468, -0.08369430096589922, -0.5861193088253818, 0.07376091977963342, 0.7884874788388441, -0.3726054351136945, 0.4679935012308758, -0.27316128863284606, 0.028601531993281554, 0.8545481538884996, 0.6437943582798766, 0.21898832036381768, 0.708663398816133, -0.7730241257827326};
	double lengthConcrete32[] = {0.2555752836700756, -0.3706141100827832, -0.3328255004293449, 0.8051607187273113, -0.2542568294239081, 0.3254303459947345, 0.04072319279175579, -0.9492441928779090, -0.4064138185055319, -0.6273592451842601, 0.1985476962812595, -0.5343833212191307, -0.3109329265721914, 0.7925981648931077, 0.9730757396777286, -0.1153616665894674, 0.2318934378422005, 0.09617894481097644, -0.3796494475653738, -0.6118173399572018, 0.02712805442982380, 0.7468049517861790, 0.9613608510057130, -0.6383264727674991, -0.7189161075672504, 0.6166963505535785, 0.05947328040228178, -0.5654445241185055, 0.08133358305005228, -0.3483258882073241, 0.7127105203542661, 0.2391802747966369};
	// double lengthConcrete8[] = {-0.4480226512328169, 0.4928344278504513, 0.9721753924653720, -0.2367285839000268, -0.8164793880375470, -0.08279708727936647, 0.2051669918684522, -0.08614910173451834};
	// double lengthConcrete8[] = {-0.4480226512328169, 0.4928344278504513, -0.8164793880375470, -0.08279708727936647, 0.9721753924653720, -0.2367285839000268, 0.2051669918684522, -0.08614910173451834};
	// double lengthConcrete8[] = {-0.9171786598573886, -0.6968483735543598, -0.6787219230526584, -0.5674278557198684, 0.35209781545952484, 0.7621351560306098, 0.8244677200486106, 0.9214761206455304};
	// above: sorted. below: unsorted.
	double lengthConcrete8[] = {-0.6787219230526584, 0.7621351560306098, -0.9171786598573886, 0.35209781545952484, -0.5674278557198684, 0.9214761206455304, 0.8244677200486106, -0.6968483735543598};
	// better lengths ? see John's email from 4 Sept 2024

long numPendulums;						// current number of pendulums
long subNum;							// numPendulums - 1

long dtPerPeriod;						// n of integration steps per forcing period

double avgLength;						// variable pendulum parameters
double gama;	// formerly viscosity
double kappa;	// formerly coupling
double tau0;	// formerly torqueDC
double tau1;	// formerly torqueAC
double torqueACPeriod, omega;	// omega formerly torqueACFreq

double initAngle;						// initial condition parameters
double initAngVel;
double delta;	// formerly disorder
unsigned long constantSeed;

long nLyapTrans;						// now defunct?
long stepsPerRenorm;					// Auto Lyapunov variables
long numRenorms;
long nLyapTrials;

double startDelta;
double endDelta;
double stepDelta;
double startKappa;
double endKappa;
double stepKappa;

long double t;							// times
double dt;
unsigned long nDt;						// => can handle 2^32-1 (> 4 billion) steps

long	disorderType;       	        // option variables
long	intMethod;
boolean useConstantSeed;
boolean randomizeInitCond;
boolean ifRankIsDelta;	// use rank for delta?

boolean ifLoopKappa, ifLoopDelta, ifLogStepKappa, ifLogStepDelta;

boolean oneDtErrorPerDelta;

double missingDeltas[] = {0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.9825, 0.9825, 0.9825, 0.9825, 0.9825, 0.9825, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.9875, 0.9875, 0.9875, 0.9875, 0.9875, 0.9875, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.9925, 0.9925, 0.9925, 0.9925, 0.9925, 0.9925, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.9975, 0.9975, 0.9975, 0.9975, 0.9975, 0.9975, 0.9975, 0.9975, 0.9975, 0.9975};
double missingKappas[] = {6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 6.5621, 6.71534, 6.87215, 21.7944, 22.3034, 22.8242, 1.72024, 6.41236, 6.5621, 6.71534, 6.87215, 20.8111, 21.2971, 21.7944, 22.3034, 22.8242};

/**********************************
 *
 * Function Prototypes 
 *
 *********************************/

int main(int argc, char *argv[]);
    void InitPendulums(char *argv[], int rank);
	void ReadConfigFile(char *argv[]);
	void GetRandomSeed(void);
	void ResetPendulums(void);
	void ResetLengths(void);

void LoopFuncOverCoupDis(void (*func)(void), boolean ifLog);

void EvolvePendulums(void);
	void Derivatives(double t, double x[], double dx_dt[]);
	void Euler(void);
	void RK4(void);
		void AccelFree(double *x, double *v, double t, double *a);
	void RK5Driver(int nEqn);
		void rk5(double y[], double dy_dx[], int n, double x, double h,	// in
         		double yOut[]);											// out
	void RK45Driver(int nEqn);
		void rk45(double y[], double dy_dx[], int n, double x, double h,// in
        		double yOut[], double yErr[]);                        	// out
void AutoEvolveLyapunov(void);
	double EvolveAndEstimateMaxLyap(void);
	double GetLyapunov(void);
		void LyapunovEuler(void);
		void LyapunovRK4(void);
		void DeltaAccelFree(double *x, double *v, double *dx, double *dv, double *a);
		void ResetDeltaVector(void);
		double GetDeltaVectorLength(void);
		void RenormalizeDeltaVector(double norm);
void BurnTransient(long long nBurn);

void LoopFunc(boolean ifLoop1, double *var1, double min1, double max1, int numSteps1, boolean ifLog1,
			  boolean ifLoop2, double *var2, double min2, double max2, int numSteps2, boolean ifLog2,
		void (*func)(void));
		