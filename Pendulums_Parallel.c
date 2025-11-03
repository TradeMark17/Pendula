
// Parallel version of Pendulums.c using MPI
// Pass config file after executable

// Future work: write NEW RK4 and Euler to take x instead of angle, angVel
// should eliminate Euler, RK4, LyapEuler, LyapRK4, AccelFree, DeltaAccelFree, GetLyap, and all GetLyap dependencies
// should use EvolveAndEst for lyap with any integrator
// (test)

#include "./Pendulums.h"
#include <mpi.h>

//--------------------------------------------------------------------
int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);								// begin parallel environment

	int rank, size;										// rank, size = thread #, total # of threads
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
//--------------------------------------------------------------------

	InitPendulums(argv, rank);

    LoopFunc(ifLoopKappa, &kappa, startKappa, endKappa, stepKappa, ifLogStepKappa,
             ifLoopDelta, &delta, startDelta, endDelta, stepDelta, ifLogStepDelta,
        AutoEvolveLyapunov);
    
    // double allKappas[400];
    // double currentKappa = startKappa;
    // double four = stepKappa/size;
    // double factor = pow(endKappa/startKappa, 1.0/(stepKappa-1.0));
    // for (int i=0; i<stepKappa; i++)
    // {
    //     allKappas[i] = currentKappa;
    //     currentKappa *= factor;
    // }
    // for (int i=0; i<four; i++)
    // {
    //     kappa = allKappas[4*rank + i];//each of the 100 nodes gets 4 kappas, each divided into 8 deltas
    //     for (int d=0; d<8; d++)
    //     {
    //         oneDtErrorPerDelta = true;
    //         delta = 0.98 + d*0.0025;
    //         AutoEvolveLyapunov();
    //     }
    // }

    // delta = missingDeltas[rank];
    // kappa = missingKappas[rank];
    // AutoEvolveLyapunov();
    
//--------------------------------------------------------------------
	MPI_Barrier(MPI_COMM_WORLD);						// wait for all threads to reach this point (does this actually work?)
	MPI_Finalize();										// end parallel environment

    return 0;
}

//--------------------------------------------------------------------
void ReadConfigFile(char *argv[])
{
	FILE *fr = fopen(argv[1], "r");
	for (int i=0; i<64; i++) // max length of arrays arb. set to 100
    {
        if (i%2 == 1)
        {
            switch (i)
            {
                case 5:
                case 7:
                case 9:
                case 11:
                case 13:
                case 15:
                case 17:
                case 27:
                case 29:
                case 33:
                case 35:
                case 37:
                case 39:
                case 45:
                case 51:
                case 63: fscanf(fr, "%ld", &config_vars_ld[i]);  break;
                default: fscanf(fr, "%lf", &config_vars_lf[i]);
            }
        }
        else fscanf(fr, "%s", &config_vars_s[0]);
    }
	fclose(fr);

}//ReadConfigFile

//--------------------------------------------------------------------
void InitPendulums(char *argv[], int rank)
{
    ReadConfigFile(argv);

	numPendulums = N_PEN;
	subNum = SUB_PEN;

    ifRankIsDelta = (boolean) config_vars_ld[63];

    delta = config_vars_lf[1];
    // if (ifRankIsDelta) delta = 2.65893 + rank*(6.9825-2.65893) / 100.0;
    if (ifRankIsDelta) delta = rank / 100.0;
    kappa = config_vars_lf[3];

    disorderType = config_vars_ld[5];
    intMethod = config_vars_ld[7];
    randomizeInitCond = (boolean) config_vars_ld[9];

    initAngle = PI;
	initAngVel = 4;

	constantSeed = 3141;
    useConstantSeed = false;

    ifLoopKappa = (boolean) config_vars_ld[11];
    ifLoopDelta = (boolean) config_vars_ld[13];
    ifLogStepKappa = (boolean) config_vars_ld[15];
    ifLogStepDelta = (boolean) config_vars_ld[17];

    dtTry = config_vars_lf[19];
    dtMin = config_vars_lf[21];
    dtMax = config_vars_lf[23];
    error = config_vars_lf[25];

    dtPerPeriod = config_vars_ld[27];
    nLyapTrans = config_vars_ld[29];

    tDiverge = config_vars_lf[31];
    nDiverge = config_vars_ld[33];
    stepsPerRenorm = config_vars_ld[35];
    numRenorms = config_vars_ld[37];
    nLyapTrials = config_vars_ld[39];

    startDelta = delta + config_vars_lf[41];
    endDelta = delta + config_vars_lf[43];
    stepDelta = config_vars_ld[45];

    startKappa = config_vars_lf[47];
    endKappa = config_vars_lf[49];
    stepKappa = config_vars_ld[51];

    avgLength = config_vars_lf[53];
    gama = config_vars_lf[55];
    tau0 = config_vars_lf[57];
    tau1 = config_vars_lf[59];
    omega = config_vars_lf[61];

	torqueACPeriod = 2.0*PI/omega;

	dt = torqueACPeriod/dtPerPeriod;

	GetRandomSeed();
	
	ResetLengths();							// create pendulum array
	
	ResetPendulums();						// set initial conditions to pendulums
	
}//InitPendulums

//--------------------------------------------------------------------
void GetRandomSeed(void)
{
	if (useConstantSeed)					// constant seed pseudo-random n generator
		srandom( constantSeed );
	else
		srandom( clock() );					// randomly seed pseudo-random n generator

}//GetRandomSeed

//--------------------------------------------------------------------
void ResetPendulums(void)
{		
	nDt = 0; // used in Euler and RK4 (unnecessary)
	t = 0.0;
    dtTry = config_vars_lf[19]; // reset to init. value

	for (int n=0; n<numPendulums; n++)
	{
		if (randomizeInitCond == true)		// random init. conds.
		{
			angle[n] = RAND_IN(-initAngle, initAngle);
			angVel[n] = RAND_IN(-initAngVel, initAngVel);
		}
		else								// zero init. conds.
		{
			angle[n] = 0;
			angVel[n] = 0;
		}
		x[n] = angle[n];
		x[n + numPendulums] = angVel[n];
	}
	// x[2 * numPendulums] = RAND_IN(-1, 1); // phase
	
}//ResetPendulums

//--------------------------------------------------------------------
void ResetLengths(void)
{		
	double freq = 2*PI/(subNum);
	double amp = avgLength * delta;
	
	double shortest = avgLength - amp;
	double longest = avgLength + amp;
		
	for (int n=0; n<numPendulums; n++)
    {
		switch (disorderType)
        {
			case homo:			length[n] = avgLength;									break;
			case random:		length[n] = avgLength + amp*lengthConcrete8[n];         break;
			case alternate:		if (n%2 == 0) length[n] = longest;
								else length[n] = shortest;				        		break;
            case linear:        length[n] = avgLength - delta*(1.0 - 2.0*n/subNum);     break;
            case impurity:      if (n == 2) length[n] = avgLength + delta;
                                else length[n] = avgLength - delta/(subNum);            break;
		}
	}
	
}//ResetLengths

//--------------------------------------------------------------------
void EvolvePendulums(void)
{
	ResetLengths();
    ResetPendulums();
    BurnTransient(nLyapTrans);

    for (long long step = 1; step <= 100*dtPerPeriod; step++)
    {
        switch (intMethod)
        {
            case euler:     Euler();            break;
            case RuKu4:     RK4();              break;
            case RuKu5:     RK5Driver(DIM);     break;
            case RuKu45:    RK45Driver(DIM);    break;
        }
        
		if (step%50 == 0)
            printf("t = %.16Lf\t dt = %.16lf\t error = %.16lf\n", t, dtTry, errorNow);
    }

}//EvolvePendulums

//--------------------------------------------------------------------
/****
 * LoopFunc must have all looped variables in front of all non-looped variables
 * i.e. ifLoop1 = false, ifLoop2 = true <-- WILL NOT WORK as intended
 * also, (obviously) min must be >0 if ifLog == true
 * 
 * Future work: let LoopFunc choose between int and double vars
 * Worth it to adapt to work with
 *      LoopFunc(...LoopFunc(..., func))
 * and
 *      LoopFunc(..., func)
 * etc. ?
****/
void LoopFunc(boolean ifLoop1, double *var1, double min1, double max1, int numSteps1, boolean ifLog1,
              boolean ifLoop2, double *var2, double min2, double max2, int numSteps2, boolean ifLog2,
            void (*func)(void))//easily tileable/scaleable to more looped variables
{
    if (ifLoop1)
    {
        double factor1;
        if (ifLog1) factor1 = pow(max1/min1, 1.0/(numSteps1-1.0));
        else factor1 = (max1-min1) / numSteps1;

        (*var1) = min1;

        for (int step1 = 1; step1 <= numSteps1; step1++)
        {
            if (ifLoop2)
            {
                double factor2;
                if (ifLog2) factor2 = pow(max2/min2, 1.0/(numSteps2-1.0));
                else factor2 = (max2-min2) / numSteps2;

                (*var2) = min2;

                for (int step2 = 1; step2 <= numSteps2; step2++)
                {
                    (*func)();

                    if (ifLog2) (*var2) *= factor2;//log step var2
                    else (*var2) += factor2;//lin step var2
                }
            }
            else (*func)();

            if (ifLog1) (*var1) *= factor1;//log step var1
            else (*var1) += factor1;//lin step var1
        }
    }
    else (*func)();

    /********************************
    * STRUCTURE of LoopFunc()
    * 
    *  loop 1st var?
    *      YES
    *          set up factor for 1st var
    *          init. 1st var
    *          FOR(1st var):
    *              loop 2nd var?
    *                  YES
    *                      set up factor for 2nd var
    *                      init. 2nd var
    *                      FOR(2nd var):
    *                          do func
    *                  NO
    *                      do func
    *      NO
    *          do func
    ********************************/
}//LoopFunc

//--------------------------------------------------------------------		
void Euler(void)
{
	double alpha[N_PEN];    // changed from MAX_P to N_PEN
	
	AccelFree(angle,angVel,t,alpha);
	
	for (int n=0; n<numPendulums; n++)
	{
		angVel[n] += alpha[n]*dt; // Euler-Cromer integration // semi-implicit Euler
		angle[n] += angVel[n]*dt;				
	}
	
	nDt++;
	t = nDt*dt;
	
}//Euler

//--------------------------------------------------------------------
void RK4(void)
{
	double theta1[N_PEN], theta2[N_PEN], theta3[N_PEN];
	double omega1[N_PEN], omega2[N_PEN], omega3[N_PEN];
	double alpha0[N_PEN], alpha1[N_PEN], alpha2[N_PEN], alpha3[N_PEN];
	
	AccelFree(angle, angVel, t, alpha0);
	
	for (int n=0; n<numPendulums; n++)
	{
		theta1[n] = angle[n] + angVel[n]*dt/2;
		omega1[n] = angVel[n] + alpha0[n]*dt/2;
	}
	double t1 = t + dt/2;
	AccelFree(theta1, omega1, t1, alpha1);
	
	for (int n=0; n<numPendulums; n++)
	{
		theta2[n] = angle[n] + omega1[n]*dt/2;
		omega2[n] = angVel[n] + alpha1[n]*dt/2;
	}
	double t2 = t + dt/2;
	AccelFree(theta2, omega2, t2, alpha2);
	
	for (int n=0; n<numPendulums; n++)
	{
		theta3[n] = angle[n] + omega2[n]*dt;
		omega3[n] = angVel[n] + alpha2[n]*dt;
	}
	double t3 = t + dt;
	AccelFree(theta3, omega3, t3, alpha3);

	for (int n=0; n<numPendulums; n++)
	{
		angle[n] += (angVel[n]/6 + omega1[n]/3 + omega2[n]/3 + omega3[n]/6)*dt; 	// x += v*dt
		angVel[n] += (alpha0[n]/6 + alpha1[n]/3 + alpha2[n]/3 + alpha3[n]/6)*dt;	// v += a*dt
	}
	
	nDt ++;
	t = nDt*dt;
    // why not t += dt; ?
	
}//RK4

//--------------------------------------------------------------------
void RK5Driver(int nEqn)
{
    Derivatives(t, x, dx_dt);
    rk5(x, dx_dt, nEqn, t, dtTry, x);
    t += dtTry;
}//RK5Driver

//--------------------------------------------------------------------
void RK45Driver(int nEqn)
{
    Derivatives(t, x, dx_dt);

    boolean step_accepted = false;
    while (!step_accepted)
    {
        rk45(x, dx_dt, nEqn, t, dtTry, temp_x, xError);

        for(short i = 0; i < nEqn; i++)
            xScale[i] = fabs(temp_x[i]) + fabs(dx_dt[i]*dtTry) + DBL_EPSILON;

        errorNow = DBL_EPSILON;   // 2.2e-16 on Mac Mini
        for(short i = 0; i < nEqn; i++)
            errorNow = MAX(errorNow, fabs(xError[i] / xScale[i]));
        
        if (errorNow <= error)
        {
            t += dtTry;
            step_accepted = true;
            for (int i = 0; i < nEqn; i++)
                x[i] = temp_x[i];
        }

        if (dtTry == dtMin && step_accepted == false)
        {
            if (oneDtErrorPerDelta)
            {
                printf("(delta = %lf) ERROR: dtTry at min. value and errorNow too high; now skipping forward\n", delta);
                oneDtErrorPerDelta = false;
            }
            step_accepted = true;
        }
        
        //dtTry *= 0.9 * pow(error / errorNow, step_accepted*0.2+(!step_accepted)*0.25);
        dtTry *= 0.9 * pow(error / errorNow, 0.2); // err ~ dt^5 --> dt ~ err^0.2
        dtTry = PIN(dtMin, dtTry, dtMax); // not too small or too large
    }
}//RK45Driver

//--------------------------------------------------------------------
void rk5(double y[], double dy_dx[], int n, double x, double h, // in
         double yOut[])                                         // out
{
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+b21*h*dy_dx[i];
    Derivatives(x+a2*h,yTemp,ak2);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b31*dy_dx[i]+b32*ak2[i]);
    Derivatives(x+a3*h,yTemp,ak3);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b41*dy_dx[i]+b42*ak2[i]+b43*ak3[i]);
    Derivatives(x+a4*h,yTemp,ak4);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b51*dy_dx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
    Derivatives(x+a5*h,yTemp,ak5);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b61*dy_dx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
    Derivatives(x+a6*h,yTemp,ak6);
    
    for(int i = 0; i < n; i++) // RK5 output
        yOut[i] = y[i]+h*(c1*dy_dx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
}//rk5

//--------------------------------------------------------------------
void rk45(double y[], double dy_dx[], int n, double x, double h,// in
           double yOut[], double yErr[])                        // out
{
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+b21*h*dy_dx[i];
    Derivatives(x+a2*h,yTemp,ak2);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b31*dy_dx[i]+b32*ak2[i]);
    Derivatives(x+a3*h,yTemp,ak3);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b41*dy_dx[i]+b42*ak2[i]+b43*ak3[i]);
    Derivatives(x+a4*h,yTemp,ak4);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b51*dy_dx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
    Derivatives(x+a5*h,yTemp,ak5);
    
    for(int i = 0; i < n; i++)
        yTemp[i] = y[i]+h*(b61*dy_dx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
    Derivatives(x+a6*h,yTemp,ak6);
    
    for(int i = 0; i < n; i++) // RK5 output
        yOut[i] = y[i]+h*(c1*dy_dx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
        
    for(int i = 0; i < n; i++) // RK5 - RK4 error estimate
        yErr[i] = h*(dc1*dy_dx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);
}//rk45

//--------------------------------------------------------------------
void AccelFree(double *x, double *v, double t_, double *a)
{
	double torqueSinPhase = tau1 * sin(omega*t_);  // t_ = t' = t1
	
	{// left end
		a[0] = ( - gama*v[0]
				- length[0]*sin(x[0])
				+ tau0 + torqueSinPhase
				+ kappa*(x[1]-x[0]) ) / (SQR(length[0]));
	}
	for (int n=1; n<subNum; n++)	
	{// interior
		a[n] = ( - gama*v[n]
				- length[n]*sin(x[n])
				+ tau0 + torqueSinPhase
				+ kappa*(x[n+1]+x[n-1]-2*x[n]) ) / (SQR(length[n]));
	}
	{// right end
		a[subNum] = ( - gama*v[subNum]
					- length[subNum]*sin(x[subNum])
					+ tau0 + torqueSinPhase 
					+ kappa*(x[numPendulums-2]-x[subNum]) ) / (SQR(length[subNum]));	
	}
}//AccelFree

//--------------------------------------------------------------------
void DeltaAccelFree(double *x, double *v, double *dx, double *dv, double *da)
{
	{// left end
		
		double lenSqr = SQR(length[0]);
		double k0 = kappa/lenSqr;
		double c0 = (-length[0]*cos(x[0])-kappa)/lenSqr;
		double g0 = -gama/lenSqr;
		
		da[0] = c0*dx[0] + k0*dx[1] + g0*dv[0];
	}
	for (int n=1; n<subNum; n++)
	{// interior
	
		double lenSqr = SQR(length[n]);
		double kn = kappa/lenSqr;
		double cn = (-length[n]*cos(x[n])-2*kappa)/lenSqr;
		double gn = -gama/lenSqr;
		
		da[n] = kn*(dx[n-1] + dx[n+1]) + cn*dx[n] + gn*dv[n];
	}
	{// right end
	
		double lenSqr = SQR(length[subNum]);
		double ks = kappa/lenSqr;
		double cs = (-length[subNum]*cos(x[subNum])-kappa)/lenSqr;
		double gs = -gama/lenSqr;
		
		da[subNum] = ks*dx[numPendulums-2] + cs*dx[subNum] + gs*dv[subNum];
	}
}//DeltaAccelFree

//--------------------------------------------------------------------
void BurnTransient(long long nBurn)
{
	for (long long i=0; i<nBurn*dtPerPeriod; i++)
	{
        switch (intMethod)
        {
            case euler:     Euler();            break;
            case RuKu4:     RK4();              break;
            case RuKu5:     RK5Driver(DIM);     break;
            case RuKu45:    RK45Driver(DIM);    break;
        }
	}
    switch (intMethod)
    {
        case euler:
        case RuKu4:
            for (int i=0; i<numPendulums; i++)
            {
                x[i] = angle[i];
                x[i+numPendulums] = angVel[i];
            }
            break;
    }
}//BurnTransient

//--------------------------------------------------------------------
void AutoEvolveLyapunov()
{
	double lambdaMaxMean = 0.0;
    double lambdaMaxMeanSqr = 0.0;

	for (long trial=0; trial<nLyapTrials; trial++)		// averaging nLyapTrials times
	{			
		ResetLengths();									// new realization of the disorder
		ResetPendulums();								// new realization of the initial conditions
		BurnTransient(nLyapTrans);						// burn the transient
	
        double lambdaMax;
        switch (intMethod)
        {
            case euler:
            case RuKu4:     lambdaMax = GetLyapunov();              break;
            case RuKu5:
            case RuKu45:    lambdaMax = EvolveAndEstimateMaxLyap(); break;
        }
        lambdaMaxMean += lambdaMax;
        lambdaMaxMeanSqr += SQR(lambdaMax);
	}
    lambdaMaxMean /= nLyapTrials;
    lambdaMaxMeanSqr /= nLyapTrials;
    double lambdaMaxSigma = sqrt(lambdaMaxMeanSqr - SQR(lambdaMaxMean));

    if (nLyapTrials == 1) printf("%.16lf\t %.16lf\t %.16lf\n", delta, kappa, lambdaMaxMean);
    else printf("%.16lf\t %d\t %.16lf\t %.16lf ± %.16lf\t %lf\t %lf\n", error, nDiverge, tDiverge, lambdaMaxMean, lambdaMaxSigma, delta, kappa);

    // if (delta == 0.5 && kappa == 0.01)
    // {
    //     for (int i = 0; i < numPendulums; i++)
    //     {
    //         printf("Length of pend. %d = %.16lf\n", i+1, length[i]);
    //     }
    // }

}//AutoEvolveLyapunov

//--------------------------------------------------------------------
double EvolveAndEstimateMaxLyap(void)
{
    double deltaVec[DIM];
    for (int i = 0; i < DIM; i++)
        deltaVec[i] = RAND_IN(-1, 1); // normalize perturbation later

    double logSum = 0.0;
    for (int n = 0; n < nDiverge; n++)
    {
        // init Phi
        int s = DIM; // Phi = id
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                x[s++] = (i == j)? 1 : 0;

        dtTry = config_vars_lf[19]; // reset dt to init. value
        
        while (t <= (n + 1) * tDiverge)
        {
            switch (intMethod)
            {// evolve t & x = {xVec, PhiMat}
                case RuKu5:     RK5Driver(N_EQN);   break;
                case RuKu45:    RK45Driver(N_EQN);  break;
            }
        }

        // normalize perturbation
        double sum2 = 0.0;
        for (int i = 0; i < DIM; i++)
        {
            sum2 += SQR(deltaVec[i]);
        }
        double deltaMag = sqrt(sum2);
        
        double uVec[DIM];
        for (int i = 0; i < DIM; i++)
        {
            uVec[i] = deltaVec[i] / deltaMag;
        }
        // matrix multiply updates deltaVec <- PhiMat.uVec
        s = DIM;
        for (int i = 0; i < DIM; i++)
        {
            deltaVec[i] = 0;
            for (int j = 0; j < DIM; j++)
            {
                int index = s++;
                deltaVec[i] += x[index] * uVec[j];
            }
        }
        
        logSum += log(deltaMag);
	}

	double logMean = logSum / nDiverge;
    double lambdaMax = logMean / tDiverge;
    return lambdaMax;

}//EvolveAndEstimateMaxLyap

//--------------------------------------------------------------------
double GetLyapunov(void)
{
	ResetDeltaVector();										// compute deltas for each pendulum
	double vLength = GetDeltaVectorLength();
	double logSum = 0;

	for (long renorm = 1; renorm <= numRenorms; renorm++)
    {
		RenormalizeDeltaVector(vLength);					// renormalize the variation
		
		for (long step = 1; step <= stepsPerRenorm; step++)		// integrate between renormalizations
            switch (intMethod)
            {
                case euler:     LyapunovEuler();        break;
                case RuKu4:     LyapunovRK4();          break;
            }
		
		vLength = GetDeltaVectorLength();
		
		logSum += log(vLength);								// sum the logs of the norms
	}	
    double lyap = logSum/(numRenorms*stepsPerRenorm*dt);
		
	return lyap;

}//GetLyapunov

//--------------------------------------------------------------------
void LyapunovEuler(void)
{
	double alpha[N_PEN];    // changed from MAX_P to N_PEN
	double deltaAlpha[N_PEN];   // changed from MAX_P to N_PEN
	
	AccelFree(angle, angVel, t, alpha);
	DeltaAccelFree(angle, angVel, deltaAngle, deltaAngVel, deltaAlpha);
	
	for (int n=0; n<numPendulums; n++)
	{
		angVel[n] += alpha[n]*dt;
		angle[n] += angVel[n]*dt;
		
		deltaAngVel[n] += deltaAlpha[n]*dt;
		deltaAngle[n] += deltaAngVel[n]*dt;
	}
	
	nDt ++;
	t = nDt*dt;
}//LyapunovEuler

//--------------------------------------------------------------------
void LyapunovRK4(void)
{
    // changed from MAX_P to N_PEN
	double theta1[N_PEN],theta2[N_PEN],theta3[N_PEN];
	double omega1[N_PEN],omega2[N_PEN],omega3[N_PEN];
	double alpha0[N_PEN],alpha1[N_PEN],alpha2[N_PEN],alpha3[N_PEN];
	double deltaTheta1[N_PEN],deltaTheta2[N_PEN],deltaTheta3[N_PEN];
	double deltaOmega1[N_PEN],deltaOmega2[N_PEN],deltaOmega3[N_PEN];
	double deltaAlpha0[N_PEN],deltaAlpha1[N_PEN],deltaAlpha2[N_PEN],deltaAlpha3[N_PEN];
	
	AccelFree(angle,angVel,t,alpha0);
	DeltaAccelFree(angle,angVel,deltaAngle,deltaAngVel,deltaAlpha0);
	
	for (int n=0; n<numPendulums; n++)
	{
		theta1[n] = angle[n] + angVel[n]*dt/2;
		omega1[n] = angVel[n] + alpha0[n]*dt/2;
		
		deltaTheta1[n] = deltaAngle[n] + deltaAngVel[n]*dt/2;
		deltaOmega1[n] = deltaAngVel[n] + deltaAlpha0[n]*dt/2;
	}
	double t1 = t + dt/2;
	AccelFree(theta1,omega1,t1,alpha1);
	DeltaAccelFree(theta1,omega1,deltaTheta1,deltaOmega1,deltaAlpha1);
	
	for (int n=0; n<numPendulums; n++)
	{
		theta2[n] = angle[n] + omega1[n]*dt/2;
		omega2[n] = angVel[n] + alpha1[n]*dt/2;
		
		deltaTheta2[n] = deltaAngle[n] + deltaOmega1[n]*dt/2;
		deltaOmega2[n] = deltaAngVel[n] + deltaAlpha1[n]*dt/2;
	}
	double t2 = t + dt/2;
	AccelFree(theta2,omega2,t2,alpha2);
	DeltaAccelFree(theta2,omega2,deltaTheta2,deltaOmega2,deltaAlpha2);
	
	for (int n=0; n<numPendulums; n++)
	{
		theta3[n] = angle[n] + omega2[n]*dt;
		omega3[n] = angVel[n] + alpha2[n]*dt;
		
		deltaTheta3[n] = deltaAngle[n] + deltaOmega2[n]*dt;
		deltaOmega3[n] = deltaAngVel[n] + deltaAlpha2[n]*dt;
	}
	double t3 = t + dt;
	AccelFree(theta3,omega3,t3,alpha3);
	DeltaAccelFree(theta3,omega3,deltaTheta3,deltaOmega3,deltaAlpha3);

	for (int n=0; n<numPendulums; n++)
	{
		angle[n] += (angVel[n]/6 + omega1[n]/3 + omega2[n]/3 + omega3[n]/6)*dt; 	// x += v*dt
		angVel[n] += (alpha0[n]/6 + alpha1[n]/3 + alpha2[n]/3 + alpha3[n]/6)*dt;	// v += a*dt
		
		deltaAngle[n] += (deltaAngVel[n]/6 + deltaOmega1[n]/3 + deltaOmega2[n]/3 + deltaOmega3[n]/6)*dt; 	// dx += dv*dt
		deltaAngVel[n] += (deltaAlpha0[n]/6 + deltaAlpha1[n]/3 + deltaAlpha2[n]/3 + deltaAlpha3[n]/6)*dt;	// dv += da*dt
	}
	
	nDt ++;
	t = nDt*dt;
	
}//LyapunovRK4

//--------------------------------------------------------------------
double GetDeltaVectorLength(void)
{
	double normSum = 0;
	
	for (int i=0; i<numPendulums; i++)
		normSum += SQR(deltaAngle[i]) + SQR(deltaAngVel[i]);
	
	return sqrt(normSum);
}//GetDeltaVectorLength

//--------------------------------------------------------------------
void RenormalizeDeltaVector(double norm)
{	
	for (int i=0; i<numPendulums; i++)
    {
        deltaAngle[i] /= norm;
        deltaAngVel[i] /= norm;
    }
}//RenormalizeDeltaVector

//--------------------------------------------------------------------
void ResetDeltaVector(void)
{
	for (int i=0; i<numPendulums; i++)
    {
		deltaAngle[i] = (double) random()/RAND_MAX;
		deltaAngVel[i] = (double) random()/RAND_MAX;
	}
}//ResetDeltaVector

//--------------------------------------------------------------------
void Derivatives(double t,double x[],  // local variables t, x, dx_dt used first
                     double dx_dt[])
{
    dx_dt[0] = x[8];
    dx_dt[1] = x[9];
    dx_dt[2] = x[10];
    dx_dt[3] = x[11];
    dx_dt[4] = x[12];
    dx_dt[5] = x[13];
    dx_dt[6] = x[14];
    dx_dt[7] = x[15];
    dx_dt[8] = (tau0 + tau1*sin(omega*t) - length[0]*sin(x[0]) + kappa*(-x[0] + x[1]) - gama*x[8])/SQR(length[0]);
    dx_dt[9] = (tau0 + tau1*sin(omega*t) - length[1]*sin(x[1]) + kappa*(x[0] - 2*x[1] + x[2]) - gama*x[9])/SQR(length[1]);
    dx_dt[10] = (tau0 + tau1*sin(omega*t) - length[2]*sin(x[2]) + kappa*(x[1] - 2*x[2] + x[3]) - gama*x[10])/SQR(length[2]);
    dx_dt[11] = (tau0 + tau1*sin(omega*t) - length[3]*sin(x[3]) + kappa*(x[2] - 2*x[3] + x[4]) - gama*x[11])/SQR(length[3]);
    dx_dt[12] = (tau0 + tau1*sin(omega*t) - length[4]*sin(x[4]) + kappa*(x[3] - 2*x[4] + x[5]) - gama*x[12])/SQR(length[4]);
    dx_dt[13] = (tau0 + tau1*sin(omega*t) - length[5]*sin(x[5]) + kappa*(x[4] - 2*x[5] + x[6]) - gama*x[13])/SQR(length[5]);
    dx_dt[14] = (tau0 + tau1*sin(omega*t) - length[6]*sin(x[6]) + kappa*(x[5] - 2*x[6] + x[7]) - gama*x[14])/SQR(length[6]);
    dx_dt[15] = (tau0 + tau1*sin(omega*t) - length[7]*sin(x[7]) + kappa*(x[6] - x[7]) - gama*x[15])/SQR(length[7]);
    dx_dt[16] = x[144];
    dx_dt[17] = x[145];
    dx_dt[18] = x[146];
    dx_dt[19] = x[147];
    dx_dt[20] = x[148];
    dx_dt[21] = x[149];
    dx_dt[22] = x[150];
    dx_dt[23] = x[151];
    dx_dt[24] = x[152];
    dx_dt[25] = x[153];
    dx_dt[26] = x[154];
    dx_dt[27] = x[155];
    dx_dt[28] = x[156];
    dx_dt[29] = x[157];
    dx_dt[30] = x[158];
    dx_dt[31] = x[159];
    dx_dt[32] = x[160];
    dx_dt[33] = x[161];
    dx_dt[34] = x[162];
    dx_dt[35] = x[163];
    dx_dt[36] = x[164];
    dx_dt[37] = x[165];
    dx_dt[38] = x[166];
    dx_dt[39] = x[167];
    dx_dt[40] = x[168];
    dx_dt[41] = x[169];
    dx_dt[42] = x[170];
    dx_dt[43] = x[171];
    dx_dt[44] = x[172];
    dx_dt[45] = x[173];
    dx_dt[46] = x[174];
    dx_dt[47] = x[175];
    dx_dt[48] = x[176];
    dx_dt[49] = x[177];
    dx_dt[50] = x[178];
    dx_dt[51] = x[179];
    dx_dt[52] = x[180];
    dx_dt[53] = x[181];
    dx_dt[54] = x[182];
    dx_dt[55] = x[183];
    dx_dt[56] = x[184];
    dx_dt[57] = x[185];
    dx_dt[58] = x[186];
    dx_dt[59] = x[187];
    dx_dt[60] = x[188];
    dx_dt[61] = x[189];
    dx_dt[62] = x[190];
    dx_dt[63] = x[191];
    dx_dt[64] = x[192];
    dx_dt[65] = x[193];
    dx_dt[66] = x[194];
    dx_dt[67] = x[195];
    dx_dt[68] = x[196];
    dx_dt[69] = x[197];
    dx_dt[70] = x[198];
    dx_dt[71] = x[199];
    dx_dt[72] = x[200];
    dx_dt[73] = x[201];
    dx_dt[74] = x[202];
    dx_dt[75] = x[203];
    dx_dt[76] = x[204];
    dx_dt[77] = x[205];
    dx_dt[78] = x[206];
    dx_dt[79] = x[207];
    dx_dt[80] = x[208];
    dx_dt[81] = x[209];
    dx_dt[82] = x[210];
    dx_dt[83] = x[211];
    dx_dt[84] = x[212];
    dx_dt[85] = x[213];
    dx_dt[86] = x[214];
    dx_dt[87] = x[215];
    dx_dt[88] = x[216];
    dx_dt[89] = x[217];
    dx_dt[90] = x[218];
    dx_dt[91] = x[219];
    dx_dt[92] = x[220];
    dx_dt[93] = x[221];
    dx_dt[94] = x[222];
    dx_dt[95] = x[223];
    dx_dt[96] = x[224];
    dx_dt[97] = x[225];
    dx_dt[98] = x[226];
    dx_dt[99] = x[227];
    dx_dt[100] = x[228];
    dx_dt[101] = x[229];
    dx_dt[102] = x[230];
    dx_dt[103] = x[231];
    dx_dt[104] = x[232];
    dx_dt[105] = x[233];
    dx_dt[106] = x[234];
    dx_dt[107] = x[235];
    dx_dt[108] = x[236];
    dx_dt[109] = x[237];
    dx_dt[110] = x[238];
    dx_dt[111] = x[239];
    dx_dt[112] = x[240];
    dx_dt[113] = x[241];
    dx_dt[114] = x[242];
    dx_dt[115] = x[243];
    dx_dt[116] = x[244];
    dx_dt[117] = x[245];
    dx_dt[118] = x[246];
    dx_dt[119] = x[247];
    dx_dt[120] = x[248];
    dx_dt[121] = x[249];
    dx_dt[122] = x[250];
    dx_dt[123] = x[251];
    dx_dt[124] = x[252];
    dx_dt[125] = x[253];
    dx_dt[126] = x[254];
    dx_dt[127] = x[255];
    dx_dt[128] = x[256];
    dx_dt[129] = x[257];
    dx_dt[130] = x[258];
    dx_dt[131] = x[259];
    dx_dt[132] = x[260];
    dx_dt[133] = x[261];
    dx_dt[134] = x[262];
    dx_dt[135] = x[263];
    dx_dt[136] = x[264];
    dx_dt[137] = x[265];
    dx_dt[138] = x[266];
    dx_dt[139] = x[267];
    dx_dt[140] = x[268];
    dx_dt[141] = x[269];
    dx_dt[142] = x[270];
    dx_dt[143] = x[271];
    dx_dt[144] = ((-kappa - cos(x[0])*length[0])*x[16])/SQR(length[0]) + (kappa*x[32])/SQR(length[0]) - (gama*x[144])/SQR(length[0]);
    dx_dt[145] = ((-kappa - cos(x[0])*length[0])*x[17])/SQR(length[0]) + (kappa*x[33])/SQR(length[0]) - (gama*x[145])/SQR(length[0]);
    dx_dt[146] = ((-kappa - cos(x[0])*length[0])*x[18])/SQR(length[0]) + (kappa*x[34])/SQR(length[0]) - (gama*x[146])/SQR(length[0]);
    dx_dt[147] = ((-kappa - cos(x[0])*length[0])*x[19])/SQR(length[0]) + (kappa*x[35])/SQR(length[0]) - (gama*x[147])/SQR(length[0]);
    dx_dt[148] = ((-kappa - cos(x[0])*length[0])*x[20])/SQR(length[0]) + (kappa*x[36])/SQR(length[0]) - (gama*x[148])/SQR(length[0]);
    dx_dt[149] = ((-kappa - cos(x[0])*length[0])*x[21])/SQR(length[0]) + (kappa*x[37])/SQR(length[0]) - (gama*x[149])/SQR(length[0]);
    dx_dt[150] = ((-kappa - cos(x[0])*length[0])*x[22])/SQR(length[0]) + (kappa*x[38])/SQR(length[0]) - (gama*x[150])/SQR(length[0]);
    dx_dt[151] = ((-kappa - cos(x[0])*length[0])*x[23])/SQR(length[0]) + (kappa*x[39])/SQR(length[0]) - (gama*x[151])/SQR(length[0]);
    dx_dt[152] = ((-kappa - cos(x[0])*length[0])*x[24])/SQR(length[0]) + (kappa*x[40])/SQR(length[0]) - (gama*x[152])/SQR(length[0]);
    dx_dt[153] = ((-kappa - cos(x[0])*length[0])*x[25])/SQR(length[0]) + (kappa*x[41])/SQR(length[0]) - (gama*x[153])/SQR(length[0]);
    dx_dt[154] = ((-kappa - cos(x[0])*length[0])*x[26])/SQR(length[0]) + (kappa*x[42])/SQR(length[0]) - (gama*x[154])/SQR(length[0]);
    dx_dt[155] = ((-kappa - cos(x[0])*length[0])*x[27])/SQR(length[0]) + (kappa*x[43])/SQR(length[0]) - (gama*x[155])/SQR(length[0]);
    dx_dt[156] = ((-kappa - cos(x[0])*length[0])*x[28])/SQR(length[0]) + (kappa*x[44])/SQR(length[0]) - (gama*x[156])/SQR(length[0]);
    dx_dt[157] = ((-kappa - cos(x[0])*length[0])*x[29])/SQR(length[0]) + (kappa*x[45])/SQR(length[0]) - (gama*x[157])/SQR(length[0]);
    dx_dt[158] = ((-kappa - cos(x[0])*length[0])*x[30])/SQR(length[0]) + (kappa*x[46])/SQR(length[0]) - (gama*x[158])/SQR(length[0]);
    dx_dt[159] = ((-kappa - cos(x[0])*length[0])*x[31])/SQR(length[0]) + (kappa*x[47])/SQR(length[0]) - (gama*x[159])/SQR(length[0]);
    dx_dt[160] = (kappa*x[16])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[32])/SQR(length[1]) + (kappa*x[48])/SQR(length[1]) - (gama*x[160])/SQR(length[1]);
    dx_dt[161] = (kappa*x[17])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[33])/SQR(length[1]) + (kappa*x[49])/SQR(length[1]) - (gama*x[161])/SQR(length[1]);
    dx_dt[162] = (kappa*x[18])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[34])/SQR(length[1]) + (kappa*x[50])/SQR(length[1]) - (gama*x[162])/SQR(length[1]);
    dx_dt[163] = (kappa*x[19])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[35])/SQR(length[1]) + (kappa*x[51])/SQR(length[1]) - (gama*x[163])/SQR(length[1]);
    dx_dt[164] = (kappa*x[20])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[36])/SQR(length[1]) + (kappa*x[52])/SQR(length[1]) - (gama*x[164])/SQR(length[1]);
    dx_dt[165] = (kappa*x[21])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[37])/SQR(length[1]) + (kappa*x[53])/SQR(length[1]) - (gama*x[165])/SQR(length[1]);
    dx_dt[166] = (kappa*x[22])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[38])/SQR(length[1]) + (kappa*x[54])/SQR(length[1]) - (gama*x[166])/SQR(length[1]);
    dx_dt[167] = (kappa*x[23])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[39])/SQR(length[1]) + (kappa*x[55])/SQR(length[1]) - (gama*x[167])/SQR(length[1]);
    dx_dt[168] = (kappa*x[24])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[40])/SQR(length[1]) + (kappa*x[56])/SQR(length[1]) - (gama*x[168])/SQR(length[1]);
    dx_dt[169] = (kappa*x[25])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[41])/SQR(length[1]) + (kappa*x[57])/SQR(length[1]) - (gama*x[169])/SQR(length[1]);
    dx_dt[170] = (kappa*x[26])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[42])/SQR(length[1]) + (kappa*x[58])/SQR(length[1]) - (gama*x[170])/SQR(length[1]);
    dx_dt[171] = (kappa*x[27])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[43])/SQR(length[1]) + (kappa*x[59])/SQR(length[1]) - (gama*x[171])/SQR(length[1]);
    dx_dt[172] = (kappa*x[28])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[44])/SQR(length[1]) + (kappa*x[60])/SQR(length[1]) - (gama*x[172])/SQR(length[1]);
    dx_dt[173] = (kappa*x[29])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[45])/SQR(length[1]) + (kappa*x[61])/SQR(length[1]) - (gama*x[173])/SQR(length[1]);
    dx_dt[174] = (kappa*x[30])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[46])/SQR(length[1]) + (kappa*x[62])/SQR(length[1]) - (gama*x[174])/SQR(length[1]);
    dx_dt[175] = (kappa*x[31])/SQR(length[1]) + ((-2*kappa - cos(x[1])*length[1])*x[47])/SQR(length[1]) + (kappa*x[63])/SQR(length[1]) - (gama*x[175])/SQR(length[1]);
    dx_dt[176] = (kappa*x[32])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[48])/SQR(length[2]) + (kappa*x[64])/SQR(length[2]) - (gama*x[176])/SQR(length[2]);
    dx_dt[177] = (kappa*x[33])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[49])/SQR(length[2]) + (kappa*x[65])/SQR(length[2]) - (gama*x[177])/SQR(length[2]);
    dx_dt[178] = (kappa*x[34])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[50])/SQR(length[2]) + (kappa*x[66])/SQR(length[2]) - (gama*x[178])/SQR(length[2]);
    dx_dt[179] = (kappa*x[35])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[51])/SQR(length[2]) + (kappa*x[67])/SQR(length[2]) - (gama*x[179])/SQR(length[2]);
    dx_dt[180] = (kappa*x[36])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[52])/SQR(length[2]) + (kappa*x[68])/SQR(length[2]) - (gama*x[180])/SQR(length[2]);
    dx_dt[181] = (kappa*x[37])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[53])/SQR(length[2]) + (kappa*x[69])/SQR(length[2]) - (gama*x[181])/SQR(length[2]);
    dx_dt[182] = (kappa*x[38])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[54])/SQR(length[2]) + (kappa*x[70])/SQR(length[2]) - (gama*x[182])/SQR(length[2]);
    dx_dt[183] = (kappa*x[39])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[55])/SQR(length[2]) + (kappa*x[71])/SQR(length[2]) - (gama*x[183])/SQR(length[2]);
    dx_dt[184] = (kappa*x[40])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[56])/SQR(length[2]) + (kappa*x[72])/SQR(length[2]) - (gama*x[184])/SQR(length[2]);
    dx_dt[185] = (kappa*x[41])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[57])/SQR(length[2]) + (kappa*x[73])/SQR(length[2]) - (gama*x[185])/SQR(length[2]);
    dx_dt[186] = (kappa*x[42])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[58])/SQR(length[2]) + (kappa*x[74])/SQR(length[2]) - (gama*x[186])/SQR(length[2]);
    dx_dt[187] = (kappa*x[43])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[59])/SQR(length[2]) + (kappa*x[75])/SQR(length[2]) - (gama*x[187])/SQR(length[2]);
    dx_dt[188] = (kappa*x[44])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[60])/SQR(length[2]) + (kappa*x[76])/SQR(length[2]) - (gama*x[188])/SQR(length[2]);
    dx_dt[189] = (kappa*x[45])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[61])/SQR(length[2]) + (kappa*x[77])/SQR(length[2]) - (gama*x[189])/SQR(length[2]);
    dx_dt[190] = (kappa*x[46])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[62])/SQR(length[2]) + (kappa*x[78])/SQR(length[2]) - (gama*x[190])/SQR(length[2]);
    dx_dt[191] = (kappa*x[47])/SQR(length[2]) + ((-2*kappa - cos(x[2])*length[2])*x[63])/SQR(length[2]) + (kappa*x[79])/SQR(length[2]) - (gama*x[191])/SQR(length[2]);
    dx_dt[192] = (kappa*x[48])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[64])/SQR(length[3]) + (kappa*x[80])/SQR(length[3]) - (gama*x[192])/SQR(length[3]);
    dx_dt[193] = (kappa*x[49])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[65])/SQR(length[3]) + (kappa*x[81])/SQR(length[3]) - (gama*x[193])/SQR(length[3]);
    dx_dt[194] = (kappa*x[50])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[66])/SQR(length[3]) + (kappa*x[82])/SQR(length[3]) - (gama*x[194])/SQR(length[3]);
    dx_dt[195] = (kappa*x[51])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[67])/SQR(length[3]) + (kappa*x[83])/SQR(length[3]) - (gama*x[195])/SQR(length[3]);
    dx_dt[196] = (kappa*x[52])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[68])/SQR(length[3]) + (kappa*x[84])/SQR(length[3]) - (gama*x[196])/SQR(length[3]);
    dx_dt[197] = (kappa*x[53])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[69])/SQR(length[3]) + (kappa*x[85])/SQR(length[3]) - (gama*x[197])/SQR(length[3]);
    dx_dt[198] = (kappa*x[54])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[70])/SQR(length[3]) + (kappa*x[86])/SQR(length[3]) - (gama*x[198])/SQR(length[3]);
    dx_dt[199] = (kappa*x[55])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[71])/SQR(length[3]) + (kappa*x[87])/SQR(length[3]) - (gama*x[199])/SQR(length[3]);
    dx_dt[200] = (kappa*x[56])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[72])/SQR(length[3]) + (kappa*x[88])/SQR(length[3]) - (gama*x[200])/SQR(length[3]);
    dx_dt[201] = (kappa*x[57])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[73])/SQR(length[3]) + (kappa*x[89])/SQR(length[3]) - (gama*x[201])/SQR(length[3]);
    dx_dt[202] = (kappa*x[58])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[74])/SQR(length[3]) + (kappa*x[90])/SQR(length[3]) - (gama*x[202])/SQR(length[3]);
    dx_dt[203] = (kappa*x[59])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[75])/SQR(length[3]) + (kappa*x[91])/SQR(length[3]) - (gama*x[203])/SQR(length[3]);
    dx_dt[204] = (kappa*x[60])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[76])/SQR(length[3]) + (kappa*x[92])/SQR(length[3]) - (gama*x[204])/SQR(length[3]);
    dx_dt[205] = (kappa*x[61])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[77])/SQR(length[3]) + (kappa*x[93])/SQR(length[3]) - (gama*x[205])/SQR(length[3]);
    dx_dt[206] = (kappa*x[62])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[78])/SQR(length[3]) + (kappa*x[94])/SQR(length[3]) - (gama*x[206])/SQR(length[3]);
    dx_dt[207] = (kappa*x[63])/SQR(length[3]) + ((-2*kappa - cos(x[3])*length[3])*x[79])/SQR(length[3]) + (kappa*x[95])/SQR(length[3]) - (gama*x[207])/SQR(length[3]);
    dx_dt[208] = (kappa*x[64])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[80])/SQR(length[4]) + (kappa*x[96])/SQR(length[4]) - (gama*x[208])/SQR(length[4]);
    dx_dt[209] = (kappa*x[65])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[81])/SQR(length[4]) + (kappa*x[97])/SQR(length[4]) - (gama*x[209])/SQR(length[4]);
    dx_dt[210] = (kappa*x[66])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[82])/SQR(length[4]) + (kappa*x[98])/SQR(length[4]) - (gama*x[210])/SQR(length[4]);
    dx_dt[211] = (kappa*x[67])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[83])/SQR(length[4]) + (kappa*x[99])/SQR(length[4]) - (gama*x[211])/SQR(length[4]);
    dx_dt[212] = (kappa*x[68])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[84])/SQR(length[4]) + (kappa*x[100])/SQR(length[4]) - (gama*x[212])/SQR(length[4]);
    dx_dt[213] = (kappa*x[69])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[85])/SQR(length[4]) + (kappa*x[101])/SQR(length[4]) - (gama*x[213])/SQR(length[4]);
    dx_dt[214] = (kappa*x[70])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[86])/SQR(length[4]) + (kappa*x[102])/SQR(length[4]) - (gama*x[214])/SQR(length[4]);
    dx_dt[215] = (kappa*x[71])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[87])/SQR(length[4]) + (kappa*x[103])/SQR(length[4]) - (gama*x[215])/SQR(length[4]);
    dx_dt[216] = (kappa*x[72])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[88])/SQR(length[4]) + (kappa*x[104])/SQR(length[4]) - (gama*x[216])/SQR(length[4]);
    dx_dt[217] = (kappa*x[73])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[89])/SQR(length[4]) + (kappa*x[105])/SQR(length[4]) - (gama*x[217])/SQR(length[4]);
    dx_dt[218] = (kappa*x[74])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[90])/SQR(length[4]) + (kappa*x[106])/SQR(length[4]) - (gama*x[218])/SQR(length[4]);
    dx_dt[219] = (kappa*x[75])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[91])/SQR(length[4]) + (kappa*x[107])/SQR(length[4]) - (gama*x[219])/SQR(length[4]);
    dx_dt[220] = (kappa*x[76])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[92])/SQR(length[4]) + (kappa*x[108])/SQR(length[4]) - (gama*x[220])/SQR(length[4]);
    dx_dt[221] = (kappa*x[77])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[93])/SQR(length[4]) + (kappa*x[109])/SQR(length[4]) - (gama*x[221])/SQR(length[4]);
    dx_dt[222] = (kappa*x[78])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[94])/SQR(length[4]) + (kappa*x[110])/SQR(length[4]) - (gama*x[222])/SQR(length[4]);
    dx_dt[223] = (kappa*x[79])/SQR(length[4]) + ((-2*kappa - cos(x[4])*length[4])*x[95])/SQR(length[4]) + (kappa*x[111])/SQR(length[4]) - (gama*x[223])/SQR(length[4]);
    dx_dt[224] = (kappa*x[80])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[96])/SQR(length[5]) + (kappa*x[112])/SQR(length[5]) - (gama*x[224])/SQR(length[5]);
    dx_dt[225] = (kappa*x[81])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[97])/SQR(length[5]) + (kappa*x[113])/SQR(length[5]) - (gama*x[225])/SQR(length[5]);
    dx_dt[226] = (kappa*x[82])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[98])/SQR(length[5]) + (kappa*x[114])/SQR(length[5]) - (gama*x[226])/SQR(length[5]);
    dx_dt[227] = (kappa*x[83])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[99])/SQR(length[5]) + (kappa*x[115])/SQR(length[5]) - (gama*x[227])/SQR(length[5]);
    dx_dt[228] = (kappa*x[84])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[100])/SQR(length[5]) + (kappa*x[116])/SQR(length[5]) - (gama*x[228])/SQR(length[5]);
    dx_dt[229] = (kappa*x[85])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[101])/SQR(length[5]) + (kappa*x[117])/SQR(length[5]) - (gama*x[229])/SQR(length[5]);
    dx_dt[230] = (kappa*x[86])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[102])/SQR(length[5]) + (kappa*x[118])/SQR(length[5]) - (gama*x[230])/SQR(length[5]);
    dx_dt[231] = (kappa*x[87])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[103])/SQR(length[5]) + (kappa*x[119])/SQR(length[5]) - (gama*x[231])/SQR(length[5]);
    dx_dt[232] = (kappa*x[88])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[104])/SQR(length[5]) + (kappa*x[120])/SQR(length[5]) - (gama*x[232])/SQR(length[5]);
    dx_dt[233] = (kappa*x[89])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[105])/SQR(length[5]) + (kappa*x[121])/SQR(length[5]) - (gama*x[233])/SQR(length[5]);
    dx_dt[234] = (kappa*x[90])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[106])/SQR(length[5]) + (kappa*x[122])/SQR(length[5]) - (gama*x[234])/SQR(length[5]);
    dx_dt[235] = (kappa*x[91])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[107])/SQR(length[5]) + (kappa*x[123])/SQR(length[5]) - (gama*x[235])/SQR(length[5]);
    dx_dt[236] = (kappa*x[92])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[108])/SQR(length[5]) + (kappa*x[124])/SQR(length[5]) - (gama*x[236])/SQR(length[5]);
    dx_dt[237] = (kappa*x[93])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[109])/SQR(length[5]) + (kappa*x[125])/SQR(length[5]) - (gama*x[237])/SQR(length[5]);
    dx_dt[238] = (kappa*x[94])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[110])/SQR(length[5]) + (kappa*x[126])/SQR(length[5]) - (gama*x[238])/SQR(length[5]);
    dx_dt[239] = (kappa*x[95])/SQR(length[5]) + ((-2*kappa - cos(x[5])*length[5])*x[111])/SQR(length[5]) + (kappa*x[127])/SQR(length[5]) - (gama*x[239])/SQR(length[5]);
    dx_dt[240] = (kappa*x[96])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[112])/SQR(length[6]) + (kappa*x[128])/SQR(length[6]) - (gama*x[240])/SQR(length[6]);
    dx_dt[241] = (kappa*x[97])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[113])/SQR(length[6]) + (kappa*x[129])/SQR(length[6]) - (gama*x[241])/SQR(length[6]);
    dx_dt[242] = (kappa*x[98])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[114])/SQR(length[6]) + (kappa*x[130])/SQR(length[6]) - (gama*x[242])/SQR(length[6]);
    dx_dt[243] = (kappa*x[99])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[115])/SQR(length[6]) + (kappa*x[131])/SQR(length[6]) - (gama*x[243])/SQR(length[6]);
    dx_dt[244] = (kappa*x[100])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[116])/SQR(length[6]) + (kappa*x[132])/SQR(length[6]) - (gama*x[244])/SQR(length[6]);
    dx_dt[245] = (kappa*x[101])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[117])/SQR(length[6]) + (kappa*x[133])/SQR(length[6]) - (gama*x[245])/SQR(length[6]);
    dx_dt[246] = (kappa*x[102])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[118])/SQR(length[6]) + (kappa*x[134])/SQR(length[6]) - (gama*x[246])/SQR(length[6]);
    dx_dt[247] = (kappa*x[103])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[119])/SQR(length[6]) + (kappa*x[135])/SQR(length[6]) - (gama*x[247])/SQR(length[6]);
    dx_dt[248] = (kappa*x[104])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[120])/SQR(length[6]) + (kappa*x[136])/SQR(length[6]) - (gama*x[248])/SQR(length[6]);
    dx_dt[249] = (kappa*x[105])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[121])/SQR(length[6]) + (kappa*x[137])/SQR(length[6]) - (gama*x[249])/SQR(length[6]);
    dx_dt[250] = (kappa*x[106])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[122])/SQR(length[6]) + (kappa*x[138])/SQR(length[6]) - (gama*x[250])/SQR(length[6]);
    dx_dt[251] = (kappa*x[107])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[123])/SQR(length[6]) + (kappa*x[139])/SQR(length[6]) - (gama*x[251])/SQR(length[6]);
    dx_dt[252] = (kappa*x[108])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[124])/SQR(length[6]) + (kappa*x[140])/SQR(length[6]) - (gama*x[252])/SQR(length[6]);
    dx_dt[253] = (kappa*x[109])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[125])/SQR(length[6]) + (kappa*x[141])/SQR(length[6]) - (gama*x[253])/SQR(length[6]);
    dx_dt[254] = (kappa*x[110])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[126])/SQR(length[6]) + (kappa*x[142])/SQR(length[6]) - (gama*x[254])/SQR(length[6]);
    dx_dt[255] = (kappa*x[111])/SQR(length[6]) + ((-2*kappa - cos(x[6])*length[6])*x[127])/SQR(length[6]) + (kappa*x[143])/SQR(length[6]) - (gama*x[255])/SQR(length[6]);
    dx_dt[256] = (kappa*x[112])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[128])/SQR(length[7]) - (gama*x[256])/SQR(length[7]);
    dx_dt[257] = (kappa*x[113])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[129])/SQR(length[7]) - (gama*x[257])/SQR(length[7]);
    dx_dt[258] = (kappa*x[114])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[130])/SQR(length[7]) - (gama*x[258])/SQR(length[7]);
    dx_dt[259] = (kappa*x[115])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[131])/SQR(length[7]) - (gama*x[259])/SQR(length[7]);
    dx_dt[260] = (kappa*x[116])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[132])/SQR(length[7]) - (gama*x[260])/SQR(length[7]);
    dx_dt[261] = (kappa*x[117])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[133])/SQR(length[7]) - (gama*x[261])/SQR(length[7]);
    dx_dt[262] = (kappa*x[118])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[134])/SQR(length[7]) - (gama*x[262])/SQR(length[7]);
    dx_dt[263] = (kappa*x[119])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[135])/SQR(length[7]) - (gama*x[263])/SQR(length[7]);
    dx_dt[264] = (kappa*x[120])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[136])/SQR(length[7]) - (gama*x[264])/SQR(length[7]);
    dx_dt[265] = (kappa*x[121])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[137])/SQR(length[7]) - (gama*x[265])/SQR(length[7]);
    dx_dt[266] = (kappa*x[122])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[138])/SQR(length[7]) - (gama*x[266])/SQR(length[7]);
    dx_dt[267] = (kappa*x[123])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[139])/SQR(length[7]) - (gama*x[267])/SQR(length[7]);
    dx_dt[268] = (kappa*x[124])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[140])/SQR(length[7]) - (gama*x[268])/SQR(length[7]);
    dx_dt[269] = (kappa*x[125])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[141])/SQR(length[7]) - (gama*x[269])/SQR(length[7]);
    dx_dt[270] = (kappa*x[126])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[142])/SQR(length[7]) - (gama*x[270])/SQR(length[7]);
    dx_dt[271] = (kappa*x[127])/SQR(length[7]) + ((-kappa - cos(x[7])*length[7])*x[143])/SQR(length[7]) - (gama*x[271])/SQR(length[7]);
}//Derivatives
