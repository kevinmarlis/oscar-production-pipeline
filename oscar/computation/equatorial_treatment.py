from .constants import *
from .physics import *
import numpy as np
import numpy.matlib
import math
import scipy as sp
import warnings


def do_equatorial_treatment(U1g, U1Wh, U1Bh, um,vm, do_eq, F1_GrH, F1_TAU0, F1_GrT, U1_GrH, U1_TAU0, U1_GrT):
    if do_eq:
        LATRANGE=5
        LATRANGE=LATRANGE*np.array([-1,1])*math.pi/180; # (rad) DOMAIN OF SOLUTION WITH ORTH. POLYNOMIALS
        LATRANGE=LATRANGE/PHIm
        ynl=np.max(LATRANGE)
        ysl=np.min(LATRANGE)
        # this just assigns F1 to F2

        F2_GrH = F1_GrH
        F2_TAU0 = F1_TAU0
        F2_GrT = F1_GrT


        NORDEREF = 5
        U2g,U2Wh,U2Bh = EQ(y,U1_GrH,ysl,ynl,U1_TAU0,U1_GrT,f,NORDEREF,F1_GrH,F2_GrH,F1_TAU0,F2_TAU0,F1_GrT,F2_GrT,eta)

        UgC,UwC,UbC = combu1u2(y,ysl,ynl,U1g,U1Wh,U1Bh,U2g,U2Wh,U2Bh)
        Ug,Uw,Ub = rescale_velocities(UgC, UwC, UbC, um,  vm)
        Ug, Uw, Ub = remove_spurious(Ug, Uw, Ub)

        return Ug, Uw, Ub
        
    Ug, Uw, Ub = rescale_velocities(U1g, U1Wh, U1Bh, um,vm)
    Ug, Uw, Ub = remove_spurious(Ug, Uw, Ub)

    return Ug, Uw, Ub


def EQ(y,U1_GrH,ysl,ynl,U1_TAU0,U1_GrT,f,NORDEREF,F1_GrH,F2_GrH,F1_TAU0,F2_TAU0,F1_GrT,F2_GrT,eta):

    refDa = F1_GrH*0.0
    #**************************************************
    #*********** U1 DERIVATIVES ***********************
    #**************************************************

    [U1_GrHN,dU1_GrHN,d2U1_GrHN,\
     U1_GrHS,dU1_GrHS,d2U1_GrHS]\
    =uderivatives(y,U1_GrH,ysl,ynl)

    [U1_TAU0N,dU1_TAU0N,d2U1_TAU0N,\
     U1_TAU0S,dU1_TAU0S,d2U1_TAU0S]\
    =uderivatives(y,U1_TAU0,ysl,ynl)
    
    [U1_GrTN,dU1_GrTN,d2U1_GrTN,\
    U1_GrTS,dU1_GrTS,d2U1_GrTS]\
    =uderivatives(y,U1_GrT,ysl,ynl)
    
    #**************************************************
    #*********** GENERAL POLYNOMIALS ******************
    #**************************************************
    
    P=polynomials(NORDEREF+1,ysl,ynl,f)
    #NOTE: HERE SUBA.POLYNOMIALS WILL RETURN NORDEREF+1 ORTHOG POLYNOMIALS, AND
    #THE MAXIMUM POLYNOMIAL DEGREE IS NORDEREF
    
    #**************************************************
    #*********** GENERAL B MATRICES *******************
    #**************************************************
    [B_SOUTH,B_NORTH]=B(NORDEREF+1,ysl,ynl)
    #NOTE: HERE THE TRANSITION POLYNOMIAL DEGREE IS NORDEREF+1
   
    #**************************************************
    #******* LOOP OVER THE THREE FORCINGS**************
    #**************************************************
    
    for Fnum in range(3):
        if Fnum==0:
             F1=F1_GrH
             F20=F2_GrH
             UN=U1_GrHN
             US=U1_GrHS
             dUN=dU1_GrHN
             dUS=dU1_GrHS
             d2UN=d2U1_GrHN
             d2US=d2U1_GrHS
             txt='SSH GRADIENT'
        elif Fnum==1:
            F1=F1_TAU0
            F20=F2_TAU0
            UN=U1_TAU0N
            US=U1_TAU0S
            dUN=dU1_TAU0N
            dUS=dU1_TAU0S
            d2UN=d2U1_TAU0N
            d2US=d2U1_TAU0S
            txt='WIND'
        elif Fnum==2:
            F1=F1_GrT
            F20=F2_GrT
            UN=U1_GrTN
            US=U1_GrTS
            dUN=dU1_GrTN
            dUS=dU1_GrTS
            d2UN=d2U1_GrTN
            d2US=d2U1_GrTS
            txt='SST GRADIENT'
        
        #**************************************************
        #************ ALLOCATE SPACE  *********************
        #**************************************************
        U2 = (np.nan + 1j*np.nan) * np.ones(F1.shape)
        #**************************************************
        #************ REDUCE MATRIX TO LAT RANGE  *********
        #**************************************************
        
        #* Reduce latitude range to just include [ysl,ynl]
        #* taking into account ysl,ynl not on grid points
        dy=min(abs(y[1:]-y[0:-1]))
        i2=np.where(np.all([ysl-dy+np.finfo(float).eps<y, y<ynl+dy-np.finfo(float).eps], axis=0))[0]
        y2=y[i2]
        F2=F20[:,i2,:]

        #**************************************************
        #************ STARTS THE TIME LOOP ****************
        #**************************************************
        
        for t in range(F2.shape[0]):
            
            #**************************************************
            #************ FIND REGULAR LONGITUDES ****************
            #**************************************************

            # CAN WE we assume real and imag components are 100% correlated?
            REGPT,IRREGPT=find_regular(y2, F2[t,:,:], UN[t,:], US[t,:])

            #**************************************************
            #********* PRE-TREATMENT OF THE MATRIX ************
            #**************************************************
            # CAN WE we assume real and imag components are 100% correlated for nan? - or de we need separate x and y REGPT, IRREGPT?
            if REGPT.size > 0:
                gridX,gridY = np.meshgrid(REGPT,y2)
                
                xt=np.matlib.repmat(REGPT,y2.size,1)
                yt=np.matlib.repmat(y2.reshape(y2.size,1),1,REGPT.size)
                # keeping Ftx,Fty as DS, results in messed up coordinates; just work with np arrays
                Ft=F2.values[t,:,REGPT].transpose()
                #* Fill in single NaNs
                mask = np.where(~np.isnan(Ft))
                points = np.array(list(zip(xt[mask], yt[mask])))
                values = Ft[mask]
                Ft=sp.interpolate.griddata(points, values, (gridX,gridY), method='linear')

                alpha=ALPHA(y2,Ft,P,f,ysl,ynl)
                count=0
                #for each longitude because of polyvals
                for ii in REGPT:
                    #**************************************************
                    #************ CALCULATE a0,a1,a2 ******************
                    #**************************************************
                    CN=np.array([UN[t,ii],dUN[t,ii],d2UN[t,ii]])
                    aNORTH= a(B_NORTH, CN, ynl, alpha[:,count], P)
                    CS=np.array([US[t,ii],dUS[t,ii],d2US[t,ii]])
                    aSOUTH= a(B_SOUTH, CS, ysl, alpha[:,count], P)
                    #**************************************************
                    #************ CALCULATE U2 ************************
                    #**************************************************
                    #Note: NORDEREF+1 is transition polynomial degree
                    # NORTH PART
                    iN = np.where(y2>=0)[0]
                    yN = y2[iN]
                    alphaP = np.matmul(alpha[:,count],P)
                    U2[t,i2[iN],ii] = np.polyval(alphaP,yN) + \
                        yN**(NORDEREF+1)*(aNORTH[0] + aNORTH[1]*yN + aNORTH[2]*yN**2)
                    # SOUTH PART
                    iS = np.where(y2<0)
                    yS = y2[iS]
                    U2[t,i2[iS],ii] = np.polyval(alphaP,yS) + \
                        yS**(NORDEREF+1)*(aSOUTH[0] + aSOUTH[1]*yS + aSOUTH[2]*yS**2)

                    count+=1

#             #**************************************************
#             #******* SPECIAL CASE LONGITUDES ******************
#             #**************************************************
            for ii in IRREGPT:
                #**************************************************
                #************ FIND THE PIECES *********************
                #**************************************************
                # Valid pieces are longer than 2, do not include NaN
                # endpoints, and can include isolated NaNs.
                IINF,ISUP=find_pieces(F2[t,:,ii])
                nP=len(IINF)

                #**************************************************
                #************ FOR EACH PIECE **********************
                #**************************************************
                for k in range(nP):
                    ik=np.array(list(range(IINF[k],ISUP[k]+1)))
                    nik=len(ik)
                    yk=np.array(y2[ik])
                    yn=np.max(yk)
                    ys=np.min(yk)
                    Fk=F2[t,ik,ii]
                    Fk = np.expand_dims(Fk, axis=1)
                    
                    #***********************************************
                    #****** REDO ORTHOGONAL POLYNOMIALS ON PIECE ***
                    #***********************************************
                    
                    Pk=polynomials(NORDEREF+1,ys,yn,f)
                    #Note: maximum orthog polynomial degree is NORDEREF, and
                    #number of polynomials is NORDEREF+1

                    #************************************************
                    #*********** B MATRICES *************************
                    #************************************************
                    [Bk_SOUTH,Bk_NORTH]=B(NORDEREF+1,ys,yn)
                    #Note: transition polynomial degree is NORDEREF+1
                    
                    #************************************************
                    #************ CALCULATE ALPHAS ******************
                    #************************************************
                    # last Fk dim is size 1
                    iknan=np.where(~np.isnan(Fk[:,0]))[0]
                    # squeeze to transfrom (6,1) to (6)
                    alphak=ALPHA(yk[iknan],Fk[iknan,:],Pk,f,ys,yn).squeeze()
                    
                    #************************************************
                    #************ CALCULATE U2 base *****************
                    #************************************************
                    U2[t,i2[ik],ii]=np.polyval(np.matmul(alphak,Pk),yk)

                    #### IF PIECE INCLUDES NORTH ENDPOINT
                    if yn>=ynl-1e-3:
                        #********************************************
                        #********* CALCULATE a0,a1,a2 ***************
                        #********************************************
                        CN=np.array([UN[t,ii], dUN[t,ii], d2UN[t,ii]])
                        akNORTH= a(Bk_NORTH,CN,ynl,alphak,Pk)
                        #********************************************
                        #********* CALCULATE U2 addition ************
                        #********************************************
                        ikN=np.where(yk>=0)[0]
                        yyk=yk[ikN]
                        U2[t,i2[ik[ikN]],ii] += \
                            yyk**(NORDEREF+1)*(akNORTH[0] + akNORTH[1]*yyk + akNORTH[2]*yyk**2)
                        #Note: transition polynomial degree is NORDEREF+1
                        
                    #### IF PIECE INCLUDES SOUTH ENDPOINT
                    if ys<=ysl+1e-3:
                        #********************************************
                        #********* CALCULATE a0,a1,a2 ***************
                        #********************************************
                        CS=np.array([US[t,ii], dUS[t,ii], d2US[t,ii]])
                        akSOUTH= a(Bk_SOUTH,CS,ysl,alphak,Pk)
                        #********************************************
                        #********* CALCULATE U2 addition ************
                        #********************************************
                        ikS=np.where(yk<=0)[0]
                        yyk=yk[ikS]
                        U2[t,i2[ik[ikS]],ii] += \
                            yyk**(NORDEREF+1)*(akSOUTH[0] + akSOUTH[1]*yyk + akSOUTH[2]*yyk**2)
                        #Note: transition polynomial degree is NORDEREF+1
                # end k
            # end IRREGPT
        # end time
        if Fnum==0:
            U2_GrH=U2
        elif Fnum==1:
            U2_TAU0=U2
        elif Fnum==2:
            U2_GrT=U2
        
#     end #for each forcing
    
#     #**************************************************
#     #************ COMBINE x,y COMPONENTS **************
#     #**************************************************
    
    # SSH-GRADIENT DERIVED VELOCITIES
    U2g = U2_GrH.real + 1j*U2_GrH.imag/eta
    # SURFACE STRESS DERIVED VELOCITY
    U2Wh = U2_TAU0.real +1j*U2_TAU0.imag/eta
     # TEMPERATURE-GRADIENT DERIVED VELOCITIES
    U2Bh = U2_GrT.real + 1j*U2_GrT.imag/eta

    return U2g,U2Wh,U2Bh


def uderivatives(y,U,ys,yn):

    a=U.shape

    yy=np.tile(y,(a[0],a[2],1)) #time,x,y
    yy=np.swapaxes(yy,1,2) #time,y,x
    
    # North
    Uxn,Uxyn,Uxyyn=uderivatives_sub(y,yn,yy,U.real)
    Uyn,Uyyn,Uyyyn=uderivatives_sub(y,yn,yy,U.imag)
 
    # South
    Uxs,Uxys,Uxyys=uderivatives_sub(y,ys,yy,U.real)
    Uys,Uyys,Uyyys=uderivatives_sub(y,ys,yy,U.imag)

    return Uxn + 1j*Uyn,Uxyn + 1j*Uyyn,Uxyyn + 1j*Uyyyn,Uxs + 1j* Uys,Uxys + 1j*Uyys,Uxyys + 1j*Uyyys


def uderivatives_sub(y,yn,yy,U):
    
    a = np.absolute(y-yn)
    index = a.argmin()
    yy=np.where(np.isnan(U),0.0,yy)
    U=np.where(np.isnan(U),0.0,U)
    yyn=yy[:,index-1:index+2,:]-yn
    UUN=U[:,index-1:index+2,:]

    DET=(yyn[:,0,:]-yyn[:,1,:])*(yyn[:,0,:]-yyn[:,2,:])*(yyn[:,1,:]-yyn[:,2,:])
    # if any(DET(:)==0), error('Two latitudes cannot be equal'), end
    Un=(UUN[:,2,:]*yyn[:,0,:]*yyn[:,1,:]*(yyn[:,0,:]-yyn[:,1,:])+ \
        UUN[:,0,:]*yyn[:,1,:]*yyn[:,2,:]*(yyn[:,1,:]-yyn[:,2,:])+ \
        UUN[:,1,:]*yyn[:,2,:]*yyn[:,0,:]*(yyn[:,2,:]-yyn[:,0,:]))/DET
    Uyn=(UUN[:,2,:]*(yyn[:,1,:]**2-yyn[:,0,:]**2)+ \
        UUN[:,1,:]*(yyn[:,0,:]**2-yyn[:,2,:]**2)+ \
        UUN[:,0,:]*(yyn[:,2,:]**2-yyn[:,1,:]**2))/DET
    Uyyn=2*(UUN[:,2,:]*(yyn[:,0,:]-yyn[:,1,:])+ \
        UUN[:,0,:]*(yyn[:,1,:]-yyn[:,2,:])+ \
        UUN[:,1,:]*(yyn[:,2,:]-yyn[:,0,:]))/DET
    
    return Un,Uyn,Uyyn


def polynomials(n,a,b,KER):

    #n FIRST ORTHOGONAL POLYNOMIALS DEFINED BY THE KERNEL FUNCTION KER(Y).
    #HERE KER(Y) IS A POLYNOMIAL GIVEN IN THE MATLAB FORMAT.
    #NOTE: THE MAXIMUM DEGREE NUMBER IS n-1
    #EXAMPLES:
    #KER(Y)=1=[1]                   -> LEGENDRE POLYNOMIALS
    #KER(Y)=Y=[1,0]                 -> EQUATORIAL BETA-PLANE ORTHOGONAL POLYNOMIAL
    #KER(Y)=Y-(1/6)Y^3+... ~ SIN(Y) -> GENERAL ORTHOGONAL POLYNOMIALS FOR CORIOLIS PB

    E=np.fliplr(np.eye(n))
    S=np.convolve(KER,np.conj(KER))

    #Below, Q is a row vector containing the first 2*n-1 values of
    #the integrals int(x^k*abs(KER)^2,x,a,b), k=0,..,2*n-2 that are required
    #to calculate scalar products
    Q=np.zeros(2*n-1)
    for k in range(2*n-2+1):
        Pk=np.zeros(k+1)
        Pk[0]=1
        #POLYNOMIAL INTEGRATION
        PRODk=np.convolve(Pk,S)
        rk=np.arange(PRODk.size-1,-1,-1)
        Ik=np.ndarray.sum(PRODk*(b**(rk+1)-a**(rk+1))/(rk+1))
        Q[k]=Ik
        
    PHI=np.zeros([n,n])
    NORI=sub_scalar_product(E[0,:],E[0,:],Q)
    NORI=np.sqrt(NORI)

    PHI[0,:]=E[0,:]/NORI

    
    for k in range(1,n):
        EEk=np.matlib.repmat(E[k,:],k,1)
        Bk=sub_scalar_product(EEk,PHI[0:k,:],Q)
        NORk=sub_scalar_product(E[k,:],E[k,:],Q)
        NORk=np.sqrt(NORk)
        NorSqIn = NORk**2-sum(Bk**2)
        if NorSqIn < 0:
            NorSqIn = 0.0
        Ckk=1.0/np.sqrt(NorSqIn)
        BBk=np.matlib.repmat(Bk,1,n)
        PHI[k,:]=Ckk*(E[k,:]-sum(BBk*PHI[0:k,:],0))

    return PHI


def sub_scalar_product(P1,P2,Q):

    #scalar product for polynomials
    #P1 and P2 are polynomials given by raw vectors (there can be several polynomials)
    #Q contains previously calculated integrals

    #In python 1-D arrays are not subsetted the same way as 2-D
    
    # N is the number of different polynomials, stacked in rows
    #Start with 1-D
    if P1.ndim ==1:
        M=len(P1)+len(P2)-1
        N=1
        Q=Q[0:M+1]

        A=np.convolve(P1,np.conjugate(P2))
        A=np.flipud(A)
        SP=np.sum(Q*A)
  
    #Now do 2-D
    if P1.ndim ==2:
        M=P1.shape[1]+P2.shape[1]-1
        N=P1.shape[0]
        Q=Q[...,0:M]
        SP=np.zeros((N,1))+np.nan
        
        for k in range(N):
            A=np.convolve(P1[k,:],np.conjugate(P2[k,:]))
            A=np.flip(A,0)
            SP[k,0]=np.sum(Q*A,axis=0)
    
    return SP


def find_regular(y2,F2,UN,US):
    #**************************************************
    #************ FIND REGULAR LONGITUDES ****************
    #**************************************************
    #
    #y2=column restricted to enough pts to just include [ysl,ynl]
    #F2=forcing at time t to just include [ysl,ynl]
    #UN=U1 at ynl at time t as calculated in SUB_UDERIVATIVES
    #US=U1 at ysl at time t as calculated in SUB_UDERIVATIVES
    #
    # REGPT= longitudes where both ynl and ysl exist in the
    # forcing and there are no islands = two contiguous NaNs
    # IRREGPT = all leftover points that are still
    # not all land points that need velocity calculations

    # Bad points:
    # Columns with islands or no north point or no south point

    BAD = np.where(np.any([np.all([np.any(np.isnan(F2[1:,:]), axis=0), np.any(np.isnan(F2[:-2,:]), axis=0)], axis=0),
        np.isnan(UN), np.isnan(US), np.isnan(F2[0,:]), np.isnan(F2[-1,:])], axis=0), True, False)
    BNL = np.where(np.all([BAD, np.any(~ np.isnan(F2), axis=0)], axis=0), True, False)
    REGPT=np.where(~ BAD)[0]
    IRREGPT=np.where(BNL)[0]
    return REGPT,IRREGPT


def ALPHA(y,FORCE,POL,f,ysl,ynl):
    # where DNX=longitude points
    #       size(ALPHA)=[#polynomials,DNX]
    #          i.e. each column contains the coefficients for
    #          poly k at one longitude
    #       FORCE = F2(y,DNX,t)
    #          i.e. size(FORCE) =[length(y),DNX]
    #          = force in LAT-LONG at a single time
    #       POL=[#polynomials,order of polynomial]
    #          i.e. each row contains the poly k
    # #***** PARAMETERS
    TOL=1e-2

    # #***** REDUCING LATITUDE RANGE TAKING INTO ACCOUNT ynl,ysl
    # #***** NOT ON GRID POINTS
    dy=np.min(np.abs(y[1:]-y[0:-1]));
    IRED=np.where(np.all([ysl-dy+1e-05<y, y<ynl+dy-1e-05], axis=0))[0]

    y=y[IRED]
    FORCE=FORCE[IRED,:]

    # #***** CHECK N >= 2
    # NUMBER OF LATITUDES
    N=y.size
    if N<2:
         raise Exception('NUMBER OF LATITUDES IS LESS THAN 2')

    NX=FORCE.shape[1]
    #***** ORDER: NUMBER OF POLYNOMIALS
    NP=POL.shape[0]

    # #** INITIALIZATION
    ALPHA=np.nan*(np.ones([NP,NX]) + 1j*np.ones([NP,NX]))

    # #***** CALCULATE COEFFICIENTS OF POLYNOMIAL DECOMPOSITION
    # #***** OF FORCE ONTO POL AT ALL LONGITUDES
    for k in range(NP):
          # For each polynomial k, calculate the coefficient alpha_k
          KSP=np.convolve(f,POL[k,:])
          # integration is only processing real part of complex number, so separate them
          Areal=vector_integer_p(y,np.real(FORCE),KSP,ysl,ynl,TOL)
          Aimag=vector_integer_p(y,np.imag(FORCE),KSP,ysl,ynl,TOL)
          A = Areal + 1j*Aimag
          ALPHA[k,:]=-1j* A
  
    return ALPHA


def vector_integer_p(D,FUNC,P,y0,yF,TOL):
    #function I=SUB_VECTINTEG_P(D,FUNC,P,y0,yF,TOL)
    #
    #% ONLY FUNC IS VECTORIZED AS AN INPUT (NOT D): INTEGRATION IS DONE ALONG THE
    #% COLUMNS (!) FOR EACH COLUMN, SO THAT THE OUTPUT I IS A LINE VECTOR.
    #% HOWEVER THE SUBROUTINE SUB_VECTQUAD8 AND FUNCTION SUB_VECTDATAFUNC_P ARE DESIGNED
    #% TO INTEGRATE ALONG LINES, THEREFORE TWO RESHAPINGS MUST BE DONE (D AND FUNC,
    #% AND THEN I). THAT CAN BE CHANGED BY MODIFYING SUB_VECTQUAD8 AND SUB_VECTDATAFUNC_P, IF
    #% THERE IS NEED OF OPTIMIZING THE ALGORITHM.
    #
    #%**** CHECKS THE INTERVAL OF INTEGRATION
    #TEST=y0>=min(D) & y0<=max(D) & yF>=min(D) & yF<=max(D);
    #%if ~TEST
    #%  error('Out of range interval')
    #%end
    # mv spline of of VECTDATAFUNC_P up
    (NFUNC,MFUNC)=FUNC.shape
    # in matlab, MFUNC is the lon dim
    # in matlab, the y input to VECTDATAFUNC_P is a vector, but in python it is a scalar
    lonarr = 1.0*np.arange(0, MFUNC)
    if MFUNC >1:
        finterp = sp.interpolate.RectBivariateSpline(D, lonarr, FUNC)
        result, err = sp.integrate.quad_vec(vector_data_func_p, y0, yF, epsrel = TOL, args=(D,lonarr,finterp,P))
    else:

        finterp = sp.interpolate.CubicSpline(D, FUNC)

        with warnings.catch_warnings():
            # IntegrationWarning from quad() and/or AccuracyWarning from quadrature()
            warnings.simplefilter("ignore", sp.integrate.IntegrationWarning)
            #warnings.simplefilter("ignore", sp.integrate.AccuracyWarning)

            # using quad, with our epsrel = TOL (0.01) we get multiple occurances of
            #      IntegrationWarning: The occurrence of roundoff error is detected, which prevents the requested tolerance from being achieved.  The error may be underestimated.
            result, err = sp.integrate.quad(vector_data_func_p, y0, yF, epsrel = TOL, args=(D,lonarr,finterp,P))

            # using quadrature() below has slightly more diffs wrt to matlab (runtime about the same), we get these warning:
            # AccuracyWarning: maxiter (50) exceeded. Latest difference = 7.475699e-05 output than quad
            #result, err = sp.integrate.quadrature(VECTDATAFUNC_P, y0, yF, rtol = TOL, args=(D,lonarr,finterp,P), vec_func=False)

            # quad_vec below doesn't complete unless workers = 1, in which case it runs 70% slowere than quad(), but without warning messages:
            # with workers = -1, there are lo of python processes, but somehow it doesn't stop running!
            #result, err = sp.integrate.quad_vec(VECTDATAFUNC_P, y0, yF, epsrel = TOL, args=(D,lonarr,finterp,P), workers=-1, limit=50)

    return result


def vector_data_func_p(y,D,lonarr,finterp,P):
    #function F=SUB_VECTDATAFUNC_P(y,D,FUNC,P)
    #% THE SUB_VECTDATAFUNC_P ROUTINE CALCULATES THE PRODUCT OF THE FUNCTION GIVEN
    #% BY FUNC TIMES THE POLYNOMIAL P. FUNC IS A DATA COLUMN VECTOR CORRESPONDING TO THE
    #% DOMAIN D WHICH IS ALSO A COLUMN VECTOR, AND ITS ESTIMATE VALUE AT ANY POINT y
    #% IS OBTAINED BY INTERPOLATION. P IS A COMPLETELY DEFINED POLYNOMIAL FUNCTION WHICH
    #% VALUE AT y IS KNOWN WITH MAXIMUM PRECISION. THIS IS THE REASON WHY THIS
    #% DECOMPOSITION IS RELEVANT, AND THIS SUBROUTINE IS CALLED MANY TIMES BY THE
    #% SUBROUTINE SUB_VECTQUAD8 IN THE INTEGRATION PROCEDURE.
    #
    #if MFUNC>1;
    #    F1=interp2(D,(1:MFUNC)',FUNC,y,repmat(Q,[1,Ny]),'spline');
    #    %REMARK1: SPLINE INTERPOLATION IS USED, WHICH MAY TAKE MORE TIME THAN CUBIC.
    #    %IT COULD BE WORTHWHILE TESTING IT WITH 'CUBIC'. 'LINEAR' IS ABSOLUTELY NOT
    #    %RECOMMENDED IN THIS CONTEXT.
    
    #FUNCshape=(41, 814)  is lat-first, then lon
    if lonarr.size > 1:
        F1=finterp(y, lonarr)
    else:
        F1=finterp(y)
    F2 = np.polyval(P,y)
    result = F1*F2
    return result


def find_pieces(X):
    # Find contiguous non-NaN pieces in X allowing
    # pieces to have isolated single NaNs.
    # Pieces shorter than 2 in length are rejected.
    # Idea: islands have 2 or more NaNs in a row.
    # The rest are interpreted as missed data, not land.

    INF=[]
    SUP=[]
    if np.any(~np.isnan(X)):
        nX=X.size
        I=np.where(~np.isnan(X))[0]
        nI=I.size
        if nI>1:
            TEST=I[1:]-I[:-1]
            J = np.where(TEST>2)[0]
            J = np.append(J,nI-1)
            SUP=I[J]
            X=np.flip(X)
            I=np.where(~np.isnan(X))[0]
            nI=I.size
            TEST=I[1:]-I[:-1]
            J = np.where(TEST>2)[0]
            J = np.append(J,nI-1)
            INF = np.flip(nX-I[J]-1);
            #remove pieces that are less than 2 in length
            J = np.where(SUP-INF<=1)[0]
            SUP = np.delete(SUP, J)
            INF = np.delete(INF, J)
    return INF,SUP


def a(B,C,yi,alpha,P):
    #  calculate A=[A0;A1;A2] from alpha,P,yi
    #     where alpha=column vector of coefficients of
    #           the polynomials in each row of P
    #  generate C from U1 and derivatives at y=yi
    #  calculate a=[a0;a1;a2] from B\(C-A)

    alphaP = np.matmul(alpha,P)
    A0=np.polyval(alphaP,yi)
    A1=np.polyval(np.polyder(alphaP),yi)
    A2=np.polyval(np.polyder(np.polyder(alphaP)),yi)

    A=[A0,A1,A2]
    a=np.linalg.solve(B,C-A)
    return a


def B(N,ys,yn):

    # # Calcuate the B matrix needed in the solution for U2
    # # Note: N is the transition polynomial degree
    B_SOUTH=[[ys**N,ys**(N+1),ys**(N+2)],
        [N*ys**(N-1), (N+1)*ys**N, (N+2)*ys**(N+1)],
        [(N-1)*N*ys**(N-2), N*(N+1)*ys**(N-1), (N+1)*(N+2)*ys**N]]

    B_NORTH=[[yn**N,yn**(N+1),yn**(N+2)],
        [N*yn**(N-1), (N+1)*yn**N, (N+2)*yn**(N+1)],
        [(N-1)*N*yn**(N-2), N*(N+1)*yn**(N-1), (N+1)*(N+2)*yn**N]]

    return B_SOUTH,B_NORTH


def remove_spurious(Ug: np.ndarray, Uw: np.ndarray, Ub: np.ndarray, threshold: float = 3.0):
    """
    Removes spurious (unrealistically large) values from ocean current components.
    
    Any value with magnitude > `threshold` in an individual component or in the combined sum
    is replaced with NaN + 1j * NaN.
    
    Parameters:
        Ug (np.ndarray): Geostrophic current (complex array)
        Uw (np.ndarray): Wind-driven current (complex array)
        Ub (np.ndarray): Buoyancy-driven current (complex array)
        threshold (float): Maximum allowed magnitude before values are marked spurious
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Cleaned versions of Ug, Uw, and Ub
    """
    # Create mask for each component and for combined magnitude
    valid_mask = (
        (np.abs(Ug) <= threshold) &
        (np.abs(Uw) <= threshold) &
        (np.abs(Ub) <= threshold) &
        (np.abs(Ug + Uw + Ub) <= threshold)
    )
    
    # Apply mask
    Ug_clean = np.where(valid_mask, Ug, np.nan + 1j * np.nan)
    Uw_clean = np.where(valid_mask, Uw, np.nan + 1j * np.nan)
    Ub_clean = np.where(valid_mask, Ub, np.nan + 1j * np.nan)

    return Ug_clean, Uw_clean, Ub_clean