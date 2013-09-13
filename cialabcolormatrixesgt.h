/*
 * FILE: 
 *      cialabcolormatrixesgt.h
 *
 * DESCRIPTION: 
 *   Contains groundtruth cialab color matrices. The matrixes have been
 *   obtained from matlab running the code:
 *  
 *       mser    =   Blobs(i); 
 *       A = permute(mser.pvec, [3,2,1]);
 *       C = applycform(A, reg);
 *       colour = permute(C,[3,2,1]);
 *                       
 *       dataf(i).Mmvec=mser.mvec;
 *       dataf(i).Mpvec=colour;  
 *      
 *
 * FUNCTIONS:
 *
 * PROJECT:
 *
 * AUTHORs: 
 *
 * VERSION:
 *      $Id$
 *
 * COYRIGHT:
 *
 * REVISIONS:
 *      $Log$
 */


float sz3[117] = {42.8835,41.2296,31.5148,7.40785,34.8764,19.4293,39.6036,
    22.2059,25.4845,3.9721,11.1227,1.99988,29.728,17.9073,23.2586,2.59575,
    92.7045,86.4152,59.5178,90.6344,86.5134,65.3641,90.5365,79.9364,
    66.4876,59.9459,74.3542,70.0302,81.2998,82.1976,78.8734,84.7675,
    70.3267,69.414,82.1393,61.0694,25.2495,10.7396,52.6226,4.30302,
    4.0287,4.92453,1.4618,5.84931,2.63787,4.22794,2.93456,3.27194,0.328944,
    1.56302,0.142002,3.35592,2.72569,3.74784,0.253781,-2.54691,-1.62522,
    -1.4187,0.00297509,0.676003,0.346658,-1.20997,-0.805009,-1.35577,
    -1.26141,-2.25515,-2.22048,-2.52369,-1.59301,-2.3762,-1.88595,-1.98041,
    -1.63147,1.95503,-2.53411,0.456124,1.98686,2.12798,-17.5312,-18.7703,
    -16.2733,-5.5646,-17.3324,-9.65772,-16.3233,-10.7847,-11.9189,-1.51849,
    -6.41145,-0.886111,-13.0217,-9.73903,-12.4455,-1.11123,-6.89153,
    -5.95073,0.590969,-13.1492,-12.4474,-0.485313,-9.43125,-4.36364,
    0.118441,0.671609,2.42568,1.36099,2.74826,0.751823,-0.990216,-5.20625,
    2.237,-2.12269,-9.10061,5.02156,-6.1492,-7.05127,-6.55639};

float sz4[162] ={51.2324,53.385,54.4504,21.0217,32.9396,49.1442,32.4404,
    32.627,48.3246,30.184,30.8174,4.8461,29.1443,28.4019,12.411,10.2674,
    4.61528,21.9761,21.0217,98.2882,81.3489,97.0114,68.8964,21.4495,82.335,
    81.7032,77.39,67.9638,83.4551,34.8912,34.7386,79.8427,76.9917,3.7488,
    80.0208,30.293,71.4403,92.6932,12.1101,16.2378,9.51714,18.2741,56.4197,
    69.1554,69.4742,35.7256,76.9209,35.797,22.3916,1.91399,40.9581,1.82883,
    3.80353,55.3924,7.45864,7.82587,9.64163,3.82259,10.8261,7.77103,
    9.38009,9.21383,7.71493,8.73014,8.52115,0.714725,7.92717,7.36064,
    3.46968,2.08273,0.719151,3.88444,3.82259,-1.37819,0.124539,-1.14517,
    -1.20177,7.11149,0.862819,13.8204,0.325999,4.76818,3.23305,14.0684,
    12.6192,2.19714,2.59695,0.527884,13.016,6.92531,-3.45366,4.14324,
    3.47535,4.09084,2.35484,4.37665,8.24872,10.2791,9.86154,6.04568,
    -3.01044,11.5214,9.89577,0.603889,6.33514,0.0882578,0.778662,2.49099,
    4.99866,6.25025,2.93187,-9.64527,-23.2062,-1.54654,-18.5003,-20.0016,
    -2.75619,-16.1347,-16.3013,-1.3386,-17.7145,-15.6832,-4.80546,-5.39332,
    -1.73403,-5.11109,-9.64527,-0.497241,4.57995,-0.609304,8.98506,
    -8.91866,1.52207,9.25154,3.5213,-0.45931,-8.76793,-25.2756,-20.147,
    -2.52644,-0.280231,-0.637934,7.09133,-14.7841,9.15879,-9.07075,-5.9856,
    -7.61763,-5.36277,-8.77967,2.14544,2.73502,3.34753,-4.07567,7.59405,
    -14.8848,-14.6353,-0.606457,-10.8641,0.17571,-1.20265,-1.61964};