/*
 * FILE: 
 *  mscrdist.cpp
 *
 * DESCRIPTION: 
 *  MSCR distances mapping project.     
 *
 * PROJECT:
 *  ReId Demo
 *
 * AUTHORs: 
 *  Carlos Beltran-Gonzalez.
 *
 * VERSION:
 *      $Id$
 *
 * COYRIGHT:
 *      Copyright (c) 2013 Istituto Italiano di Tecnologia. Genova
 *
 * REVISIONS:
 *      $Log$
 */

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iterator>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

float invgamma(float Rp) {
    float R = 0.0;
    if ( Rp <= 0.03928 )
        R = Rp/12.92;
    else
        R = pow(((Rp + 0.055)/1.055),2.4);
    return R;
}

float f(float t) {
    float result = 0.0;
    if (t > 0.008856)
        result = pow(t,(1.0/3.0));
    else
        result = 7.787 * t + (16/116);
    return result;
}

void rgb2lab(Mat & rgbmat) {

    float mat[9] = {
        0.4361, 0.3851, 0.1431,
        0.2225, 0.7169, 0.0606, 
        0.0139, 0.0971, 0.7141};

    //Sacred by hand
    //float mat[9] = {
    //    0.436113, 0.3851462, 0.14303300, 
    //    0.2224733, 0.716878, 0.060600, 
    //    0.013879, 0.09710, 0.7141};

    Mat M(3,3,CV_32FC1,&mat);

    int i = 0;
    for ( int i = 0; i < rgbmat.cols;i++) {
        vector<float> rgb;
        rgb.push_back(invgamma(rgbmat.at<float>(0,i)));
        rgb.push_back(invgamma(rgbmat.at<float>(1,i)));
        rgb.push_back(invgamma(rgbmat.at<float>(2,i)));
        //cout << " Rgb = ";
        //copy(rgb.begin(), rgb.end(), std::ostream_iterator<float>(cout, " "));
        //cout << endl;

        Mat RGB = Mat::zeros(3,1,CV_32FC1);
        RGB.at<float>(0,0) = rgb.at(0);
        RGB.at<float>(1,0) = rgb.at(1);
        RGB.at<float>(2,0) = rgb.at(2);

        Mat XYZ =  M * RGB;
        //cout << "XYZ = " << XYZ;
        float X,Y,Z;
        X = XYZ.at<float>(0,0) / 0.9642;
        Y = XYZ.at<float>(1,0);
        Z = XYZ.at<float>(2,0) / 0.8251;
        //cout << "X = " << X;
        //cout << "Y = " << Y;
        //cout << "Z = " << Z;
        //cout << endl << endl;

        float L,a,b;
        if ( Y > 0.008856)
            L = 116.0 * pow(Y,(1.0/3.0)) - 16.0;
        else
            L = 903.3 * Y;

        a = 500.0 * (f(X) - f(Y));
        b = 200.0 * (f(Y) - f(Z));
        //cout << "Lab = " << L << " " << a << " " << b << endl << endl;

        rgbmat.at<float>(0,i) = L;
        rgbmat.at<float>(1,i) = a;
        rgbmat.at<float>(2,i) = b;
    }
}

float matmax(const Mat &themat) {
    float value = 0.0;
    for (int row = 0; row < themat.rows; row++)
        for (int col = 0; col < themat.cols; col++)
            if (themat.at<float>(row,col) > value)
                value = themat.at<float>(row,col);

    return value;
}

void dist(Mat &blobPos1, Mat &blobPos2, Mat &blobCol1, Mat &blobCol2, float gamma) {

    int lenDATAF = 0;
    int i = 1;
    int j = 2;
    int num1,num2;

    cout << "Printing array results" << endl;
    cout << "blobPos = " << endl << " " << blobPos1 << endl << endl;
    cout << "blobPos = " << endl << " " << blobPos2 << endl << endl;

    cout << "blobPos1 size " << blobPos1.size() << endl;
    cout << "blobPos2 size " << blobPos2.size() << endl;
    
    cout << "Printing array results" << endl;
    cout << "blobCol = " << endl << " " << blobCol1 << endl << endl;
    cout << "blobCol = " << endl << " " << blobCol2 << endl << endl;

    cout << "blobCol1 size " << blobCol1.size() << endl;
    cout << "blobCol2 size " << blobCol2.size() << endl;

    lenDATAF = blobPos1.cols + blobPos2.cols;
    cout << "lenDATAF = " << lenDATAF << endl;

    num1 = blobPos1.cols;
    num2 = blobPos2.cols;

    Mat dist_y = Mat(num2,num1,CV_32FC1);

    // %dist_y = abs( blobPos{1}(ones(num2,1),:)' - blobPos{2}(ones(num1,1),:) )';
    // %dist_y = abs(bsxfun(@minus,(blobPos{1})',blobPos{2}))';
    for(int a = 0; a < num1;a++) {
        for (int b = 0; b <num2; b++) {
            dist_y.at<float>(b,a) = abs(blobPos1.at<float>(0,a) - blobPos2.at<float>(0,b));
        }
    }
    cout << "Dist_y = " << endl << " " << dist_y << endl << endl;

    //%Sxx1 = sum(blobCol{1}.^2,1);
    //Mat Sxx = Mat::zeros(1,num1,CV_32FC1);
    std::vector<float> Sxx(num1);
    for (int a = 0; a < num1; a++)
        for (int b = 0; b < blobCol1.rows; b++)
            Sxx.at(a) = Sxx.at(a) + pow(blobCol1.at<float>(b,a),2);
    //cout << "Sxx = " << endl << " " << Sxx << endl << endl;
    cout << " Sxx = ";
    copy(Sxx.begin(), Sxx.end(), std::ostream_iterator<float>(cout, " "));
    cout << endl;

    //%Sxx1 = sum(blobCol{1}.^2,1);
    //Mat Syy = Mat::zeros(1,num2,CV_32FC1);
    std::vector<float> Syy(num2);
    for (int a = 0; a < num2; a++)
        for (int b = 0; b < blobCol2.rows; b++)
            Syy.at(a) = Syy.at(a) + pow(blobCol2.at<float>(b,a),2);
    //cout << "Syy = " << endl << " " << Syy << endl << endl;
    cout << "Syy = ";
    copy(Syy.begin(), Syy.end(), std::ostream_iterator<float>(cout, " "));
    cout << endl;

    //%Sxy = blobCol{1}' * blobCol{2};
    Mat blobCol1_t = Mat::zeros(blobCol1.cols,blobCol1.rows,CV_32FC1);
    transpose(blobCol1,blobCol1_t);
    Mat Sxy = blobCol1_t * blobCol2;
    //cout << "Sxy = " << endl << " " << Sxy << endl << endl;

    //%dist_color = sqrt( Sxx(ones(num2,1),:)' + Syy(ones(num1,1),:) - 2*Sxy)';
    Mat dist_color = Mat::zeros(num2,num1,CV_32FC1);
    for(int a = 0; a < num1; a++)
        for(int b = 0; b < num2; b++)
            dist_color.at<float>(b,a) = sqrt(Sxx.at(a) + Syy.at(b) - 2*Sxy.at<float>(a,b));

    //cout << "dist_color = " << endl << " " << dist_color << endl << endl;

    // Get an array with the minimum value of each row
    // %ref_y = min(dist_y); 
    //Mat ref_y = Mat(1,num1,CV_32FC1, Scalar(1000));
    std::vector<float> ref_y(num1, 1000);
    for (int a = 0; a < dist_y.cols; a++)
        for (int b = 0; b < dist_y.rows; b++)
            if (dist_y.at<float>(b,a) < ref_y.at(a))
                ref_y.at(a) = dist_y.at<float>(b,a);
    //cout << "ref_y = " << endl << " " << ref_y << endl << endl;
    cout << " ref_y = ";
    copy(ref_y.begin(), ref_y.end(), std::ostream_iterator<float>(cout, " "));
    cout << endl;

    // Get the mean of the min array
    // %me_ref_y = mean(ref_y);
    float sum = std::accumulate(ref_y.begin(),ref_y.end(),0.0);    
    float me_ref_y = sum / ref_y.size();
    //Scalar me_ref_y_sca = mean(ref_y);
    //float me_ref_y = me_ref_y_sca.val[0];
    cout << "Mean = " << me_ref_y << endl;
    //Expected ground trouth = 1.6005

    // Get the standard deviation of the min array
    // MATLAB: std_ref_y = std(ref_y);
    float std_ref_y = 0;
    for (int a = 0; a < ref_y.size(); a++)
        std_ref_y = std_ref_y + pow((ref_y.at(a) - me_ref_y),2);
    std_ref_y = std_ref_y / (ref_y.size() - 1);
    std_ref_y = pow(std_ref_y,0.5);
    cout << "Std = " << std_ref_y << endl;
    // Expected ground trouth = 2.0263

    //Color statistics 
    //%ref_color = min(dist_color); 
    Mat ref_color = Mat(1,dist_color.cols,CV_32FC1,Scalar(1000));
    for (int a = 0; a < dist_color.cols;a++)
        for (int b = 0; b < dist_color.rows; b++)
            if (dist_color.at<float>(b,a) < ref_color.at<float>(0,a))
                ref_color.at<float>(0,a) = dist_color.at<float>(b,a);
    //cout << "ref_color = " << endl << " " << ref_color << endl << endl;

    //%me_ref_color = mean(ref_color); 
    Scalar me_ref_color_sca = mean(ref_color);
    float me_ref_color = me_ref_color_sca.val[0];
    //cout << "Mean color = " << me_ref_color << endl;

    //%std_ref_color = std(ref_color);
    float std_ref_color = 0;
    for (int a = 0; a < ref_color.cols;a++)
        std_ref_color = std_ref_color + pow(ref_color.at<float>(0,a) - me_ref_color,2);
    std_ref_color = std_ref_color / (ref_color.cols - 1);
    std_ref_color = pow(std_ref_color,0.5);
    //cout << "Std color" << std_ref_color << endl;

    //%%%% Good candidate selection 
    //%good = find((ref_y<(me_ref_y+3.5*std_ref_y))&(ref_color<(me_ref_color+3.5*std_ref_color)));
    std::vector<float> good;
    for (int a = 0; a < ref_y.size(); a++)
        if ((ref_y.at(a)<(me_ref_y+3.5*std_ref_y))&(ref_color.at<float>(0,a)<(me_ref_color+3.5*std_ref_color)))
            good.push_back(a); //% Accumulate the index in the good vector 
    //cout << "Good vector" << good << endl << endl;
    //copy(good.begin(), good.end(), std::ostream_iterator<float>(cout, " "));
    
    int max_useful_info = good.size();

    //%dist_y2 = dist_y(:,good);
    Mat dist_y2 = Mat(num2,max_useful_info,CV_32FC1);
    for (int col = 0; col < max_useful_info; col++)
        for (int row = 0; row < num2; row++)
            dist_y2.at<float>(row,col) = dist_y.at<float>(row,good.at(col));
    cout << "Dist_y2 = " << endl << " " << dist_y2 << endl << endl;

    Mat dist_color2 = Mat(num2,max_useful_info,CV_32FC1);
    for (int col = 0; col < max_useful_info; col++)
        for (int row = 0; row < num2; row++)
            dist_color2.at<float>(row,col) = dist_color.at<float>(row,good.at(col));
    cout << "Dist_color2 = " << endl << " " << dist_color2 << endl << endl;
    // Normalize

    float DEN1;
    if (dist_y2.cols == 0) 
        DEN1 = 1;
    else
        DEN1 = matmax(dist_y2);
    Mat dist_y_n = dist_y / DEN1;
    cout << "dist_y_n = " << endl << " " << dist_y_n << endl << endl;

    if (dist_color2.cols == 0)
        DEN1 = 1;
    else 
        DEN1 = matmax(dist_color2);
    Mat dist_color_n = dist_color / DEN1;
    cout << "dist_color_n = " << endl << " " << dist_color_n << endl << endl;

    //Composite distance computation
    Mat totdist_n = Mat(num2,max_useful_info,CV_32FC1);
    for (int col = 0; col< max_useful_info; col++)
        for (int row = 0; row < num2; row++)
            totdist_n.at<float>(row,col) = (gamma*dist_y_n.at<float>(row,good.at(col)) + (1 - gamma) * dist_color_n.at<float>(row,good.at(col)));
    cout << "totdist_n = " << endl << " " << totdist_n << endl << endl;


    //%%Minimization
    //%[unused,matching] = min(totdist_n);
    //tmpmin = 1000 * ones(1,size(totdist_n,2));
    
    //Mat tmpmin = Mat(1,totdist_n.cols,Scalar(1000));
    std::vector<float> tmpmin(totdist_n.cols,1000);
    std::vector<float> matching(totdist_n.cols,1000);
    for (int a = 0; a < totdist_n.cols; a++)
        for (int b = 0; b < totdist_n.rows; b++)
            if (totdist_n.at<float>(b,a) < tmpmin.at(a)){
                tmpmin.at(a) = totdist_n.at<float>(b,a);
                matching.at(a) = b;
            }
    cout << "Matching = ";
    copy(matching.begin(), matching.end(), std::ostream_iterator<float>(cout, " "));
    cout << endl;

    // Compute final distance y
    //%final_dist_y(i,j)  = sum(dist_y2(useful_i))/max_useful_info;
    float final_dist_sum = 0;
    float final_dist_y = 0;
    for (int col = 0; col < max_useful_info; col++)
        final_dist_sum = final_dist_sum + dist_y2.at<float>(matching.at(col),col);
    // Final_dist_y matrix should be passed as argument to this function.
    //final_dist_y.at<float>(i,j) = final_dist_sum / max_useful_info;
    final_dist_y = final_dist_sum / max_useful_info;
    cout << "Final_dist_y = " << final_dist_y << endl << endl;

    // Compute final distance_color

    //%final_dist_color(i,j) = sum(dist_color2(useful_i))/max_useful_info;
    float final_dist_color_sum = 0;
    float final_dist_color = 0;
    for (int col = 0; col < max_useful_info; col++)
        final_dist_color_sum = final_dist_color_sum + dist_color2.at<float>(matching.at(col),col);
    //final_dist_color.at<float>(i,j) = final_dist_color_sum / max_useful_info;
    final_dist_color = final_dist_color_sum / max_useful_info;
    cout << "Final_dist_color = " << final_dist_color << endl << endl;
}

void readmat(const char * filename, Mat& _mat) {

    ifstream infile;
    int rows = 0;
    int cols = 0;

    //check size
    infile.open(filename ,ios::in);
    while(infile)
    {
        string s;
        vector<float> row;
        if(!getline(infile,s)) break;
        istringstream ss(s);
        rows++;

        cols = 0;
        while(ss)
        {
            string s;
            if (!getline(ss,s,',')) break;
            cols++;
        }
    }

    _mat = Mat(rows,cols,CV_32FC1);
    infile.close();
    infile.open(filename,ios::in);
    
    //cout << "Creating matrix = " << rows << "X" << cols << endl << endl;
    rows = 0;
    cols = 0;
    // Fill matrix
    while(infile)
    {
        string s;
        vector<float> row;
        if(!getline(infile,s)) break;
        istringstream ss(s);

        while(ss)
        {
            string s;
            if (!getline(ss,s,',')) break;
            row.push_back((float)atof(s.c_str()));
        }

        for (cols = 0; cols < row.size();cols++)
            _mat.at<float>(rows,cols) = row.at(cols); 

        //copy(row.begin(), row.end(), std::ostream_iterator<float>(cout, " "));
        //cout << endl << endl;
        rows++;

    }
    infile.close();
}

int main( int argc, char** argv)
{
    Mat image;
    image = imread(argv[1],1);

    if (argc != 2 || !image.data)
    {
        printf("No image data\n");
        return -1;
    }

    Mat blobPos_1_;
    readmat("./gtdata/mscrmvec_00001.txt", blobPos_1_);
    //cout << "matrix " << endl << " " << blobPos_1_ << endl << endl;
    Mat blobPos_1(blobPos_1_,cv::Range(2,3),cv::Range(0,39));
    //cout << "blobPos_1 = " << endl << " " << blobPos_1 << endl << endl;

    
    //float sz2[54] = { 27.7,26.2,22.47445,44.625,44.5,30.88889,37.95455,44.72093,30.87097,37.975,
    //    39.33333,33.5625,43.72973,45.6,33.40426,34.89655,37.88235,43.36735,49.65385,
    //    94.25,86.15789,94.11594,77.92308,53,89.19792,65.15789,85.68098,111.3529,
    //    117.4444,53.89655,61,116.5926,114.5217,57.9375,64.47059,52.27778,110.4375,
    //    54.5625,58.58824,56.03401,51.27778,54.10317,65.41176,65.64151,65.64179,
    //    58.32244,124,64.5,51.61905,72.84615,54.12963,69.75,71.68478,75.75069 };
   //Mat blobPos_2(1,54,CV_32FC1,&sz2);

    Mat blobPos_2_;
    readmat("./gtdata/mscrmvec_00002.txt", blobPos_2_);
    Mat blobPos_2(blobPos_2_,cv::Range(2,3),cv::Range(0,54));
    //cout << "blobPos_2 = " << endl << " " << blobPos_2 << endl << endl;

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
    Mat blobCol_1(3,39,CV_32FC1,&sz3);
    cout << "blobCol1 " << endl << " " << blobCol_1 << endl << endl;

    //Mat blobCol_1;
    //readmat("./gtdata/mscrpvec_00001.txt", blobCol_1);
    //rgb2lab(blobCol_1);
    //cout << "bloblCol2" << endl << " " << blobCol_1 << endl << endl;

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
    Mat blobCol_2(3,54,CV_32FC1,&sz4);
    
    //Mat blobCol_2;
    //readmat("./gtdata/mscrpvec_00002.txt", blobCol_2);
    //rgb2lab(blobCol_2);
    //cout << "blobCol_2 " << endl << " " << blobCol_2 << endl << endl;

    dist(blobPos_1,blobPos_2, blobCol_1, blobCol_2, 0.4);

    namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    waitKey(0);
}
