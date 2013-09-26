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
#include <ctime>
//Get groundtruth matrixes
#include "cialabcolormatrixesgt.h"

using namespace cv;
using namespace std;

class MSCR {
public:
    Mat blobPos;
    Mat blobColor;
};

float invgamma(float Rp) {
    float R = 0.0;
    if ( Rp <= 0.04045 )
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
        result = 7.787 * t + (16.0/116.0);
    return result;
}

void rgb2lab(Mat & rgbmat) {

    float mat[9] = {
        0.4361, 0.3851, 0.1431,
        0.2225, 0.7169, 0.0606, 
        0.0139, 0.0971, 0.7141};

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

//void dist(Mat &blobPos1, Mat &blobPos2, Mat &blobCol1, Mat &blobCol2, float gamma) {
void dist(vector<MSCR>& mscrvec, Mat& _final_dist_y, Mat& _final_dist_color,float gamma) {

    int lenDATAF = 0;
    int i = 1;
    int j = 2;
    int num1,num2;

    lenDATAF = mscrvec.size();

    //Initialize final matrices
    _final_dist_y = Mat::ones(lenDATAF,lenDATAF,CV_32FC1)*1000.;
    _final_dist_color = Mat::ones(lenDATAF,lenDATAF,CV_32FC1)*1000.;
    
    //put diagonal to 0
    for (int dig = 0; dig < lenDATAF; dig++) {
        _final_dist_y.at<float>(dig,dig) = 0;
        _final_dist_color.at<float>(dig,dig) = 0;
    }

    //cout  << "Finaldisty = " << endl << _final_dist_y << endl;
    //cout  << "Finaldistcolor = " << endl << _final_dist_color << endl;

    for (int gals_index = 0; gals_index < mscrvec.size(); gals_index += 2) {

        Mat &blobPos1 = mscrvec.at(gals_index).blobPos;
        num1 = blobPos1.cols;
        Mat &blobCol1 = mscrvec.at(gals_index).blobColor;

        for (int probs_index = 1; probs_index < mscrvec.size(); probs_index += 2) {

            Mat &blobPos2 = mscrvec.at(probs_index).blobPos;
            num2 = blobPos2.cols;
            Mat &blobCol2 = mscrvec.at(probs_index).blobColor;

            Mat dist_y = Mat(num2,num1,CV_32FC1);

            // %dist_y = abs( blobPos{1}(ones(num2,1),:)' - blobPos{2}(ones(num1,1),:) )';
            // %dist_y = abs(bsxfun(@minus,(blobPos{1})',blobPos{2}))';
            for(int a = 0; a < num1;a++) {
                for (int b = 0; b <num2; b++) {
                    dist_y.at<float>(b,a) = abs(blobPos1.at<float>(0,a) - blobPos2.at<float>(0,b));
                }
            }
            //cout << "Dist_y = " << endl << " " << dist_y << endl << endl;

            //%Sxx1 = sum(blobCol{1}.^2,1);
            //Mat Sxx = Mat::zeros(1,num1,CV_32FC1);
            std::vector<float> Sxx(num1);
            for (int a = 0; a < num1; a++)
                for (int b = 0; b < blobCol1.rows; b++)
                    Sxx.at(a) = Sxx.at(a) + pow(blobCol1.at<float>(b,a),2);
            //cout << "Sxx = " << endl << " " << Sxx << endl << endl;
            //cout << " Sxx = ";
            //copy(Sxx.begin(), Sxx.end(), std::ostream_iterator<float>(cout, " "));
            //cout << endl;

            //%Sxx1 = sum(blobCol{1}.^2,1);
            //Mat Syy = Mat::zeros(1,num2,CV_32FC1);
            std::vector<float> Syy(num2);
            for (int a = 0; a < num2; a++)
                for (int b = 0; b < blobCol2.rows; b++)
                    Syy.at(a) = Syy.at(a) + pow(blobCol2.at<float>(b,a),2);
            //cout << "Syy = " << endl << " " << Syy << endl << endl;
            //cout << "Syy = ";
            //copy(Syy.begin(), Syy.end(), std::ostream_iterator<float>(cout, " "));
            //cout << endl;

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
            //cout << " ref_y = ";
            //copy(ref_y.begin(), ref_y.end(), std::ostream_iterator<float>(cout, " "));
            //cout << endl;

            // Get the mean of the min array
            // %me_ref_y = mean(ref_y);
            float sum = std::accumulate(ref_y.begin(),ref_y.end(),0.0);    
            float me_ref_y = sum / ref_y.size();
            //Scalar me_ref_y_sca = mean(ref_y);
            //float me_ref_y = me_ref_y_sca.val[0];
            //cout << "Mean = " << me_ref_y << endl;
            //Expected ground trouth = 1.6005

            // Get the standard deviation of the min array
            // MATLAB: std_ref_y = std(ref_y);
            float std_ref_y = 0;
            for (int a = 0; a < ref_y.size(); a++)
                std_ref_y = std_ref_y + pow((ref_y.at(a) - me_ref_y),2);
            std_ref_y = std_ref_y / (ref_y.size() - 1);
            std_ref_y = pow(std_ref_y,0.5);
            //cout << "Std = " << std_ref_y << endl;
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
            //cout << "Dist_y2 = " << endl << " " << dist_y2 << endl << endl;

            Mat dist_color2 = Mat(num2,max_useful_info,CV_32FC1);
            for (int col = 0; col < max_useful_info; col++)
                for (int row = 0; row < num2; row++)
                    dist_color2.at<float>(row,col) = dist_color.at<float>(row,good.at(col));
            //cout << "Dist_color2 = " << endl << " " << dist_color2 << endl << endl;
            // Normalize

            float DEN1;
            if (dist_y2.cols == 0) 
                DEN1 = 1;
            else
                DEN1 = matmax(dist_y2);
            Mat dist_y_n = dist_y / DEN1;
            //cout << "dist_y_n = " << endl << " " << dist_y_n << endl << endl;

            if (dist_color2.cols == 0)
                DEN1 = 1;
            else 
                DEN1 = matmax(dist_color2);
            Mat dist_color_n = dist_color / DEN1;
            //cout << "dist_color_n = " << endl << " " << dist_color_n << endl << endl;

            //Composite distance computation
            Mat totdist_n = Mat(num2,max_useful_info,CV_32FC1);
            for (int col = 0; col< max_useful_info; col++)
                for (int row = 0; row < num2; row++)
                    totdist_n.at<float>(row,col) = (gamma*dist_y_n.at<float>(row,good.at(col)) + (1 - gamma) * dist_color_n.at<float>(row,good.at(col)));
            //cout << "totdist_n = " << endl << " " << totdist_n << endl << endl;


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
            //cout << "Matching = ";
            //copy(matching.begin(), matching.end(), std::ostream_iterator<float>(cout, " "));
            //cout << endl;

            // Compute final distance y
            //%final_dist_y(i,j)  = sum(dist_y2(useful_i))/max_useful_info;
            float final_dist_sum = 0;
            //float final_dist_y = 0;
            for (int col = 0; col < max_useful_info; col++)
                final_dist_sum = final_dist_sum + dist_y2.at<float>(matching.at(col),col);
            // Final_dist_y matrix should be passed as argument to this function.
            _final_dist_y.at<float>(gals_index,probs_index) = final_dist_sum / max_useful_info;
            //final_dist_y = final_dist_sum / max_useful_info;
            //cout << "Final_dist_y = " << final_dist_y << endl << endl;

            // Compute final distance_color

            //%final_dist_color(i,j) = sum(dist_color2(useful_i))/max_useful_info;
            float final_dist_color_sum = 0;
            //float final_dist_color = 0;
            for (int col = 0; col < max_useful_info; col++)
                final_dist_color_sum = final_dist_color_sum + dist_color2.at<float>(matching.at(col),col);
            _final_dist_color.at<float>(gals_index,probs_index) = final_dist_color_sum / max_useful_info;
            //final_dist_color = final_dist_color_sum / max_useful_info;
            //fprintf(stdout,"(%d %d) %.20f %.20f \n",gals_index+1,probs_index+1,final_dist_y,final_dist_color);
            //cout << "Final_dist_color = " << final_dist_color << endl << endl;
        }
    }
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

    vector<MSCR> mscr_vec(100);

    for (int i = 0; i < 100; i++) {
        char buf[100];
        sprintf(buf,"./gtdata/mscrmvec_%05d.txt",i+1);
        Mat blobPos_1_;
        readmat(buf, blobPos_1_);
        Mat blobPos_1(blobPos_1_,cv::Range(2,3),cv::Range(0,blobPos_1_.cols));
        mscr_vec.at(i).blobPos = blobPos_1.clone();

        sprintf(buf,"./gtdata/mscrpvec_%05d.txt",i+1);
        readmat(buf, mscr_vec.at(i).blobColor);
        rgb2lab(mscr_vec.at(i).blobColor);
    }

    //dist(blobPos_1,blobPos_2, blobCol_1, blobCol_2, 0.4);
    //dist(mscr_vec.at(0).blobPos,
    //     mscr_vec.at(1).blobPos, 
    //     mscr_vec.at(0).blobColor, 
    //     mscr_vec.at(1).blobColor,0.4);
    Mat final_dist_y;
    Mat final_dist_color;
    clock_t t0 = clock();
    dist(mscr_vec, final_dist_y, final_dist_color, 0.4);
    float elapsed = float(clock() - t0) / float(CLOCKS_PER_SEC);
    cout << "Time = " << elapsed << endl;
    //cout  << "Finaldisty = " << endl << final_dist_y << endl;
    //cout  << "Finaldistcolor = " << endl << final_dist_color << endl;

    namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    waitKey(0);
}
