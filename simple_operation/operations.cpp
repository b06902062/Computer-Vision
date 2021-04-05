#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<iostream>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#define rows 512
#define cols 512

using namespace cv;
using namespace std;

int kernel[21][2] = {    {-2, -1},{-2, 0},{-2, 1},
				{-1, -2},{-1, -1},{-1, 0},{-1, 1},{-1, 2},
				{0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2},
				{1, -2}, {1, -1}, {1, 0}, {1, 1}, {1, 2},
						 {2, -1}, {2, 0}, {2, 1} };

bool inrange(int i, int j, int k, int l) {
	if ((i + k) >= 0 && (i + k) < 512 && (j + l) >= 0 && (j + l) < 512)
		return true;
	else
		return false;
}

Mat Dil(Mat img) {
	Mat dil = img.clone();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (img.at<uchar>(i, j) > 0){
				int max=0;
				for (int k = 0; k < 21; k++)
					if (inrange(i, j, kernel[k][0], kernel[k][1]))
						if(img.at<uchar>(i + kernel[k][0], j + kernel[k][1])>max)
							max=img.at<uchar>(i + kernel[k][0], j + kernel[k][1]);
				for (int k = 0; k < 21; k++)
					if (inrange(i, j, kernel[k][0], kernel[k][1]))
						dil.at<uchar>(i + kernel[k][0], j + kernel[k][1])=max;
							
			}
	return dil;
}

Mat Ero(Mat img) {
	Mat ero = img.clone();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (img.at<uchar>(i, j) > 0){
				bool exist=true;
				int min=256;
				for (int k = 0; k < 21; k++){
					if (inrange(i, j, kernel[k][0], kernel[k][1])){
						if(img.at<uchar>(i + kernel[k][0], j + kernel[k][1])==0){
							exist=false;	break;	
						}
						if(img.at<uchar>(i + kernel[k][0], j + kernel[k][1])<min)
							min=img.at<uchar>(i + kernel[k][0], j + kernel[k][1]);
					}
				}
				exist=true;
				for (int k = 0; k < 21; k++){
					if (inrange(i, j, kernel[k][0], kernel[k][1]))
						if(img.at<uchar>(i + kernel[k][0], j + kernel[k][1])==0){
							exist=false;	break;	
						}
					if (inrange(i, j, kernel[k][0], kernel[k][1]) && exist)
						ero.at<uchar>(i + kernel[k][0], j + kernel[k][1])=min;
				}			
			}
	return ero;
}

Mat Open(Mat img){
	return Dil(Ero(img));
}

Mat Close(Mat img){
	return Ero(Dil(img));
}

int main(){
	Mat img = imread("lena.bmp", CV_8UC1);
	Mat g_dil=Dil(img);		imshow("G_Dilation",g_dil);		imwrite("hw4-dil.jpg", g_dil);	waitKey(0);
	Mat g_ero=Ero(img);		imshow("G_Erosion",g_ero);		imwrite("hw4-ero.jpg", g_ero);	waitKey(0);
	Mat g_ope=Open(img);	imshow("G_Open", g_ope);		imwrite("hw4-ope.jpg", g_ope);	waitKey(0);
	Mat g_clo=Close(img);	imshow("G_Close", g_clo);		imwrite("hw4-clo.jpg", g_clo);	waitKey(0);
	
	return 0;
}