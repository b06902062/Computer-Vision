#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdlib.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <random>
#include <vector>
#include <math.h> 

using namespace cv;
using namespace std;

Mat gaussian_noise(Mat img, int amp) {
	Mat gn = img.clone();
	default_random_engine generator;
	normal_distribution<double> distribution(0.0, 1.0);
	for (int i = 0; i < gn.rows; i++)
		for (int j = 0; j < gn.cols; j++) {
			double number = distribution(generator);
			int tmp = gn.at<uchar>(i, j) + amp * number;
			if (tmp <= 0)	gn.at<uchar>(i, j) = 0;
			else if (tmp >= 255)	gn.at<uchar>(i, j) = 255;
			else  gn.at<uchar>(i, j) = tmp;
		}
	return gn;
}

Mat salt_pepper(Mat img, double threshold) {
	Mat sp = img.clone();
	default_random_engine generator;
	uniform_real_distribution<double> distribution(0.0, 1.0);
	for (int i = 0; i < sp.rows; i++)
		for (int j = 0; j < sp.cols; j++) {
			double number = distribution(generator);
			if (number < threshold)
				sp.at<uchar>(i, j) = 0;
			else if (number > 1 - threshold)
				sp.at<uchar>(i, j) = 255;
		}
	return sp;
}

Mat box_filter(Mat img, int filter) {
	Mat tmp = img.clone();
	Mat bf(img.rows - filter + 1, img.cols - filter + 1, CV_8UC1);
	for (int i = 0; i < bf.rows; i++)
		for (int j = 0; j < bf.cols; j++) {
			double mean = 0;
			for (int a = i; a < i + filter; a++)
				for (int b = j; b < j + filter; b++)
					mean += tmp.at<uchar>(a, b);
			mean /= (filter * filter);
			bf.at<uchar>(i, j) = mean;
		}
	return bf;
}

int compare(const void* a, const void* b){	return (*(int*)a - *(int*)b);	}

Mat median_filter(Mat img, int filter) {
	Mat tmp = img.clone();
	Mat mf(img.rows - filter + 1, img.cols - filter + 1, CV_8UC1);
	int array[30], cnt = 0;
	for (int i = 0; i < mf.rows; i++)
		for (int j = 0; j < mf.cols; j++) {
			memset(array, 0, 30*sizeof(int)); cnt = 0;
			for (int a = i; a < i + filter; a++)
				for (int b = j; b < j + filter; b++)
					array[cnt++] = tmp.at<uchar>(a, b);
			
			qsort(array, filter * filter, sizeof(int), compare);
			mf.at<uchar>(i, j) = array[filter * filter/2];
		}
	return mf;
}

int kernel[21][2] = {	 {-2, -1},{-2, 0},{-2, 1},
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
	Mat dil(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < dil.rows; i++)
		for (int j = 0; j < dil.cols; j++){
			int max = 0;
			for (int k = 0; k < 21; k++)
				if (inrange(i, j, kernel[k][0], kernel[k][1]) && img.at<uchar>(i + kernel[k][0], j + kernel[k][1]) > max)
					max = img.at<uchar>(i + kernel[k][0], j + kernel[k][1]);
			for (int k = 0; k < 21; k++)
				if (inrange(i, j, kernel[k][0], kernel[k][1]))
					dil.at<uchar>(i + kernel[k][0], j + kernel[k][1]) = max;
		}
	return dil;
}

Mat Ero(Mat img) {
	Mat ero(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < ero.rows; i++)
		for (int j = 0; j < ero.cols; j++){
			int min = 1024;
			for (int k = 0; k < 21; k++)
				if (inrange(i, j, kernel[k][0], kernel[k][1]) && img.at<uchar>(i + kernel[k][0], j + kernel[k][1]) < min)
					min = img.at<uchar>(i + kernel[k][0], j + kernel[k][1]);
							for (int k = 0; k < 21; k++)
				if (inrange(i, j, kernel[k][0], kernel[k][1]))
					ero.at<uchar>(i + kernel[k][0], j + kernel[k][1]) = min;
		}
	return ero;
}

Mat Open(Mat img) { return Dil(Ero(img)); }

Mat Close(Mat img) { return Ero(Dil(img)); }

double SNR(Mat noise, Mat img, double vs) {
	double vn=0, meu = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			meu += (noise.at<uchar>(i, j) - img.at<uchar>(i, j));
	meu = meu / (img.rows * img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			vn += (noise.at<uchar>(i, j) - img.at<uchar>(i, j) - meu) * (noise.at<uchar>(i, j) - img.at<uchar>(i, j) - meu);
	vn = vn / (img.rows * img.cols);
	vn = sqrt(vn);
	return 20 * log10(vs / vn);
}

int main() {
	Mat img = imread("lena.bmp", CV_8UC1);
	double vs=0, meu=0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			meu += img.at<uchar>(i, j);
	meu = meu/(img.rows * img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			vs += (img.at<uchar>(i, j) - meu) * (img.at<uchar>(i, j) - meu);
	vs = vs/(img.rows * img.cols);
	vs = sqrt(vs);

	Mat gn10 = gaussian_noise(img, 10);	imshow("gn10", gn10);	imwrite("./task1/gn10.jpg", gn10);
	Mat gn30 = gaussian_noise(img, 30);	imshow("gn30", gn30);	imwrite("./task1/gn30.jpg", gn30);
	Mat sp5 = salt_pepper(img, 0.05);	imshow("sp5", sp5);		imwrite("./task2/sp5.jpg", sp5);
	Mat sp1 = salt_pepper(img, 0.1);	imshow("sp1", sp1);		imwrite("./task2/sp1.jpg", sp1);	waitKey(0);

	Mat box_3_gn10 = box_filter(gn10, 3);	imshow("box_3_gn10", box_3_gn10);	imwrite("./task3/box_3_gn10.jpg", box_3_gn10);
	Mat box_3_gn30 = box_filter(gn30, 3);	imshow("box_3_gn30", box_3_gn30);	imwrite("./task3/box_3_gn30.jpg", box_3_gn30);
	Mat box_3_sp5 = box_filter(sp5, 3);		imshow("box_3_sp5", box_3_sp5);		imwrite("./task3/box_3_sp5.jpg", box_3_sp5);
	Mat box_3_sp1 = box_filter(sp1, 3);		imshow("box_3_sp1", box_3_sp1);		imwrite("./task3/box_3_sp1.jpg", box_3_sp1);	waitKey(0);

	Mat box_5_gn10 = box_filter(gn10, 5);	imshow("box_5_gn10", box_5_gn10);	imwrite("./task3/box_5_gn10.jpg", box_5_gn10);
	Mat box_5_gn30 = box_filter(gn30, 5);	imshow("box_5_gn30", box_5_gn30);	imwrite("./task3/box_5_gn30.jpg", box_5_gn30);
	Mat box_5_sp5 = box_filter(sp5, 5);		imshow("box_5_sp5", box_5_sp5);		imwrite("./task3/box_5_sp5.jpg", box_5_sp5);
	Mat box_5_sp1 = box_filter(sp1, 5);		imshow("box_5_sp1", box_5_sp1);		imwrite("./task3/box_5_sp1.jpg", box_5_sp1);	waitKey(0);

	Mat median_3_gn10 = median_filter(gn10, 3);	imshow("median_3_gn10", median_3_gn10);	imwrite("./task4/median_3_gn10.jpg", median_3_gn10);
	Mat median_3_gn30 = median_filter(gn30, 3);	imshow("median_3_gn30", median_3_gn30);	imwrite("./task4/median_3_gn30.jpg", median_3_gn30);
	Mat median_3_sp5 = median_filter(sp5, 3);	imshow("median_3_sp5", median_3_sp5);	imwrite("./task4/median_3_sp5.jpg", median_3_sp5);
	Mat median_3_sp1 = median_filter(sp1, 3);	imshow("median_3_sp1", median_3_sp1);	imwrite("./task4/median_3_sp1.jpg", median_3_sp1);	waitKey(0);

	Mat median_5_gn10 = median_filter(gn10, 5);	imshow("median_5_gn10", median_5_gn10);	imwrite("./task4/median_5_gn10.jpg", median_5_gn10);
	Mat median_5_gn30 = median_filter(gn30, 5);	imshow("median_5_gn30", median_5_gn30);	imwrite("./task4/median_5_gn30.jpg", median_5_gn30);
	Mat median_5_sp5 = median_filter(sp5, 5);	imshow("median_5_sp5", median_5_sp5);	imwrite("./task4/median_5_sp5.jpg", median_5_sp5);
	Mat median_5_sp1 = median_filter(sp1, 5);	imshow("median_5_sp1", median_5_sp1);	imwrite("./task4/median_5_sp1.jpg", median_5_sp1);	waitKey(0);

	Mat cl_op_gn10 = Open(Close(gn10));	imshow("cl_op_gn10", cl_op_gn10);	imwrite("./task5/cl_op_gn10.jpg", cl_op_gn10);
	Mat cl_op_gn30 = Open(Close(gn30));	imshow("cl_op_gn30", cl_op_gn30);	imwrite("./task5/cl_op_gn30.jpg", cl_op_gn30);
	Mat cl_op_sp5 = Open(Close(sp5));	imshow("cl_op_sp5", cl_op_sp5);		imwrite("./task5/cl_op_sp5.jpg", cl_op_sp5);
	Mat cl_op_sp1 = Open(Close(sp1));	imshow("cl_op_sp1", cl_op_sp1);		imwrite("./task5/cl_op_sp1.jpg", cl_op_sp1);	waitKey(0);

	Mat op_cl_gn10 = Close(Open(gn10));	imshow("op_cl_gn10", op_cl_gn10);	imwrite("./task5/op_cl_gn10.jpg", op_cl_gn10);
	Mat op_cl_gn30 = Close(Open(gn30));	imshow("op_cl_gn30", op_cl_gn30);	imwrite("./task5/op_cl_gn30.jpg", op_cl_gn30);
	Mat op_cl_sp5 = Close(Open(sp5));	imshow("op_cl_sp5", op_cl_sp5);		imwrite("./task5/op_cl_sp5.jpg", op_cl_sp5);
	Mat op_cl_sp1 = Close(Open(sp1));	imshow("op_cl_sp1", op_cl_sp1);		imwrite("./task5/op_cl_sp1.jpg", op_cl_sp1);	waitKey(0);

	cout << "gn10 " << SNR(gn10, img, vs) << endl;
	cout << "gn30 " << SNR(gn30, img, vs) << endl;
	cout << "sp5 " << SNR(sp5, img, vs) << endl;
	cout << "sp1 " << SNR(sp1, img, vs) << endl;

	cout << "box_3_gn10 " << SNR(box_3_gn10, img, vs) << endl;
	cout << "box_3_gn30 " << SNR(box_3_gn30, img, vs) << endl;
	cout << "box_3_sp5 " << SNR(box_3_sp5, img, vs) << endl;
	cout << "box_3_sp1 " << SNR(box_3_sp1, img, vs) << endl;

	cout << "box_5_gn10 " << SNR(box_5_gn10, img, vs) << endl;
	cout << "box_5_gn30 " << SNR(box_5_gn30, img, vs) << endl;
	cout << "box_5_sp5 " << SNR(box_5_sp5, img, vs) << endl;
	cout << "box_5_sp1 " << SNR(box_5_sp1, img, vs) << endl;

	cout << "median_3_gn10 " << SNR(median_3_gn10, img, vs) << endl;
	cout << "median_3_gn30 " << SNR(median_3_gn30, img, vs) << endl;
	cout << "median_3_sp5 " << SNR(median_3_sp5, img, vs) << endl;
	cout << "median_3_sp1 " << SNR(median_3_sp1, img, vs) << endl;

	cout << "median_5_gn10 " << SNR(median_5_gn10, img, vs) << endl;
	cout << "median_5_gn30 " << SNR(median_5_gn30, img, vs) << endl;
	cout << "median_5_sp5 " << SNR(median_5_sp5, img, vs) << endl;
	cout << "median_5_sp1 " << SNR(median_5_sp1, img, vs) << endl;

	cout << "cl_op_gn10 " << SNR(cl_op_gn10, img, vs) << endl;
	cout << "cl_op_gn30 " << SNR(cl_op_gn30, img, vs) << endl;
	cout << "cl_op_sp5 " << SNR(cl_op_sp5, img, vs) << endl;
	cout << "cl_op_sp1 " << SNR(cl_op_sp1, img, vs) << endl;

	cout << "op_cl_gn10 " << SNR(op_cl_gn10, img, vs) << endl;
	cout << "op_cl_gn30 " << SNR(op_cl_gn30, img, vs) << endl;
	cout << "op_cl_sp5 " << SNR(op_cl_sp5, img, vs) << endl;
	cout << "op_cl_sp1 " << SNR(op_cl_sp1, img, vs) << endl;
	
	imshow("img", img);	waitKey(0);

	return 0;
}

